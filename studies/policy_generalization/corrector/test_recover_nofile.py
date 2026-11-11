"""文件描述符继承与限定错误恢复；不运行远程训练。"""
import json
from pathlib import Path
import shutil
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import corrector_recover_nofile as fix


class RecoverNofileTests(unittest.TestCase):
    def test_real_child_inherits_restored_limit(self):
        source=str(Path(fix.__file__).parent)
        code="import sys,resource,json;sys.path.insert(0,"+repr(source)+");s,h=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(1024,h));from corrector_recover_nofile import raise_nofile;print(json.dumps(raise_nofile()))"
        result=json.loads(subprocess.check_output([sys.executable,'-c',code],text=True))
        self.assertGreaterEqual(result['after'][0],8192);self.assertEqual(result['after'],result['child'])

    def test_raise_soft_limit_and_verify_child(self):
        with patch.object(fix.resource,'getrlimit',side_effect=[(1024,65536),(1024,65536),(32768,65536)]), \
             patch.object(fix.resource,'setrlimit') as setter, \
             patch.object(fix.subprocess,'check_output',return_value='[32768,65536]'):
            result=fix.raise_nofile()
        setter.assert_called_once_with(fix.resource.RLIMIT_NOFILE,(32768,65536))
        self.assertEqual(result['child'][0],32768)

    def test_hard_limit_not_raiseable_but_8192_or_more_is_usable(self):
        with patch.object(fix.resource,'getrlimit',side_effect=[(1024,16384),(1024,16384),(16384,16384)]), \
             patch.object(fix.resource,'setrlimit',side_effect=[PermissionError(),None]), \
             patch.object(fix.subprocess,'check_output',return_value='[16384,16384]'):
            self.assertEqual(fix.raise_nofile()['after'][0],16384)

    def test_small_hard_limit_stops_before_modifying_experiment(self):
        with patch.object(fix.resource,'getrlimit',return_value=(1024,1024)), \
             patch.object(fix.resource,'setrlimit',side_effect=PermissionError()), \
             patch.object(fix.subprocess,'check_output') as child:
            with self.assertRaises(RuntimeError):fix.raise_nofile()
            child.assert_not_called()

    def fixture(self,root):
        guardian=root/'guardian';guardian.mkdir();code=root/'guardian_code';code.mkdir()
        shutil.copyfile(Path(fix.__file__).with_name('corrector_guardian.py'),code/'corrector_guardian.py')
        (code/'guardian_watch.sh').write_text('# frozen watch')
        original=root/'original';followup=root/'followup'
        oldcode=root/'oldcode';oldcode.mkdir();newcode=root/'newcode';newcode.mkdir()
        for folder in (oldcode,newcode):(folder/'train.py').write_text('# frozen trainer')
        settings={'original':str(original),'original_code':str(oldcode),'followup':str(followup),
                  'followup_code':str(newcode),'output':str(guardian),'hours':36}
        fix.write(guardian/'guardian_plan.json',{'source_sha256':fix.sha(code/'corrector_guardian.py'),
            'settings':settings,'deadline_unix':12345678900})
        for folder,name,c in [(original,'sweep_plan.json',oldcode),(followup,'followup_plan.json',newcode)]:
            fix.write(folder/name,{'code_sha256':{'train.py':fix.sha(c/'train.py')}})
        job='transformer-h1_s43';run=original/'runs'/job;run.mkdir(parents=True)
        (run/'latest.pt').write_bytes(b'complete prior epoch checkpoint')
        fix.write(run/'protocol.json',{'seed':43})
        log=original/'logs/train.log';log.parent.mkdir();log.write_text('OSError: [Errno 24] Too many open files')
        fix.write(original/'sweep_status.json',{'status':'error','phase':job+'_training',
            'job':{'name':job},'log':str(log),'error':'training exit 1'})
        fix.write(followup/'followup_status.json',{'status':'error','error':'nested error'})
        fix.write(guardian/'guardian_status.json',{'status':'error','error':'upstream error'})
        return guardian,code,original,run,log

    def test_confirmed_error_backed_up_resume_state_and_deadline_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            guardian,code,original,run,log=self.fixture(Path(tmp))
            sha=fix.sha(run/'latest.pt');plan=(guardian/'guardian_plan.json').read_bytes()
            command=fix.prepare(guardian,code,{'after':[32768,65536]})
            self.assertEqual(command[:2],['bash',str(code/'guardian_watch.sh')])
            self.assertEqual(fix.read(original/'sweep_status.json')['status'],'paused')
            self.assertEqual(fix.sha(run/'latest.pt'),sha);self.assertEqual((guardian/'guardian_plan.json').read_bytes(),plan)
            self.assertTrue((guardian/'nofile_recovery.json').exists())
            self.assertTrue(list((guardian/'recovery_events').glob('nofile_*/0_sweep_status.json')))

    def test_other_error_does_not_reset_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            guardian,code,original,run,log=self.fixture(Path(tmp));log.write_text('ValueError: shape mismatch')
            with self.assertRaises(RuntimeError):fix.prepare(guardian,code,{})
            self.assertEqual(fix.read(original/'sweep_status.json')['status'],'error')
            self.assertFalse((guardian/'nofile_recovery.json').exists())

    def test_missing_latest_or_changed_source_refuses_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            guardian,code,original,run,log=self.fixture(Path(tmp));(run/'latest.pt').unlink()
            with self.assertRaises(RuntimeError):fix.prepare(guardian,code,{})
            self.assertEqual(fix.read(original/'sweep_status.json')['status'],'error')


if __name__=='__main__':unittest.main()
