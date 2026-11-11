"""36h监督、无损续训状态转存和配对统计；不启动远程任务。"""
import json
import os
import subprocess
import zipfile
import contextlib
import io
import package_corrector_guardian as package
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import corrector_guardian as g


class GuardianTests(unittest.TestCase):
    def finished_run(self,root):
        root.mkdir();(root/'latest.pt').write_bytes(b'full optimizer+weights+rng')
        (root/'best_model.pt').write_bytes(b'best model')
        g.write(root/'JOB_COMPLETE.json',{'checkpoint_sha256':g.digest(root/'best_model.pt')})
        g.write(root/'result.json',{'checkpoint_reload_verified':True})
        g.write(root/'training_status.json',{'status':'completed'})

    def test_tiering_is_lossless_original_path_still_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';self.finished_run(run)
            sha=g.digest(run/'latest.pt');target=root/'system/run/latest.pt'
            self.assertTrue(g.tier_checkpoint(run,target,root/'journal.json'))
            self.assertTrue((run/'latest.pt').is_symlink());self.assertEqual(g.digest(run/'latest.pt'),sha)
            self.assertEqual(g.digest(target),sha)
            self.assertFalse(g.tier_checkpoint(run,target,root/'journal.json'))

    def test_active_or_changed_job_never_relocated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';self.finished_run(run)
            (run/'JOB_COMPLETE.json').unlink()
            self.assertFalse(g.tier_checkpoint(run,root/'archive',root/'journal'))
            self.assertFalse((run/'latest.pt').is_symlink())
            g.write(run/'JOB_COMPLETE.json',{'checkpoint_sha256':'wrong'})
            with self.assertRaises(RuntimeError):g.tier_checkpoint(run,root/'archive',root/'journal')

    def test_different_archive_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';self.finished_run(run)
            target=root/'archive';target.write_bytes(b'other contents')
            with self.assertRaises(RuntimeError):g.tier_checkpoint(run,target,root/'journal')
            self.assertEqual(target.read_bytes(),b'other contents');self.assertFalse((run/'latest.pt').is_symlink())

    def test_resume_exact_frozen_followup_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code=root/'code';code.mkdir();script=code/'corrector_followup_queue.py';script.write_text('# frozen')
            output=root/'output'
            g.write(output/'followup_plan.json',{'code_sha256':{script.name:g.digest(script)},
                 'settings':{'models':['transformer-a10-h1','transformer-a10-h3'],'seeds':[42,43,44],'no_eval':False,'hours':None}})
            cmd=g.reconstruct(code,output,12)
            self.assertIn(str(script),cmd);self.assertEqual(cmd[-1],'12');self.assertNotIn('--no-eval',cmd)
            script.write_text('# changed')
            with self.assertRaises(RuntimeError):g.reconstruct(code,output,12)

    def test_paired_summary_uses_condition_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for index,(condition,method,success) in enumerate([('t0_i3','H50',False),('t0_i3','ogg',True),('t0_i4','H50',True),('t0_i4','ogg',True)]):
                g.write(root/'trials'/f'{index}.json',{'case':{'condition':condition,'method':method,'seed':1000},
                    'status':'completed','result':{'success':success}})
            g.paired_summary(root);result=g.read(root/'analysis/paired_discordance.json')['pairs']['H50_vs_ogg']
            self.assertEqual(result['matched'],2);self.assertEqual(result['ogg_only'],1);self.assertEqual(result['both_success'],1)

    def test_deadline_survives_restart_and_only_explicit_extension_changes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=Namespace(output=Path(tmp),mode='run',hours=36,extend_hours=0)
            with patch.object(g.time,'time',return_value=1000),patch.object(g.Guardian,'remaining',return_value=0):
                g.Guardian(args).run()
            deadline=g.read(args.output/'guardian_plan.json')['deadline_unix']
            self.assertEqual(deadline,1000+36*3600)
            with patch.object(g.time,'time',return_value=2000),patch.object(g.Guardian,'remaining',return_value=0):
                g.Guardian(args).run()
            self.assertEqual(g.read(args.output/'guardian_plan.json')['deadline_unix'],deadline)
            args.extend_hours=1
            with patch.object(g.time,'time',return_value=2000),patch.object(g.Guardian,'remaining',return_value=0):
                g.Guardian(args).run()
            self.assertEqual(g.read(args.output/'guardian_plan.json')['deadline_unix'],deadline+3600)

    def test_only_saved_latest_process_interruption_is_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=Namespace(output=root/'guardian',original=root/'original',original_code=root/'original_code',
                followup=root/'followup',followup_code=root/'followup_code')
            args.output.mkdir();guardian=g.Guardian(args);guardian.deadline=10**12
            run=args.original/'runs/test';run.mkdir(parents=True);(run/'latest.pt').write_bytes(b'resumable state')
            g.write(args.original/'sweep_status.json',{'status':'error','error':'RuntimeError: training退出码-9','job':{'name':'test'}})
            g.write(args.followup/'followup_status.json',{'status':'error','error':'nested training failed'})
            with patch.object(g,'active',return_value=False),patch.object(g,'reconstruct',return_value=[]):
                self.assertTrue(guardian.recover_interrupted_pipeline())
                self.assertEqual(g.read(args.original/'sweep_status.json')['status'],'paused')
                g.write(args.original/'sweep_status.json',{'status':'error','error':'ValueError: shape mismatch','job':{'name':'test'}})
                self.assertFalse(guardian.recover_interrupted_pipeline())
            self.assertTrue((args.output/'recovery_events/recover_original_1.json').exists())

    def test_watchdog_retries_only_interrupts_and_does_not_reextend_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):package.build(root/'bundle.zip')
            fake=root/'python';fake.write_text('#!/usr/bin/env bash\necho "$*" >> "$GUARD_TEST_COUNTER"\nif [[ "$GUARD_TEST_MODE" == fatal ]]; then exit 1; fi\nif [[ "$GUARD_TEST_MODE" == always ]]; then exit 137; fi\nif [[ $(wc -l < "$GUARD_TEST_COUNTER") == 1 ]]; then exit 137; fi\nexit 0\n');fake.chmod(0o755)
            with zipfile.ZipFile(root/'bundle.zip') as z:watch=z.read('guardian_watch.sh').decode()
            watch=watch.replace('/root/autodl-tmp/corrector/bootstrap/bin/python',str(fake)).replace('sleep 60','sleep 0')
            script=root/'watch.sh';script.write_text(watch)
            for mode,count,code in [('once',2,0),('fatal',1,1),('always',3,137)]:
                counter=root/(mode+'.txt')
                result=subprocess.run(['bash',str(script),'--hours','36','--extend-hours','2'],
                    env={**os.environ,'GUARD_TEST_COUNTER':str(counter),'GUARD_TEST_MODE':mode},capture_output=True)
                lines=counter.read_text().splitlines()
                self.assertEqual(len(lines),count);self.assertEqual(result.returncode,code)
                if count>1:self.assertNotIn('--extend-hours',lines[1])

    def test_existing_lock_means_wait_not_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with g.lock(root/'.followup.lock'):self.assertTrue(g.active(root,'followup'))
            self.assertFalse(g.active(root,'followup'))


if __name__=='__main__':unittest.main()
