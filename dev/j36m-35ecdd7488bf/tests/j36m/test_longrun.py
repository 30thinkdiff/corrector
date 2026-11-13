import json,sys,tempfile,unittest,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import corrector_j36m_longrun as q

class Longrun(unittest.TestCase):
    def test_frozen_fifteen_jobs_match_budget_and_two_disks(self):
        work=q.jobs(Path('/root/supervision'),Path('/code'),'/python',Path('/root/autodl-tmp/new-study'))
        self.assertEqual(len(work),23)
        train=[j for j in work if j['mode']=='train'];self.assertEqual(len(train),15)
        self.assertEqual(train[0]['name'],'J_s42')
        self.assertEqual({j['name'] for j in train},{v+'_s'+str(s) for s in (42,43,44) for v in ('J','C0','J_equal','S2','S8')})
        for j in train:
            cmd=j['command'];self.assertEqual(cmd[cmd.index('--epochs')+1],'60');self.assertEqual(cmd[cmd.index('--seed')+1],j['name'][-2:])
            self.assertIn('--precision',cmd)
            self.assertEqual(j['output'].startswith('/root/supervision'),j['name'].endswith('44'))
    def test_unknown_errors_never_automatically_retry(self):
        self.assertEqual(q.classify('ValueError SHA mismatch'),'stop_unknown_error')
        self.assertEqual(q.classify('CUDA out of memory'),'stop_unknown_error')
        self.assertEqual(q.classify('ResourcePause: /root free below 8GiB'),'resources')
        self.assertEqual(q.classify('OSError: [Errno 24] Too many open files'),'retry_checkpoint')
    def test_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'one.py').write_text('source')
            manifest={'files':{'one.py':q.digest(p/'one.py')}};q.write(p/'DEPLOYMENT_MANIFEST.json',manifest)
            q.verify_code(p,manifest);(p/'one.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'source changed'):q.verify_code(p,manifest)
    def test_completed_model_requires_reload_and_all_epochs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';j={'artifact':str(p),'mode':'train'}
            self.assertFalse(q.artifact_valid(j))
            q.write(p,{'status':'completed','epochs_completed':59,'checkpoint_reload_verified':True})
            self.assertFalse(q.artifact_valid(j));q.write(p,{'status':'completed','epochs_completed':60,'checkpoint_reload_verified':True})
            self.assertTrue(q.artifact_valid(j))

if __name__=='__main__':unittest.main()
