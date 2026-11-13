import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_eval_health import inspect,process_matches


class HealthTests(unittest.TestCase):
    def test_pid_reuse_does_not_make_stale_running_live(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'experiment';root.mkdir();proc=Path(t)/'proc';(proc/'100').mkdir(parents=True)
            (proc/'100/cmdline').write_bytes(b'python\0unrelated.py\0')
            (root/'queue_status.json').write_text(json.dumps({'status':'running','pid':100,'phase':'plus_evaluation'}))
            d=inspect(root,proc);self.assertEqual(d['queue']['status'],'interrupted_process_missing')
            self.assertEqual(json.loads((root/'queue_status.json').read_text())['status'],'running')
    def test_live_exact_script_and_output_required(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'experiment';root.mkdir();proc=Path(t)/'proc';(proc/'100').mkdir(parents=True)
            (proc/'100/cmdline').write_bytes(('python\0/code/corrector_plus_queue.py\0--output\0'+str(root)+'\0').encode())
            self.assertTrue(process_matches(100,'corrector_plus_queue.py',root,proc))
            self.assertFalse(process_matches(100,'corrector_plus_queue.py',root/'other',proc))
            self.assertFalse(process_matches(True,'corrector_plus_queue.py',root,proc))
    def test_completed_results_remain_completed_after_process_exit(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'queue_status.json').write_text(json.dumps({'status':'completed','pid':100}))
            self.assertEqual(inspect(root,root/'missing_proc')['queue']['status'],'completed')
