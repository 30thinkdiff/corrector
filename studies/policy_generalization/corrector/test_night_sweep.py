"""无人值守队列的计划、恢复和OOM预检；模拟结果不代表GPU实测。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('night_sweep',ROOT/'scripts/corrector_night_sweep.py')
night=importlib.util.module_from_spec(spec);spec.loader.exec_module(night)


class NightSweepTests(unittest.TestCase):
    def args(self,root):
        return night.parser().parse_args(['run','--output',str(root/'night'),'--runtime',str(root/'runtime'),
                                         '--archive',str(root/'archive')])

    def test_default_queue_is_finite_and_preserves_common_anchors(self):
        args=self.args(Path('/tmp/test_night_sweep'))
        jobs=night.specifications(args)
        self.assertEqual(len(jobs),9)
        self.assertEqual({j['seed'] for j in jobs},{42,43,44})
        self.assertEqual(jobs[0]['variant'],'transformer-h1')
        for job in jobs:
            cmd=night.fit_command(args,job,{'batch':8,'workers':0},'inspect')
            self.assertEqual(cmd[cmd.index('--anchor-history')+1],'3')
            self.assertEqual(int(cmd[cmd.index('--batch-size')+1])*int(cmd[cmd.index('--accumulate')+1]),512)
        cmd=night.evaluation_command(args,jobs[0],Path('/tmp/best.pt'),Path('/tmp/eval'),1)
        self.assertNotIn('monitor',cmd[cmd.index('--methods')+1:cmd.index('--corrector-checkpoint')])
        self.assertNotIn('--debug-force-step',cmd)
        forced=night.evaluation_command(args,jobs[0],Path('/tmp/best.pt'),Path('/tmp/eval'),1,True)
        self.assertIn('--debug-force-step',forced)

    def test_probe_oom_changes_batch_before_fit_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(Path(tmp));(args.output/'logs').mkdir(parents=True)
            sweep=night.Sweep(args);sweep.compute_started=time.monotonic()
            job=night.specifications(args)[0];calls=[]
            def child(command,label):
                calls.append(command)
                if len(calls)==1:
                    (args.output/'logs'/f'{label}.log').write_text('torch.OutOfMemoryError: CUDA out of memory')
                    raise RuntimeError('probe failed')
                out=Path(command[command.index('--output')+1])
                night.write(out/'probe.json',{'scope':'synthetic fixture'})
            with patch.object(sweep,'wait_gpu',return_value=True),patch.object(sweep,'child',side_effect=child), \
                 patch.object(night.shutil,'disk_usage',return_value=SimpleNamespace(free=2**30)):
                selected=sweep.resources(job)
            self.assertEqual(selected['batch'],4)
            self.assertTrue(all(command[3]=='probe' for command in calls))
            self.assertFalse((args.output/'runs').exists())

    def test_existing_protocol_requires_complete_latest_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(Path(tmp));job=night.specifications(args)[0]
            out=args.output/'runs'/job['name']
            night.write(out/'protocol.json',{})
            with self.assertRaises(RuntimeError):night.fit_command(args,job,{'batch':8,'workers':0},'fit')
            (out/'latest.pt').touch()
            self.assertIn('--resume',night.fit_command(args,job,{'batch':8,'workers':0},'fit'))

    def test_full_stage_order_and_resume_skips_finished_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(Path(tmp));args.models=['transformer-h3'];args.seeds=[42]
            (args.output/'logs').mkdir(parents=True)
            manifest=json.loads((ROOT/'studies/policy_generalization/corrector/manifests/libero_object10_8695891.json').read_text())
            night.write(args.archive/'manifest.json',manifest)
            calls=[]
            def child(runner,cmd,label,**kwargs):
                calls.append(label)
                out=Path(cmd[cmd.index('--output')+1])
                if label.endswith('_training'):
                    night.write(out/'training_status.json',{'status':'completed'})
                    night.write(out/'result.json',{'checkpoint_reload_verified':True,'parameters':37316544,
                                'epochs_completed':1,'optimizer_steps':1,'best_validation':{'loss':1}})
                    (out/'best_model.pt').write_bytes(b'synthetic test weights, not GPU results')
                elif label.endswith('_report'):
                    ckpt=Path(cmd[cmd.index('--checkpoint')+1])
                    night.write(out/'summary.json',{'checkpoint_sha256':night.digest(ckpt),'checkpoint_epoch':1,
                        'sample_count':12089,'split':{'index_sha256':{'train':'fixed','validation':'fixed'}},
                        'metrics':{'cosine':.5,'mse':1,'zero_mse':2}})
                elif label.endswith('_forced_check') or label.endswith('_paired'):
                    night.write(out/'execution.json',{'status':'completed','errors':0})
                    night.write(out/'trials/test.json',{'result':{'guidance_calls':4}})
            with patch.object(night.subprocess,'check_output',return_value=night.SOURCE_SHA+'\n'), \
                 patch.object(night.Sweep,'wait_gpu',return_value=True), \
                 patch.object(night.Sweep,'resources',return_value={'batch':8,'workers':0}), \
                 patch.object(night.Sweep,'child',new=child), \
                 patch.object(night.shutil,'disk_usage',return_value=SimpleNamespace(free=50*2**30)):
                first=night.Sweep(args);first.run()
                self.assertEqual([c.rsplit('_',1)[-1] for c in calls[:3]],['training','report','plots'])
                self.assertTrue(calls[-1].endswith('_paired'))
                self.assertEqual(first.state['status'],'completed')
                calls.clear()
                second=night.Sweep(args);second.run()
                self.assertFalse(any(c.endswith('_training') or c.endswith('_report') for c in calls))
            self.assertTrue((args.output/'comparison.csv').exists())


if __name__=='__main__': unittest.main()
