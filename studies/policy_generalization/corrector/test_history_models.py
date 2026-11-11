"""历史预测器真实CPU训练/重载/报告/OGG接入；小构造缓存不是4090成绩。"""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .test_training_entry import runner, SOURCE, fake_cache
from . import test_training_entry as entry
from .test_online_runtime import FakePolicy, online
from .test_prediction_report import report
import corrector_models as models


def history_cache(root):
    fake_cache(root)
    for directory in root.glob('episode_*'):
        np.save(directory/'timestamp.npy', np.arange(7)/10)
        runner.archive.write_json(directory/'COMPLETE.json', {'files':{
            p.name:runner.archive.digest(p) for p in directory.iterdir() if p.name!='COMPLETE.json'}})


@unittest.skipUnless((SOURCE/'.git').exists(),'需要固定上游源码')
class HistoryModelTests(unittest.TestCase):
    def args(self,root,name):
        args=entry.TrainingEntryTests().args(root,name)
        args.local_cache=None;args.model_type='transformer';args.h_window=3;args.anchor_history=3
        args.transformer_dim=8;args.transformer_heads=2;args.transformer_layers=1
        return args

    def test_h1_and_h3_use_identical_current_future_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history_cache(root/'features')
            _,cls,_,_,_=runner.reference_api(SOURCE)
            a,_,_=runner.datasets(root/'features',None,cls,1,1,3)
            b,_,_=runner.datasets(root/'features',None,cls,1,3,3)
            for split in a:
                self.assertEqual(a[split].datasets[0].pairs,b[split].datasets[0].pairs)
                torch.testing.assert_close(a[split][0]['delta_z'],b[split][0]['delta_z'])
                self.assertEqual(b[split][0]['z_hist'].shape,(3,2,4))
            _,info1,_=runner.select_split(a,'episode',.1,42)
            _,info2,_=runner.select_split(b,'episode',.1,43)
            self.assertEqual(info1,info2)

    def test_transformer_training_resume_report_and_online_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history_cache(root/'features')
            full=self.args(root,'full');split=self.args(root,'split')
            full.output.mkdir();split.output.mkdir()
            runner.run(full)
            split.stop_after_epoch=1;runner.run(split)
            split.resume=True;split.stop_after_epoch=0;runner.run(split)
            a=torch.load(full.output/'latest.pt',weights_only=True)
            b=torch.load(split.output/'latest.pt',weights_only=True)
            self.assertEqual(a['step'],b['step'])
            for key in a['state_dict']:
                torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)
            loaded,protocol=online.load_predictor(full.output/'best_model.pt',SOURCE,'cpu')
            self.assertTrue(loaded.uses_history);self.assertEqual(protocol['h_window'],3)
            out=root/'report';out.mkdir()
            report.predict(Namespace(run=full.output,features=full.features,source_root=SOURCE,output=out,
                                     raw_root=None,device='cpu',batch_size=2,loader_workers=0))
            result=json.loads((out/'summary.json').read_text())
            expected=json.loads((full.output/'result.json').read_text())
            self.assertEqual(result['sample_count'],4)
            self.assertAlmostEqual(result['metrics']['mse'],expected['best_validation']['mse'],places=6)

    def test_h3_online_spacing_warmup_and_ogg_inference_boundary(self):
        torch.manual_seed(9)
        predictor=models.transformer_class(SOURCE)(models.TransformerConfig(token_dim=4,action_dim=7,
                      d_model=8,n_heads=2,n_layers=1)).eval().requires_grad_(False)
        policy=FakePolicy()
        with tempfile.TemporaryDirectory() as tmp:
            runtime=online.OnlineCorrector(policy,predictor,
                {'fps':10,'k_step':10,'prediction_seconds':1,'h_window':3,'anchor_history':3},
                'ogg',SOURCE,Path(tmp),fps=20,force_step=24,trace=True)
            base=torch.randn(1,2,4);runtime.encode=lambda batch:base+runtime.step*.01
            with runtime.installed(),torch.inference_mode():
                for _ in range(25): policy.select_action({'dummy':torch.zeros(1)})
            self.assertEqual(runtime.history_stride,2);self.assertEqual(runtime.warmup_steps,4)
            rows=[json.loads(line) for line in (Path(tmp)/'monitor_steps.jsonl').read_text().splitlines()]
            scored=[r['step'] for r in rows if r.get('score') is not None]
            self.assertEqual(scored,[24]);self.assertEqual(runtime.events,1)
            self.assertGreater(runtime.guidance_calls,0)
            torch.testing.assert_close(runtime.context_now[0,:,0,0],base[0,0,0]+torch.tensor([.20,.22,.24]))
            self.assertTrue(all(p.grad is None for p in predictor.parameters()))

    def test_probe_does_not_create_official_training_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history_cache(root/'features')
            args=self.args(root,'probe');args.mode='probe';args.output.mkdir()
            runner.run(args)
            self.assertTrue((args.output/'probe.json').is_file())
            self.assertFalse((args.output/'protocol.json').exists())
            self.assertFalse((args.output/'latest.pt').exists())

    def test_autocast_wrapper_preserves_weights_and_frozen_input_guidance(self):
        cls=models.precision_class(models.transformer_class(SOURCE),'amp-bfloat16')
        model=cls(models.TransformerConfig(token_dim=4,action_dim=7,d_model=8,n_heads=2,n_layers=1)).eval().requires_grad_(False)
        z=torch.randn(1,3,2,4);x=torch.randn(1,5,10);v=torch.randn_like(x)
        guided,loss=models.guided_history_velocity(model,z,x,v,torch.tensor([.5]),torch.randn(1,2,4))
        self.assertTrue(np.isfinite(loss))
        torch.testing.assert_close(guided[:,1:],v[:,1:],rtol=0,atol=0)
        torch.testing.assert_close(guided[:,:,7:],v[:,:,7:],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__': unittest.main()
