"""动作序列训练/重载/OGG；构造CPU证据不冒充4090评测。"""
from argparse import Namespace
from collections import deque
import json
from pathlib import Path
import tempfile
import unittest
import torch

from . import test_history_models as history
from .test_training_entry import runner, SOURCE
from .test_online_runtime import FakePolicy, online
from .test_prediction_report import report
import corrector_models as models


class PlannedPolicy(FakePolicy):
    def __init__(self):
        super().__init__();self._queues={'action':deque(maxlen=12)}
    def select_action(self,batch):
        if not self._queues['action']:
            self.calls+=1
            x=torch.arange(12).float()[None,:,None].expand(1,12,10).clone()/100
            v=self.model.denoise_step(None,None,x,torch.tensor([.5]))
            self._queues['action'].extend((x-.5*v)[:,:,:7].transpose(0,1))
        return self._queues['action'].popleft()


@unittest.skipUnless((SOURCE/'.git').exists(),'需要固定上游源码')
class ActionSequenceTests(unittest.TestCase):
    def args(self,root,name):
        args=history.HistoryModelTests().args(root,name)
        args.k_step=3;args.action_steps=3;args.model_type='transformer-action'
        return args

    def test_training_exact_resume_load_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history.history_cache(root/'features')
            full=self.args(root,'full');partial=self.args(root,'partial')
            full.output.mkdir();partial.output.mkdir();runner.run(full)
            partial.stop_after_epoch=1;runner.run(partial)
            partial.resume=True;partial.stop_after_epoch=0;runner.run(partial)
            a=torch.load(full.output/'latest.pt',weights_only=True)
            b=torch.load(partial.output/'latest.pt',weights_only=True)
            for key in a['state_dict']:torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)
            model,protocol=online.load_predictor(full.output/'best_model.pt',SOURCE,'cpu')
            self.assertTrue(model.uses_action_sequence);self.assertEqual(protocol['action_steps'],3)
            out=root/'report';out.mkdir()
            report.predict(Namespace(run=full.output,features=full.features,source_root=SOURCE,output=out,
                                     raw_root=None,device='cpu',batch_size=2,loader_workers=0))
            self.assertEqual(json.loads((out/'summary.json').read_text())['sample_count'],2)

    def test_dataset_exact_actions_no_cross_episode_or_future_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history.history_cache(root/'features')
            _,cls,_,_,_=runner.reference_api(SOURCE)
            single,_,_=runner.datasets(root/'features',None,cls,3,1,3)
            seq,_,_=runner.datasets(root/'features',None,cls,3,1,3,3)
            ds=seq['train'].datasets[0];t,future=ds.pairs[0]
            torch.testing.assert_close(ds[0]['a_seq'],torch.tensor(ds.base.a_t_np[t:t+3].copy()).float())
            torch.testing.assert_close(ds[0]['z_t'],single['train'][0]['z_t'])
            self.assertEqual(t+3,future)
            with self.assertRaises(ValueError):models.ActionSequenceDataset(ds.base,4)

    def test_sequence_ogg_only_selected_actions_change(self):
        model=models.action_transformer_class(SOURCE)(models.ActionSequenceConfig(
            token_dim=4,action_dim=7,d_model=8,n_heads=2,n_layers=1,action_steps=3)).eval().requires_grad_(False)
        z=torch.randn(1,3,2,4);x=torch.randn(1,8,10);v=torch.randn_like(x)
        guided,loss=models.guided_sequence_velocity(model,z,x,v,torch.tensor([.5]),torch.randn(1,2,4),2)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        torch.testing.assert_close(guided[:,[1,3,5,6,7]],v[:,[1,3,5,6,7]],rtol=0,atol=0)
        torch.testing.assert_close(guided[:,:,7:],v[:,:,7:],rtol=0,atol=0)
        self.assertGreater((guided[:,[0,2,4]]-v[:,[0,2,4]]).abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaises(ValueError):models.guided_sequence_velocity(model,z,x[:,:3],v[:,:3],torch.tensor([.5]),torch.randn(1,2,4),2)

    def test_online_plan_spacing_cancellation_skip_and_forced_guidance(self):
        predictor=models.action_transformer_class(SOURCE)(models.ActionSequenceConfig(
            token_dim=4,action_dim=7,d_model=8,n_heads=2,n_layers=1,action_steps=3)).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp:
            policy=PlannedPolicy()
            runtime=online.OnlineCorrector(policy,predictor,{'fps':10,'k_step':3,'prediction_seconds':.3,
                'h_window':3,'anchor_history':3,'action_steps':3},'ogg',SOURCE,Path(tmp),fps=20,force_step=10)
            runtime.encode=lambda batch:torch.ones(1,2,4)*runtime.step/100
            with runtime.installed(),torch.inference_mode():
                for _ in range(25):policy.select_action({'dummy':torch.zeros(1)})
            rows=[json.loads(line) for line in (Path(tmp)/'monitor_steps.jsonl').read_text().splitlines()]
            self.assertGreater(runtime.guidance_calls,0)
            self.assertTrue(any(r.get('condition_action_offsets')==[0,2,4] for r in rows))
            self.assertTrue(any(r.get('prediction_skipped')=='insufficient_planned_actions' for r in rows))
            self.assertTrue(any('cancelled_predictions' in r for r in rows))
            self.assertTrue(any(r.get('cancelled_on_new_plan',0)>0 for r in rows))
            self.assertEqual(next(r['step'] for r in rows if r.get('score') is not None),10)

    def test_plan_uses_only_current_and_available_queue(self):
        current=torch.zeros(1,7);queue=deque([torch.ones(1,7)*i for i in range(1,5)])
        result=models.planned_actions(current,queue,3,2)
        torch.testing.assert_close(result[0,:,0],torch.tensor([0.,2.,4.]))
        self.assertIsNone(models.planned_actions(current,queue,4,2))
        self.assertEqual(len(queue),4)


if __name__=='__main__':unittest.main()
