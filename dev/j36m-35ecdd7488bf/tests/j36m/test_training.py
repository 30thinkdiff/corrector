"""Tiny optimizer/control fixtures; real fixed J tensor gradients live in test_core."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'tests/multilayer_mvp'))
from test_mvp import fixture
from multilayer_mvp.data import validate
from j36m.audit import feature_scale_audit
import j36m.training as tr

class TinyReference(nn.Module):
    def __init__(self):super().__init__();self.p=nn.Linear(7,12)
    def forward(self,x):return {'delta_connector':self.p(x['action'])[:,None,:].expand(-1,4,-1)}

class TrainingTransactions(unittest.TestCase):
    def test_numpy_rng_checkpoint_is_weights_only_safe_and_exact(self):
        np.random.seed(100);np.random.normal(size=3)
        state=tr.numpy_rng_state();expected=np.random.normal(size=11)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rng.pt';torch.save(state,path)
            tr.restore_numpy_rng(torch.load(path,weights_only=True))
        np.testing.assert_array_equal(np.random.normal(size=11),expected)

    def test_resumption_and_partial_accumulation_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=fixture(root);records=validate(cfg)
            audit=feature_scale_audit(records,root/'audit')
            def args(path,resume=False):return SimpleNamespace(output=path,variant='C0',epochs=2,microbatch=3,accumulate=2,patience=0,
                  seed=42,lr=.001,precision='fp32',gradient_checkpointing=False,device='cpu',resume=resume)
            with patch.object(tr,'make_model',side_effect=lambda *a:TinyReference()):
                first=tr.train(args(root/'reference'),cfg,records,audit,lambda:None)
                # 8 training samples, microbatch3 x accumulation2 => two optimizer steps/epoch.
                self.assertEqual(first['optimizer_steps'],4)
                calls=[]
                def guard():
                    calls.append(True)
                    if len(calls)==2:raise RuntimeError('unit STOP at epoch boundary')
                with self.assertRaises(RuntimeError):tr.train(args(root/'paused'),cfg,records,audit,guard)
                saved=torch.load(root/'paused/latest.pt',weights_only=True)
                self.assertEqual(saved['epoch'],1);self.assertEqual(saved['step'],2)
                second=tr.train(args(root/'paused',True),cfg,records,audit,lambda:None)
                self.assertEqual(first['best_validation'],second['best_validation'])
                self.assertTrue(second['checkpoint_reload_verified'])
                a=args(root/'paused',True);a.accumulate=3
                with self.assertRaisesRegex(ValueError,'protocol mismatch'):tr.train(a,cfg,records,audit,lambda:None)

if __name__=='__main__':unittest.main()
