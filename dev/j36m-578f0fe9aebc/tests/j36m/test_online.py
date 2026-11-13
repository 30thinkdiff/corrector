"""Tiny control fixtures exercise real legacy state/queue, full J tested separately."""
from collections import deque
import ast,json,sys,tempfile,types,unittest
from pathlib import Path
from dataclasses import dataclass,field
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import j36m.online as online
from j36m.calibration import fit_calibration
from j36m.model import LAYERS


def threshold_api():
    # Exact two AST definitions from the frozen upstream source; no replacement state machine.
    path=ROOT/'tests/j36m/threshold_reference.py'
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('RobustThresholdState','robust_threshold_step')]
    mod=types.ModuleType('_j36_test_threshold');sys.modules[mod.__name__]=mod
    mod.__dict__.update(dataclass=dataclass,field=field,np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),mod.__dict__)
    return mod.RobustThresholdState,mod.robust_threshold_step,'fixed-source-cpu-fixture'


class FakeJoint(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.ones(1))
    def forward(self,x):return {'delta_'+k:x[k]*.05+self.scale*x['action'][:,0,None,None] for k in LAYERS}


class Policy:
    def __init__(self):
        self.config=types.SimpleNamespace(rtc_config=None,image_features=['observation.images.camera1','observation.images.camera2'])
        self._queues={'action':deque(maxlen=50)};self.calls=0
        self.model=types.SimpleNamespace(denoise_step=lambda *a:torch.zeros(1,50,7))
    def select_action(self,batch):
        if not self._queues['action']:
            self.calls+=1
            self.model.denoise_step(None,None,torch.ones(1,50,7),torch.tensor([.5]))
            self._queues['action'].extend(torch.ones(1,7) for _ in range(50))
        return self._queues['action'].popleft()


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        p=self.root/'src/lerobot/scripts';p.mkdir(parents=True);(p/'lerobot_eval_modified_detection.py').touch()
        self.stack=[]
        for item in [patch.object(online,'SHAPES',{k:(2,3) for k in LAYERS}),
                     patch.object(online,'SIZES',{k:6 for k in LAYERS}),
                     patch.object(online.legacy,'monitor_api',return_value=threshold_api())]:
            item.start();self.addCleanup(item.stop)
        self.cal={'status':'frozen','checkpoint_sha256':'test','weights':{k:1/3 for k in LAYERS},
                  'norm_floor':{k:1e-8 for k in LAYERS},'normal_history':[.01,.02]*10,
                  'state_settings':{'base_noise_floor':.1,'trigger_consecutive_steps':1,'jump_trigger':.01,'cooldown_steps':20}}
    def runtime(self,method):
        p=Policy();j=FakeJoint().eval().requires_grad_(False)
        r=online.JointOnlineCorrector(p,j,self.cal,method,self.root,self.root/method,checkpoint_sha='test')
        def encode(batch):
            v=torch.arange(6).reshape(1,2,3).float()
            return online.pack({k:v if r.step<20 else -v for k in LAYERS})
        r.encode=encode
        return p,j,r
    def test_same_trigger_lvm_no_guidance_ogg_nonzero_and_pending_cancel(self):
        logs={}
        for method in ('j_lvm','j_lvm_ogg'):
            p,j,r=self.runtime(method)
            before=j.scale.detach().clone()
            with r.installed():
                for _ in range(21):p.select_action({})
            self.assertTrue(torch.equal(before,j.scale))
            self.assertEqual(r.events,1);self.assertEqual(p.calls,2)
            self.assertEqual(len(r.pending),1)  # only post-replan prediction survives
            self.assertGreater(r.cancelled_predictions,0)
            if method=='j_lvm':self.assertEqual(r.guidance_calls,0)
            else:self.assertGreater(r.guidance_calls,0)
            rows=[json.loads(s) for s in (self.root/method/'monitor_steps.jsonl').read_text().splitlines()]
            logs[method]=next(row for row in rows if row.get('interrupted'))
            self.assertTrue((self.root/method/logs[method]['correction_target_file']).is_file())
        for key in ['score','threshold','trigger','layer_scores']:
            self.assertEqual(logs['j_lvm'][key],logs['j_lvm_ogg'][key])
    def test_monitor_never_clears_queue_and_no_guidance(self):
        p,j,r=self.runtime('j_monitor')
        with r.installed():
            for _ in range(21):p.select_action({})
        self.assertEqual(r.events,0);self.assertEqual(p.calls,1);self.assertEqual(len(p._queues['action']),29)
        self.assertEqual(r.guidance_calls,0)
    def test_natural_chunk_refresh_preserves_single_action_score_coverage(self):
        p,j,r=self.runtime('j_monitor')
        v=torch.arange(6).reshape(1,2,3).float()
        r.encode=lambda b:online.pack({k:v*(1+r.step*.01) for k in LAYERS})
        with r.installed():
            for _ in range(51):p.select_action({})
        rows=[json.loads(s) for s in (self.root/'j_monitor/monitor_steps.jsonl').read_text().splitlines()]
        scored=[x for x in rows if x.get('score') is not None]
        self.assertEqual(len(scored),31)
        self.assertEqual(len(r.pending),20)
        self.assertEqual(r.cancelled_predictions,0)

    def test_due_steps_and_reset_per_instance(self):
        p,j,r=self.runtime('j_monitor');self.assertEqual(r.k,20)
        with r.installed():
            for _ in range(20):p.select_action({})
        self.assertEqual(len(r.state.dynamic_threshold),0)
        p,j,other=self.runtime('j_lvm');self.assertEqual(other.step,0);self.assertFalse(other.pending)
    def test_zero_nonfinite_layers_skipped_no_fake_alarm(self):
        zero={k:torch.zeros(1,2,3) for k in LAYERS}
        score,details=online.layer_scores(zero,zero,self.cal);self.assertIsNone(score)
        p,j,r=self.runtime('j_lvm');r.encode=lambda b:online.pack(zero)
        # Fake predictor still outputs nonzero, observed is exactly zero => skipped.
        with r.installed():
            for _ in range(25):p.select_action({})
        self.assertEqual(r.events,0);self.assertEqual(r.state.dynamic_threshold,[])
    def test_incompatible_calibration_rejected(self):
        p=Policy();j=FakeJoint().eval().requires_grad_(False)
        with self.assertRaises(ValueError):online.JointOnlineCorrector(p,j,self.cal,'j_lvm',self.root,self.root/'bad',checkpoint_sha='different')
    def test_calibration_holdout_seeds_and_views_rejected(self):
        trial={'condition':'view5','seed':1000,'normal_success':True}
        with self.assertRaises(ValueError):fit_calibration([trial,dict(trial,seed=1001)],'sha','manifest')

if __name__=='__main__':unittest.main()
