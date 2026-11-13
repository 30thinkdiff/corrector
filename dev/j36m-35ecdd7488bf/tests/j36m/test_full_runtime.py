"""Full 36,793,536-parameter J + reused legacy runtime, synthetic observations.
No VLA/environment/GPU performance claim; the J tensors and gradients are real.
"""
from collections import deque
import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from j36m.model import Joint36M,LAYERS
from j36m.online import JointOnlineCorrector,pack
import corrector_online_runtime as legacy
from test_online import threshold_api
from test_core import full_inputs

class FlowPolicy:
    def __init__(self):
        self.config=types.SimpleNamespace(rtc_config=None,image_features=['observation.images.camera1','observation.images.camera2']);self._queues={'action':deque(maxlen=50)}
        self.model=types.SimpleNamespace(denoise_step=lambda *a:torch.zeros(1,50,32))
    def select_action(self,batch):
        if not self._queues['action']:
            x=torch.ones(1,50,32)
            for i in range(10):x=x-.1*self.model.denoise_step(None,None,x,torch.tensor([1.-i*.1]))
            self._queues['action'].extend(x[:,i,:7] for i in range(50))
        return self._queues['action'].popleft()

class FullRuntime(unittest.TestCase):
    def test_full_J_lvm_and_ogg_share_detection_until_action_diverges(self):
        torch.set_num_threads(2);torch.manual_seed(9)
        model=Joint36M().eval().requires_grad_(False);visual=full_inputs();visual.pop('action')
        origin=pack(visual);future=pack({k:-v for k,v in visual.items()})
        cal={'status':'frozen','checkpoint_sha256':'same','weights':dict(zip(LAYERS,(.25,.25,.5))),
             'norm_floor':{k:1e-8 for k in LAYERS},'normal_history':[.01,.02]*10,
             'state_settings':{'base_noise_floor':.1,'trigger_consecutive_steps':1,'jump_trigger':.01,'cooldown_steps':20}}
        version={k:p._version for k,p in model.named_parameters()};observations={}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'src/lerobot/scripts';source.mkdir(parents=True);(source/'lerobot_eval_modified_detection.py').touch()
            with patch.object(legacy,'monitor_api',return_value=threshold_api()):
                for method in ('j_lvm','j_lvm_ogg'):
                    policy=FlowPolicy();runtime=JointOnlineCorrector(policy,model,cal,method,root,root/method,checkpoint_sha='same')
                    runtime.encode=lambda b:origin if runtime.step<20 else future
                    actions=[]
                    with runtime.installed():
                        for i in range(21):actions.append(policy.select_action({}).detach().clone())
                    logs=[json.loads(line) for line in (root/method/'monitor_steps.jsonl').read_text().splitlines()]
                    event=next(row for row in logs if row.get('interrupted'))
                    observations[method]={'event':event,'actions':actions,'calls':runtime.guidance_calls}
        a,b=observations['j_lvm'],observations['j_lvm_ogg']
        self.assertEqual(a['calls'],0);self.assertEqual(b['calls'],4)
        for key in ('score','threshold','trigger','layer_scores'):self.assertEqual(a['event'][key],b['event'][key])
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a['actions'][:20],b['actions'][:20])))
        self.assertFalse(torch.equal(a['actions'][20],b['actions'][20]))
        self.assertEqual(version,{k:p._version for k,p in model.named_parameters()})
        self.assertTrue(all(p.grad is None for p in model.parameters()))

if __name__=='__main__':unittest.main()
