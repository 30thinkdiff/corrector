import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import corrector_expanded_eval as expanded
import test_online as fixtures
import j36m.online as online
import torch

class Plans(unittest.TestCase):
    def test_registry_and_only_new_joint_ordinary_cases(self):
        self.assertEqual(len(expanded.registry()),17)
        rows=expanded.cases('ordinary',{})
        self.assertEqual(len(rows),120);self.assertEqual(len(set(rows)),120)
        self.assertEqual({m for _,_,m in rows},{'j_lvm','j_ogg'})
        self.assertEqual(len({(c,s) for c,s,_ in rows}),60)
        self.assertEqual(len(expanded.cases('bridge',{})),4)
    def test_plus_counts_and_historical_default_smoke(self):
        conditions={'plus/default':{},'plus/view5':{}}
        self.assertEqual(len(expanded.cases('plus',conditions)),2*3*17)
        self.assertEqual(len(expanded.cases('plus_smoke',conditions)),17)

class DynamicControl(unittest.TestCase):
    setUp=fixtures.ControlTests.setUp
    def test_original_dynamic_bootstrap_warmup_and_retained_prediction(self):
        p=fixtures.Policy();encoder=object();model=fixtures.FakeJoint().eval().requires_grad_(False)
        runtime=expanded.dynamic_joint_class()(p,model,self.cal,'j_lvm',self.root,self.root/'dynamic',encoder)
        self.assertEqual(runtime.state.window_size,15)
        self.assertFalse(runtime.state.bootstrapped)
        self.assertEqual(runtime.state.cooldown_steps,0)
        self.assertEqual(runtime.warmup_steps,4)
        z=torch.arange(6).reshape(1,2,3).float()
        runtime.encode=lambda batch:online.pack({k:z*(1+runtime.step*.01) for k in online.LAYERS})
        with runtime.installed():
            for _ in range(45):p.select_action({})
        import json
        rows=[json.loads(s) for s in (self.root/'dynamic/monitor_steps.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in rows if 'score'in r and r['score']is not None],list(range(24,45)))
        self.assertTrue(runtime.state.bootstrapped)
        self.assertEqual(runtime.cancelled_predictions,0)
        self.assertEqual(len(runtime.pending),20)
    def test_joint_encoding_uses_separate_feature_encoder(self):
        from unittest.mock import patch
        p=fixtures.Policy();encoder=object();model=fixtures.FakeJoint().eval().requires_grad_(False)
        runtime=expanded.dynamic_joint_class()(p,model,self.cal,'j_lvm',self.root,self.root/'encoder',encoder)
        seen=[]
        def encode(this,batch):seen.append(this.policy);return torch.zeros(1)
        with patch.object(online.JointOnlineCorrector,'encode',encode):runtime.encode({})
        self.assertIs(seen[0],encoder);self.assertIs(runtime.policy,p)

if __name__=='__main__':unittest.main()
