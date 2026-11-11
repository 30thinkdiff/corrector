"""真实上游阈值状态机+构造策略，验证inference_mode边界及缓存中止；不是VLA任务成绩。"""
from collections import deque
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from .model import VisualDynamicsMLP, CorrectorConfig

ROOT=Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0,str(ROOT/'scripts'))
import corrector_online_runtime as online
SOURCE=ROOT/'outputs/source_audit/vla-corrector'


class FakePolicy:
    def __init__(self):
        self.config=SimpleNamespace(image_features=['one','two'],rtc_config=None)
        self._queues={'action':deque(maxlen=3)}
        self.model=SimpleNamespace(denoise_step=lambda prefix_pad_masks,past_key_values,x_t,timestep:torch.zeros_like(x_t))
        self.calls=0
    def select_action(self,batch):
        assert not batch['dummy'].is_inference()
        if not self._queues['action']:
            self.calls+=1
            x=torch.zeros(1,3,7);t=torch.tensor([.5])
            v=self.model.denoise_step(None,None,x,t)
            self._queues['action'].extend((x-.5*v).transpose(0,1))
        return self._queues['action'].popleft()


@unittest.skipUnless((SOURCE/'.git').exists(),'需要固定上游状态机源码')
class OnlineRuntimeTests(unittest.TestCase):
    def test_10fps_k10_waits_20_control_steps_at_20hz(self):
        import json
        policy = FakePolicy()
        predictor = VisualDynamicsMLP(CorrectorConfig(token_dim=4, action_dim=7,
            action_embed_dim=3, widths=(8,8))).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = online.OnlineCorrector(policy, predictor,
                {'fps':10, 'k_step':10, 'prediction_seconds':1., 'h_window':1},
                'monitor', SOURCE, Path(tmp), fps=20, trace=True)
            self.assertEqual(runtime.k, 20)
            z = torch.randn(1,2,4)
            runtime.encode = lambda batch: z+runtime.step*.01
            with runtime.installed(), torch.inference_mode():
                for _ in range(21):
                    policy.select_action({'dummy':torch.zeros(1)})
            rows = [json.loads(line) for line in (Path(tmp)/'monitor_steps.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['score'] is None for row in rows[:20]))
            self.assertIsNotNone(rows[20]['score'])
            self.assertEqual(runtime.events, 0)

    def test_forced_ogg_exits_outer_inference_mode_and_restores_methods(self):
        torch.manual_seed(7)
        policy=FakePolicy();original=policy.select_action
        predictor=VisualDynamicsMLP(CorrectorConfig(token_dim=4,action_dim=7,action_embed_dim=3,widths=(8,8)))
        predictor.eval().requires_grad_(False)
        before=[p.detach().clone() for p in predictor.parameters()]
        with tempfile.TemporaryDirectory() as tmp:
            runtime=online.OnlineCorrector(policy,predictor,{'prediction_seconds':.1,'h_window':1},'ogg',
                SOURCE,Path(tmp),force_step=2,trace=False,implementation='student')
            z=torch.randn(1,2,4);runtime.encode=lambda batch:z+runtime.step*.1
            with runtime.installed(),torch.inference_mode():
                for _ in range(4):policy.select_action({'dummy':torch.zeros(1)})
            self.assertEqual(runtime.events,1);self.assertEqual(runtime.guidance_calls,1)
            self.assertEqual(policy.calls,2)
            self.assertEqual(policy.select_action,original)
            for p,old in zip(predictor.parameters(),before):
                self.assertIsNone(p.grad);torch.testing.assert_close(p,old,atol=0,rtol=0)

    def test_reference_monitor_does_not_interrupt_and_student_is_not_silently_filled(self):
        interrupt,target=online.decisions('reference')
        self.assertFalse(interrupt('monitor',True));self.assertTrue(interrupt('lvm',True))
        self.assertFalse(interrupt('ogg',False))
        torch.testing.assert_close(target(torch.tensor([2.]),torch.tensor([.5])),torch.tensor([1.5]))
        # 用户已补齐；只用替身检查未完成时仍明确失败，不能退回reference。
        with patch.object(online, 'student_should_interrupt', side_effect=NotImplementedError('尚未实现')):
            with self.assertRaises(NotImplementedError):online.decisions('student')

    def test_student_matches_reference_and_does_not_modify_visual_inputs(self):
        reference_interrupt, reference_target = online.decisions('reference')
        student_interrupt, student_target = online.decisions('student')
        for method in ('monitor', 'lvm', 'ogg'):
            for triggered in (False, True):
                value = student_interrupt(method, triggered)
                self.assertIs(type(value), bool)
                self.assertEqual(value, reference_interrupt(method, triggered))
        expected = torch.tensor([[[.8, -.2], [.3, .1]]])
        observed = torch.tensor([[[.2, -.1], [.4, .05]]])
        before_expected, before_observed = expected.clone(), observed.clone()
        torch.testing.assert_close(student_target(expected, observed), reference_target(expected, observed))
        torch.testing.assert_close(expected, before_expected, atol=0, rtol=0)
        torch.testing.assert_close(observed, before_observed, atol=0, rtol=0)
        with self.assertRaises(ValueError):student_interrupt('oggg', True)
        with self.assertRaises(ValueError):student_target(expected, observed[:, :1])

    def test_student_monitor_ignores_forced_event_and_lvm_replans_without_guidance(self):
        for method, expected_events in [('monitor', 0), ('lvm', 1)]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                policy = FakePolicy()
                predictor = VisualDynamicsMLP(CorrectorConfig(token_dim=4, action_dim=7,
                    action_embed_dim=3, widths=(8,8))).eval().requires_grad_(False)
                runtime = online.OnlineCorrector(policy, predictor, {'prediction_seconds':.1,'h_window':1},
                    method, SOURCE, Path(tmp), force_step=2, trace=False, implementation='student')
                z = torch.randn(1,2,4)
                runtime.encode = lambda batch: z + runtime.step * .1
                with runtime.installed(), torch.inference_mode():
                    for _ in range(3):policy.select_action({'dummy':torch.zeros(1)})
                self.assertEqual(runtime.events, expected_events)
                self.assertEqual(runtime.guidance_calls, 0)
                self.assertEqual(policy.calls, 1 + expected_events)

    def test_noninteger_physical_horizon_is_rejected(self):
        model=VisualDynamicsMLP(CorrectorConfig(token_dim=4,action_dim=7,widths=(8,8))).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):
            online.OnlineCorrector(FakePolicy(),model,{'prediction_seconds':.123,'h_window':1},'lvm',SOURCE,Path(tmp))
