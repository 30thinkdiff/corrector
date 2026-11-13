"""CPU-only ordinary-LIBERO protocol/leakage/control tests; no fake GPU claims."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import corrector_j36m_object_eval as obj
import test_online as fixtures
from unittest.mock import patch
import torch


def trial(seed=9000, init=7, task=0):
    rows = [{'layer_scores': {k: {'valid': True, 'score': .1 + n * .001,
             'predicted_rms': 1., 'observed_rms': 1.1} for k in ('block_02', 'block_08', 'connector')}} for n in range(40)]
    return {'condition': f'libero_object/task_{task}/init_{init}', 'init': init, 'seed': seed,
            'normal_success': True, 'checkpoint_sha256': 'sha', 'rows': rows}


class ProtocolTests(unittest.TestCase):
    def test_fixed_pilot_and_formal_counts_and_calibration_disjoint(self):
        self.assertEqual(len(obj.phase_cases('pilot')), 24)
        self.assertEqual(len(obj.phase_cases('formal')), 360)
        self.assertEqual(len(obj.phase_cases('calibration')), 40)
        cal = {(t, i, s) for t, i, s, _ in obj.phase_cases('calibration')}
        test = {(t, i, s) for t, i, s, _ in obj.phase_cases('formal')}
        self.assertFalse(cal & test)
        for phase in ['pilot', 'formal', 'calibration', 'passive', 'forced']:
            cases = obj.phase_cases(phase)
            self.assertEqual(len(cases), len(set(cases)))

    def test_calibration_explicit_domain_and_no_test_refit(self):
        a, b = trial(), trial(9001, 8)
        fitted = obj.fit_normal([a, b], 'sha', 'manifest', ('block_02', 'block_08', 'connector'))
        self.assertEqual(fitted['domain'], 'ordinary_libero_object')
        self.assertEqual(len(fitted['normal_history']), 80)
        self.assertAlmostEqual(sum(fitted['weights'].values()), 1.)
        for invalid in [dict(a, seed=1000), dict(a, normal_success=False), dict(a, init=0), dict(a, checkpoint_sha256='other')]:
            with self.assertRaises(ValueError): obj.fit_normal([invalid, b], 'sha', 'manifest', ('connector',))
        with self.assertRaises(ValueError): obj.fit_normal([a, a], 'sha', 'manifest', ('connector',))

    def test_single_layer_estimator_does_not_import_three_fake_layers(self):
        a, b = trial(), trial(9001, 8)
        for t in [a, b]:
            for r in t['rows']: r['layer_scores'] = {'connector': r['layer_scores']['connector']}
        fit = obj.fit_normal([a, b], 'sha', 'manifest', ('connector',))
        self.assertEqual(fit['weights'], {'connector': 1.})


class ConnectorTests(unittest.TestCase):
    """Reuse the existing real-state-machine fixture, not a new controller mock."""
    setUp = fixtures.ControlTests.setUp

    def test_connector_monitor_and_same_forced_lvm_ogg_trigger(self):
        class Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__();self.cfg = type('Cfg', (), {'action_dim': 7, 'token_dim': 3})()
                self.w = torch.nn.Parameter(torch.ones(1))
            def forward(self, z, action):
                return z * .05 + self.w * action[:, :1, None]
        logs = {}
        import json
        for method in ['c0_monitor', 'c0_lvm', 'c0_ogg']:
            p = fixtures.Policy(); pred = Predictor().eval().requires_grad_(False)
            cal = dict(self.cal, norm_floor={'connector': 1e-8})
            r = obj.connector_class()(p, pred, cal, method, self.root, self.root / method, 20)
            z = torch.arange(6).reshape(1, 2, 3).float()
            r.encode = lambda batch: z if r.step < 20 else -z
            with r.installed():
                for _ in range(21): p.select_action({})
            rows = [json.loads(s) for s in (self.root / method / 'monitor_steps.jsonl').read_text().splitlines()]
            if method == 'c0_monitor':
                self.assertEqual(r.events, 0);self.assertEqual(p.calls, 1)
            else:
                self.assertEqual(r.events, 1);self.assertEqual(p.calls, 2)
                self.assertEqual(len(r.pending), 1)
                logs[method] = next(x for x in rows if x.get('interrupted'))
                self.assertEqual(r.guidance_calls > 0, method == 'c0_ogg')
            self.assertIsNone(pred.w.grad)
        for key in ['score', 'threshold', 'layer_scores']:
            self.assertEqual(logs['c0_lvm'][key], logs['c0_ogg'][key])


if __name__ == '__main__': unittest.main()
