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
    def test_actual_episode_success_schema_not_aggregate_percentage(self):
        self.assertTrue(obj.one_success({'per_episode': [{'success': True}]}))
        self.assertFalse(obj.one_success({'per_task': [{'metrics': {'successes': [False]}}]}))
        for info in [{'aggregated': {'pc_success': 100}}, {'per_episode': []},
                     {'per_episode': [{'success': 1}]}, {'per_episode': [{'success': True}, {'success': False}]}]:
            with self.assertRaises(ValueError): obj.one_success(info)

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
                self.assertEqual(len(r.pending), 20)
                self.assertEqual(r.cancelled_predictions, 0)
                logs[method] = next(x for x in rows if x.get('interrupted'))
                self.assertEqual(r.guidance_calls > 0, method == 'c0_ogg')
            self.assertIsNone(pred.w.grad)
        for key in ['score', 'threshold', 'layer_scores']:
            self.assertEqual(logs['c0_lvm'][key], logs['c0_ogg'][key])

    def test_joint_interrupt_preserves_due_predictions_and_continuous_scores(self):
        import json
        import j36m.online as online
        observations = {}
        for method in ['j_monitor', 'j_lvm', 'j_lvm_ogg']:
            p = fixtures.Policy(); model = fixtures.FakeJoint().eval().requires_grad_(False)
            r = obj.joint_class()(p, model, self.cal, method, self.root, self.root / method,
                                  checkpoint_sha='test', force_step=20)
            z = torch.arange(6).reshape(1, 2, 3).float()
            r.encode = lambda batch: online.pack({k: z if r.step < 20 else -z for k in online.LAYERS})
            with r.installed():
                for _ in range(25): p.select_action({})
            rows = [json.loads(s) for s in (self.root / method / 'monitor_steps.jsonl').read_text().splitlines()]
            scored = [row['step'] for row in rows if 'score' in row and row['score'] is not None]
            self.assertEqual(scored, list(range(20, 25)))
            self.assertEqual(len(r.pending), 20)
            self.assertEqual(r.cancelled_predictions, 0)
            self.assertIsNone(model.scale.grad)
            if method == 'j_monitor':
                self.assertEqual(r.events, 0); self.assertEqual(p.calls, 1)
            else:
                self.assertGreater(r.events, 0); self.assertGreater(p.calls, 1)
                observations[method] = next(row for row in rows if row.get('interrupted'))
                self.assertEqual(r.guidance_calls > 0, method == 'j_lvm_ogg')
        for key in ['score', 'threshold', 'layer_scores']:
            self.assertEqual(observations['j_lvm'][key], observations['j_lvm_ogg'][key])

    def test_calibration_reuse_is_byte_identical_and_rejects_changed_payload(self):
        import json
        source = self.root / 'old'; output = self.root / 'new'; output.mkdir()
        (source / 'calibration').mkdir(parents=True); (source / 'pilot').mkdir()
        obj.write(source / 'calibration/execution.json', {'status': 'completed', 'completed': 40, 'errors': 0})
        cases = []
        for name in ['j', 'c0']:
            cal = {'checkpoint_sha256': name, 'status': 'frozen', 'domain': 'ordinary_libero_object',
                   'training_manifest_sha256': 'data', 'cases': [{'seed': 9000}]}
            path = source / f'{name}_calibration.json';obj.write(path, cal)
            cases.append({'method': name + '_lvm', 'config': {'calibration_sha256': obj.digest(path)}})
        obj.write(source / 'pilot/plan.json', {'protocol': {'checkpoints': {'j': 'j', 'c0': 'c0'},
            'core_manifest_sha256': obj.BASE_MANIFEST}, 'cases': cases})
        obj.reuse_calibration(source, output, {'j': 'j', 'c0': 'c0'}, 'data')
        for name in ['j', 'c0']:
            self.assertEqual((source / f'{name}_calibration.json').read_bytes(), (output / f'{name}_calibration.json').read_bytes())
        (source / 'j_calibration.json').write_text((source / 'j_calibration.json').read_text() + '\n')
        with self.assertRaises(ValueError): obj.reuse_calibration(source, output, {'j': 'j', 'c0': 'c0'}, 'data')


if __name__ == '__main__': unittest.main()
