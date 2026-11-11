"""时间戳/保存的到期步检查；构造轨迹，不代表已读取远端300回合。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('temporal_audit', ROOT/'scripts/corrector_temporal_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TemporalAuditTests(unittest.TestCase):
    def fixture(self, root):
        features, evaluation = root/'features', root/'eval'
        directory = features/'episode_1'
        write(directory/'COMPLETE.json', {})
        write(directory/'metadata.json', {'fps': 10})
        np.save(directory/'timestamp.npy', np.arange(30)/10)
        manifest = root/'manifest.json'
        write(manifest, {'fps': 10, 'records': [{'episode': 1, 'frames': 30}]})
        case = {'id': 'case1', 'method': 'ogg', 'config': {'uses_corrector': True}}
        write(evaluation/'plan.json', {'cases': [case], 'protocol': {'fps': 20, 'corrector_settings': {'stride': 1}}})
        write(evaluation/'execution.json', {'status': 'completed'})
        write(evaluation/'trials/case1.json', {'status': 'completed', 'attempt': 1, 'result': {'env_steps': 25}})
        attempt = evaluation/'attempts/case1/attempt_001'
        write(attempt/'actual_protocol.json', {'monitor_horizon_steps': 20,
              'predictor_protocol': {'fps': 10, 'k_step': 10, 'prediction_seconds': 1}})
        (attempt/'monitor_steps.jsonl').write_text('\n'.join(json.dumps({'step': i, 'score': 0.1 if i >= 20 else None}) for i in range(25)))
        return features, manifest, evaluation, attempt

    def test_different_rates_can_have_equal_physical_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            features, manifest, evaluation, attempt = self.fixture(Path(tmp))
            value = audit.audit(features, manifest, evaluation)
            self.assertEqual(value['error_count'], 0)
            self.assertEqual(value['horizons'], [[10., 10, 1., 20., 20]])
            self.assertEqual(value['first_score_steps'], [20])
            self.assertFalse(value['source_physical_clock_verified'])

    def test_detects_unconverted_online_k_and_early_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            features, manifest, evaluation, attempt = self.fixture(Path(tmp))
            write(attempt/'actual_protocol.json', {'monitor_horizon_steps': 10,
                  'predictor_protocol': {'fps': 10, 'k_step': 10, 'prediction_seconds': 1}})
            (attempt/'monitor_steps.jsonl').write_text(json.dumps({'step': 10, 'score': .1}))
            self.assertGreaterEqual(audit.audit(features, manifest, evaluation)['error_count'], 2)

    def test_detects_timestamp_rate_disagreeing_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            features, manifest, evaluation, attempt = self.fixture(Path(tmp))
            np.save(features/'episode_1/timestamp.npy', np.arange(30)/20)
            errors = audit.audit(features, manifest, evaluation)['errors_first_10']
            self.assertTrue(any('timestamps disagree' in error for error in errors))


if __name__ == '__main__':
    unittest.main()
