import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("baseline_benchmark", Path(__file__).resolve().parents[3] / "scripts/smolvla_baseline_benchmark.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineEntryTests(unittest.TestCase):
    def test_explicit_physical_initial_state_not_just_new_seed(self):
        subenv = SimpleNamespace(_init_states=["state0", "state1"], init_state_id=0)
        env = SimpleNamespace(envs=[SimpleNamespace(unwrapped=subenv)])
        baseline.select_initial_state(env, 1)
        self.assertEqual(subenv.init_state_id, 1)

    def test_out_of_bounds_rejected_instead_of_modulo_wrap(self):
        subenv = SimpleNamespace(_init_states=["state0", "state1"], init_state_id=0)
        env = SimpleNamespace(envs=[SimpleNamespace(unwrapped=subenv)])
        for index in [-1, 2, True]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                baseline.select_initial_state(env, index)
        self.assertEqual(subenv.init_state_id, 0)
