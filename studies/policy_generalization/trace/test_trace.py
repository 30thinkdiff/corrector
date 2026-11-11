"""CPU 上验证记录器透明性；这些替身测试不等于运行真实 SmolVLA。"""
import io
import json
import sys
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__:
    from .eval_with_trace import run_with_trace
    from .policy_trace import PolicyTrace
    from .tensor_summary import summarize_tensor
else:
    from eval_with_trace import run_with_trace
    from policy_trace import PolicyTrace
    from tensor_summary import summarize_tensor


class SummaryTests(unittest.TestCase):
    def test_json_safe_bfloat16_nonfinite_and_empty(self):
        for tensor in [torch.tensor([1, 2], dtype=torch.bfloat16),
                       torch.tensor([float("nan"), float("inf"), -float("inf")]),
                       torch.tensor([], dtype=torch.float32),
                       torch.tensor([True, False]), torch.tensor([3, 1])]:
            with self.subTest(dtype=tensor.dtype, size=tensor.numel()):
                result = summarize_tensor("x", tensor, verbose=False)
                json.dumps(result, allow_nan=False)
                self.assertEqual(result["shape"], list(tensor.shape))
        result = summarize_tensor("nan", torch.tensor([float("nan")]), verbose=False)
        self.assertTrue(result["has_nan"])
        self.assertEqual(result["first_few"], ["nan"])

    def test_preserves_input_and_gradient_graph(self):
        value = torch.tensor([2.0, 3.0], requires_grad=True)
        summarize_tensor("x", value, verbose=False)
        value.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))
        torch.testing.assert_close(value.detach(), torch.tensor([2.0, 3.0]))


class FakeVLM:
    def embed_image(self, image):
        return image.mean((2, 3))[:, None, :]


class FakeFlow:
    def __init__(self):
        self.vlm_with_expert = FakeVLM()

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def embed_prefix(self, images, masks, tokens, lang_masks, state):
        for image in images:
            self.vlm_with_expert.embed_image(image)
        return state[:, None, :], lang_masks, lang_masks

    def embed_suffix(self, x, time):
        return x, torch.ones_like(x[:, :, 0], dtype=torch.bool), time

    def sample_actions(self, images, masks, tokens, lang_masks, state, noise=None):
        if noise is None:
            noise = self.sample_noise((state.shape[0], 3, 4), state.device)
        self.embed_prefix(images, masks, tokens, lang_masks, state)
        self.embed_suffix(noise, torch.ones(state.shape[0]))
        self.embed_suffix(noise, torch.zeros(state.shape[0]))
        return noise + state[:, None, :1]


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeFlow()
        self.config = SimpleNamespace(chunk_size=3, n_action_steps=3, num_steps=2,
                                      max_action_dim=4, max_state_dim=4)
        self._queues = {"action": deque(maxlen=3)}
        self.last_chunk = None

    def prepare_images(self, batch):
        return [batch["observation.images.camera1"] * 2-1], [torch.tensor([True])]

    def prepare_state(self, batch):
        return torch.nn.functional.pad(batch["observation.state"], (0, 2))

    def _get_action_chunk(self, batch, noise=None):
        images, masks = self.prepare_images(batch)
        actions = self.model.sample_actions(images, masks, batch["observation.language.tokens"],
                                           batch["observation.language.attention_mask"],
                                           self.prepare_state(batch), noise=noise)
        self.last_chunk = actions[:, :, :2]
        return self.last_chunk

    def select_action(self, batch, noise=None):
        if not self._queues["action"]:
            actions = self._get_action_chunk(batch, noise=noise)
            self._queues["action"].extend(actions.transpose(0, 1))
        self.last_action = self._queues["action"].popleft()
        return self.last_action


def batch():
    return {
        "observation.images.camera1": torch.full((1, 3, 4, 4), .5),
        "observation.state": torch.tensor([[.2, .3]]),
        "observation.language.tokens": torch.tensor([[1, 2]]),
        "observation.language.attention_mask": torch.tensor([[True, True]]),
    }


class TraceTests(unittest.TestCase):
    def test_real_calls_not_replayed_rng_queue_and_return_preserved(self):
        observation = batch()
        torch.manual_seed(42)
        plain = FakePolicy()
        expected = [plain.select_action(observation) for _ in range(4)]
        expected_rng = torch.random.get_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(42)
            policy = FakePolicy()
            trace = PolicyTrace(directory)
            trace.install(policy)
            actual = [policy.select_action(observation) for _ in range(4)]
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right)
            self.assertIs(actual[-1], policy.last_action)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
            self.assertEqual([x["generated_chunks"] for x in trace.report["queue_steps"]], [1, 0, 0])
            self.assertEqual([x["after"] for x in trace.report["queue_steps"]], [2, 1, 0])
            self.assertEqual(trace.report["tensors"]["embed_suffix.first_output.2"]["first_few"], [1.0])
            recorded = torch.load(Path(directory)/"action_chunk.pt", weights_only=True)
            torch.testing.assert_close(recorded, torch.stack(expected[:3], dim=1))
            torch.testing.assert_close(observation["observation.images.camera1"], torch.full((1,3,4,4), .5))
            self.assertTrue((Path(directory)/"observation_images_camera1_batch0.png").is_file())
            trace.close()
            self.assertNotIn("select_action", vars(policy))
            self.assertNotIn("sample_actions", vars(policy.model))
            self.assertTrue(json.loads((Path(directory)/"trace.json").read_text())["complete"])

    def test_processors_and_factories_restored_even_when_evaluation_fails(self):
        class Pipeline:
            def __init__(self):
                self.before_step_hooks, self.after_step_hooks = [], []
                self.steps = [object()]

        policy = FakePolicy()
        pre, post = Pipeline(), Pipeline()
        module = SimpleNamespace(make_policy=lambda: policy,
                                 make_pre_post_processors=lambda: (pre, post))
        original_factory = module.make_policy
        original_processors = module.make_pre_post_processors
        original_argv = sys.argv

        def main():
            loaded = module.make_policy()
            actual_pre, actual_post = module.make_pre_post_processors()
            self.assertIs(actual_pre, pre)
            self.assertIs(actual_post, post)
            for hook in pre.before_step_hooks:
                hook(0, {"observation": {"state": torch.tensor([[.4]])}})
            action = loaded.select_action(batch())
            for hook in post.after_step_hooks:
                hook(0, {"action": action * 2})
            self.assertTrue(trace.report["postprocessor_seen"])
            raise RuntimeError("simulation failed")

        module.main = main
        with tempfile.TemporaryDirectory() as directory:
            trace = PolicyTrace(directory)
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                    run_with_trace(module, trace, ["--policy.path=example"])
            self.assertIs(module.make_policy, original_factory)
            self.assertIs(module.make_pre_post_processors, original_processors)
            self.assertIs(sys.argv, original_argv)
            self.assertFalse(pre.before_step_hooks)
            self.assertFalse(post.after_step_hooks)
            self.assertNotIn("select_action", vars(policy))
            self.assertIn("simulation failed", trace.report["error"])


if __name__ == "__main__":
    unittest.main()
