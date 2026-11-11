"""CPU 检查 Notebook 的辅助采集，不需要安装 LIBERO 或加载模型。

运行：python -m unittest studies.policy_generalization.test_grasp_capture
接口替身不能替代 Notebook 14a 的真实 Plus 检查。
"""
import json
from pathlib import Path
import tempfile
from collections import deque, namedtuple
from types import SimpleNamespace as NS
import unittest
import sys
from contextlib import redirect_stdout
import io

import numpy as np
import torch


NOTEBOOK = Path(__file__).resolve().parents[2] / "scripts/libero_plus_camera_check.ipynb"


def definitions():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "camera-grasp-capture")["source"])
    namespace = {}
    exec(compile(source, str(NOTEBOOK) + "#camera-grasp-capture", "exec"), namespace)
    return namespace


def fake_environment():
    names = ["target", "left", "right", "hand"]
    model = NS(
        body_name2id=lambda name: 0,
        site_name2id=lambda name: 0,
        geom_name2id=names.index,
        geom_id2name=names.__getitem__,
    )
    data = NS(
        time=0.5,
        site_xpos=np.array([[0.1, 0.2, 0.3]]),
        site_xmat=np.eye(3).reshape(1, 9),
        xpos=np.array([[0.11, 0.21, 0.2]]),
        xquat=np.array([[1., 0., 0., 0.]]),
        qpos=np.array([0.02, -0.02]), ncon=0, contact=[],
    )

    class Sim:
        def __init__(self):
            self.data = NS(_data=data)
            self.model = model
            self.events = []
            self.calls = 0

        def step(self):
            event = self.events.pop(0) if self.events else []
            data.contact = [NS(geom1=a, geom2=b, dist=dist, efc_address=address)
                            for a, b, dist, address in event]
            data.ncon = len(data.contact)
            data.time += 0.025
            self.calls += 1

    sim = Sim()
    gripper = NS(
        important_sites={"grip_site": "grip_site"},
        important_geoms={"left_finger": ["left"], "right_finger": ["right"]},
        contact_geoms=["left", "right", "hand"],
    )
    robot = NS(gripper=gripper, eef_site_id=0, _ref_gripper_joint_pos_indexes=[0, 1])
    core = NS(robots=[robot], objects_dict={"alphabet_soup_1": NS(root_body="target", contact_geoms=["target"])})
    env = NS(_env=NS(sim=sim, env=core), control_freq=20)
    return env, sim, data


class GraspCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = definitions()

    def test_transient_contacts_and_original_return(self):
        env, sim, data = fake_environment()
        with tempfile.TemporaryDirectory() as directory:
            capture = self.namespace["GraspCapture"](env, directory, max_steps=1)
            action = np.arange(7, dtype=np.float64)
            result = ({"original": True}, 0., False, False, {})
            # 左指只在第一个子步接触；第二个子步改为右指，不能当作同时双侧。
            sim.events = [[(0, 1, -0.001, 0)], [(2, 0, -0.001, 1)]]

            def original_step(actual_env, actual_action):
                self.assertIs(actual_env, env)
                self.assertIs(actual_action, action)
                sim.step()
                sim.step()
                return result

            self.assertIs(capture.step(original_step, env, action), result)
            self.assertNotIn("step", vars(sim))
            self.assertEqual(sim.calls, 2)
            capture.flush("test")
            trace = torch.load(Path(directory) / "grasp_trace.pt", weights_only=True)
            self.assertEqual(trace["gripper_xyz"].shape, (2, 3))
            self.assertEqual(trace["executed_action"].shape, (1, 7))
            self.assertEqual(trace["contact_during_step"][1].tolist(), [True, True, False])
            self.assertEqual(trace["contact_now"][1].tolist(), [False, True, False])
            self.assertFalse(trace["both_fingers_same_physics_sample"][1].item())
            self.assertEqual(trace["physics_samples"].tolist(), [0, 2])
            np.testing.assert_array_equal(action, np.arange(7))
            data.site_xpos[:] = 99
            self.assertEqual(capture.rows[0]["gripper_xyz"][0], 0.1)  # 独立快照
            capture.step(original_step, env, action)  # 超出采集窗仍正常推进
            self.assertEqual(len(capture.rows), 2)
            self.assertEqual(sim.calls, 4)

    def test_same_sample_contacts_hand_and_inactive_constraints(self):
        env, sim, data = fake_environment()
        with tempfile.TemporaryDirectory() as directory:
            capture = self.namespace["GraspCapture"](env, directory)
            sim.events = [[(0, 1, -0.001, 0), (0, 2, -0.001, 1), (0, 3, -0.001, 2)]]
            capture.step(lambda e, a: sim.step(), env, np.zeros(7))
            self.assertTrue(capture.rows[-1]["both_fingers_same_physics_sample"])
            self.assertEqual(capture.rows[-1]["contact_now"].tolist(), [True, True, True])
            data.contact = [NS(geom1=0, geom2=1, dist=-0.001, efc_address=-1),
                            NS(geom1=0, geom2=2, dist=0.01, efc_address=0)]
            data.ncon = 2
            self.assertFalse(capture.contacts().any())

    def test_exception_restores_sim_and_keeps_readable_partial_trace(self):
        env, sim, data = fake_environment()
        with tempfile.TemporaryDirectory() as directory:
            capture = self.namespace["GraspCapture"](env, directory)

            def broken(e, a):
                sim.step()
                raise RuntimeError("physics failure")

            with self.assertRaisesRegex(RuntimeError, "physics failure"):
                capture.step(broken, env, np.zeros(7))
            self.assertNotIn("step", vars(sim))
            capture.flush("interrupted")
            trace = torch.load(Path(directory) / "grasp_trace.pt", weights_only=True)
            self.assertEqual(trace["frame_index"].tolist(), [0])
            self.assertEqual(trace["executed_action"].shape, (0, 7))

    def test_missing_physics_hook_is_rejected(self):
        env, sim, data = fake_environment()
        with tempfile.TemporaryDirectory() as directory:
            capture = self.namespace["GraspCapture"](env, directory)
            with self.assertRaisesRegex(RuntimeError, "sim.step"):
                capture.step(lambda e, a: None, env, np.zeros(7))

    def test_archive_retry_and_conflict_preserve_data(self):
        archive = self.namespace["archive_camera_run"]
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "drive"
            source.mkdir()
            (source / "trace").write_bytes(b"original")
            archive(source, target)
            archive(source, target)
            (source / "trace").write_bytes(b"changed")
            with self.assertRaises(FileExistsError):
                archive(source, target)
            self.assertEqual((target / "trace").read_bytes(), b"original")

    def test_runner_capture_resume_archive_and_exception_restoration(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = "".join(next(c for c in notebook["cells"] if c["id"] == "camera-pilot-runner")["source"])
        source = "\n".join(line for line in source.splitlines()
                           if not line.startswith(("from transformers ", "from lerobot.")))
        instruction = "pick the alphabet soup and place it in the basket"
        base = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
        names = [None] * 667
        names[653] = base + "_view_1_15_100_0_0_initstate_0"
        names[656] = base + "_view_5_15_100_0_0_initstate_0"
        names[661] = base + "_view_10_15_100_0_0_initstate_0"
        names[666] = base + "_view_15_15_100_0_0_initstate_0"
        Task = namedtuple("Task", "name language bddl_file init_states_file")
        tasks = {name: Task(name, "original language", name + ".bddl", name + ".pruned_init")
                 for name in (names[653], names[656], names[661], names[666])}
        originals = tasks.copy()
        control = {"calls": 0, "fail": False}
        test = self

        class RobotModel:
            pass

        class Env:
            def __init__(self, task_id, **kwargs):
                env, sim, data = fake_environment()
                self._env = env._env
                self.control_freq = 20
                self.control_mode = "relative"
                self.is_libero_plus = True
                self.num_steps_wait = 10
                self.init_state_id = 0
                self._init_states = [np.array([3., 4.])]
                task = tasks[names[task_id]]
                self.task = task.name
                self.task_description = task.language
                self._env.language_instruction = instruction
                self._env.robots = self._env.env.robots
                self._env.robots[0].robot_model = RobotModel()
                self._env.check_success = lambda: False
                sim.model.camera_name2id = lambda name: 0
                sim.model.cam_fovy = [45.]
                data.cam_xpos = np.zeros((1, 3))
                data.cam_xmat = np.eye(3).reshape(1, 9)
                data.qvel = np.zeros(2)
                data.act = np.empty(0)
                data.mocap_pos = np.empty((0, 3))
                data.mocap_quat = np.empty((0, 4))

            def reset(self, seed=None, **kwargs):
                return {}, {}

            def step(self, action):
                test.assertEqual(action.shape, (7,))
                self._env.sim.step()
                self._env.sim.step()
                return {}, 0., False, False, {}

            def close(self):
                control["closed"] = True

        class Flow:
            def sample_noise(self, shape, device):
                return torch.randn(shape, device=device)

        class Policy:
            def __init__(self, action_steps=50):
                self.model = Flow()
                tokenizer = NS(decode=lambda ids, **kw: instruction + "\n")
                self.model.vlm_with_expert = NS(processor=NS(tokenizer=tokenizer))
                self.config = NS(n_action_steps=action_steps, chunk_size=50, num_steps=10, rtc_config=None)
                self._queues = {"action": deque(maxlen=action_steps)}

            def parameters(self):
                return iter([torch.zeros(1)])

            def _get_action_chunk(self, batch):
                self.model.sample_noise((1, 50, 32), "cpu")
                return np.zeros((50, 7))

            def select_action(self, batch):
                if not self._queues["action"]:
                    self._queues["action"].extend(self._get_action_chunk(batch)[:self.config.n_action_steps])
                return self._queues["action"].popleft()

        class Loader:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                test.assertNotIn("torch_dtype", kwargs)
                test.assertEqual(kwargs["dtype"], "bfloat16")
                test.assertIs(kwargs["token"], False)

        reset, step, select, noise = Env.reset, Env.step, Policy.select_action, Flow.sample_noise
        get_chunk = Policy._get_action_chunk
        reference = {"qpos": torch.tensor([0.02, -0.02], dtype=torch.float64),
                     "qvel": torch.zeros(2, dtype=torch.float64),
                     "act": torch.empty(0, dtype=torch.float64),
                     "mocap_pos": torch.empty(0, 3, dtype=torch.float64),
                     "mocap_quat": torch.empty(0, 4, dtype=torch.float64),
                     "selected_init": torch.tensor([3., 4.], dtype=torch.float64)}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            archive = Path(directory) / "drive"

            def main():
                control["calls"] += 1
                args = dict(s[2:].split("=", 1) for s in sys.argv[1:])
                torch.manual_seed(int(args["seed"]))
                Loader.from_pretrained("public-model", torch_dtype="bfloat16")
                env = Env(json.loads(args["env.task_ids"])[0])
                env.reset(seed=int(args["seed"]))
                policy = ns["lerobot_eval"].make_policy()
                batch = {"observation.language.tokens": torch.tensor([[11, 2]]),
                         "observation.language.attention_mask": torch.tensor([[True, False]])}
                for _ in range(25 if "policy.n_action_steps" in args else 3):
                    env.step(policy.select_action(batch))
                if control["fail"]:
                    raise RuntimeError("intentional evaluation error")
                (Path(args["output_dir"]) / "eval_info.json").write_text(
                    json.dumps({"overall": {"n_episodes": 1, "pc_success": 100.}}))

            ns = {**self.namespace, "benchmark": NS(libero_task_map={"libero_object": names},
                                                   task_maps={"libero_object": tasks}),
                  "LiberoEnv": Env, "SmolVLAPolicy": Policy, "VLAFlowMatching": Flow,
                  "AutoModelForImageTextToText": Loader, "AutoProcessor": type("Processor", (), {}),
                  "AutoTokenizer": type("Tokenizer", (), {}), "lerobot_eval": NS(main=main),
                  "sys": sys, "pilot_base_task": base, "pilot_instruction": instruction,
                  "PILOT_ROOT": root, "CHECKPOINT": "test", "checkpoint_revision": "sha",
                  "checkpoint_snapshot": root / "checkpoint", "PLUS_SHA": "plus", "ASSET_REVISION": "asset",
                  "AUDIT_DIR": root / "reference", "pilot_reference": reference,
                  "pilot_audit": {"tasks": [{"robot": "RobotModel"}]}}

            def make_policy():
                args = dict(s[2:].split("=", 1) for s in sys.argv[1:])
                steps = int(args.get("policy.n_action_steps", 50))
                return Policy(50 if control.get("wrong_config") else steps)

            ns["lerobot_eval"].make_policy = make_policy
            for cls in [ns["AutoProcessor"], ns["AutoTokenizer"]]:
                cls.from_pretrained = lambda *a, **kw: None
            exec(compile(source, str(NOTEBOOK) + "#camera-pilot-runner", "exec"), ns)
            ns["version"] = lambda name: "test-version"
            run = ns["run_controlled_camera_pilot"]
            with redirect_stdout(io.StringIO()):
                for condition in ["default_camera", "view_15"]:
                    run(condition, capture_grasp=True, run_root=root, archive_root=archive)
                self.assertEqual(control["calls"], 2)
                run("view_15", capture_grasp=True, run_root=root, archive_root=archive)
                self.assertEqual(control["calls"], 2)  # 完整结果不会重跑
                hashes = [json.loads((root / cond / "seed_1000/noise_metadata.json").read_text())["sha256"]
                          for cond in ["default_camera", "view_15"]]
                self.assertEqual(hashes[0], hashes[1])
                trace = torch.load(archive / "view_15/seed_1000/grasp_trace.pt", weights_only=True)
                self.assertEqual(trace["gripper_xyz"].shape, (4, 3))
                self.assertEqual(trace["physics_samples"].tolist(), [0, 2, 2, 2])
                control["fail"] = True
                with self.assertRaisesRegex(RuntimeError, "intentional evaluation error"):
                    run("view_15", seed=1001, capture_grasp=True, run_root=root, archive_root=archive)
            interrupted = json.loads((archive / "view_15/seed_1001/grasp_schema.json").read_text())
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertTrue((archive / "view_15/seed_1001/run_error.json").exists())
            self.assertEqual(tasks, originals)
            self.assertIs(Env.reset, reset)
            self.assertIs(Env.step, step)
            self.assertIs(Policy.select_action, select)
            self.assertIs(Flow.sample_noise, noise)
            # 真实入口前的接口检查必须使用 ndarray，并无条件恢复 Task。
            import ast
            preflight = "".join(next(c for c in notebook["cells"] if c["id"] == "camera-grasp-preflight")["source"])
            preflight = preflight.replace("        from lerobot.envs.libero import get_libero_dummy_action", "")
            function = next(node for node in ast.parse(preflight).body if isinstance(node, ast.FunctionDef))
            ns.update(GRASP_ROOT=root, io=io, redirect_stdout=redirect_stdout,
                      get_libero_dummy_action=lambda: [0, 0, 0, 0, 0, 0, -1])
            ns["benchmark"].get_benchmark_dict = lambda: {"libero_object": lambda: NS()}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "preflight", "exec"), ns)
            with redirect_stdout(io.StringIO()):
                ns["check_grasp_interfaces"]()
            self.assertTrue(control["closed"])
            self.assertEqual(tasks, originals)
            self.assertTrue((root / "interface_check/grasp_trace.pt").is_file())

            # 新执行窗口必须进入独立目录，真正作用于队列，且记录实际预测次数。
            control["fail"] = False
            replan = Path(directory) / "replan"
            replan_archive = Path(directory) / "replan-drive"
            with redirect_stdout(io.StringIO()):
                for action_steps in [50, 10]:
                    run("view_15", seed=1002, capture_grasp=True, run_root=replan,
                        archive_root=replan_archive, n_action_steps=action_steps)
                completed_calls = control["calls"]
                run("view_15", seed=1002, capture_grasp=True, run_root=replan,
                    archive_root=replan_archive, n_action_steps=10)
                self.assertEqual(control["calls"], completed_calls)
            for action_steps, expected_calls in [(50, 1), (10, 3)]:
                run_dir = replan_archive / f"n_action_steps_{action_steps}/view_15/seed_1002"
                actual = json.loads((run_dir / "actual_policy_config.json").read_text())
                timing = json.loads((run_dir / "timing.json").read_text())
                self.assertEqual(actual, {"n_action_steps": action_steps, "chunk_size": 50, "num_steps": 10})
                self.assertEqual(timing["env_steps"], 25)
                self.assertEqual(timing["chunk_calls"], expected_calls)
                self.assertGreater(timing["chunk_wall_s_total"], 0)
                self.assertEqual(timing["execution_window_s"], action_steps / 20)
            with self.assertRaises(ValueError):
                run("view_15", n_action_steps=51, capture_grasp=True)
            with self.assertRaises(ValueError):
                run("view_15", n_action_steps=10, capture_grasp=False)
            # 模拟 CLI 未真正改变加载配置；必须在执行动作前拒绝。
            control["wrong_config"] = True
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "实际策略配置不符"):
                    run("view_15", seed=1003, capture_grasp=True, run_root=replan,
                        archive_root=replan_archive, n_action_steps=10)
            self.assertIs(ns["lerobot_eval"].make_policy, make_policy)
            self.assertIs(Policy._get_action_chunk, get_chunk)
            self.assertIs(Policy.select_action, select)
            self.assertIs(Env.step, step)
            self.assertEqual(tasks, originals)
            # 复用缓存回合完成一对的逐元素核对与配对审计归档。
            control["wrong_config"] = False
            pair_source = "".join(next(c for c in notebook["cells"] if c["id"] == "camera-replanning-definitions")["source"])
            pair_source = pair_source.replace('    assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Drive"', '')
            pair_function = next(node for node in ast.parse(pair_source).body if isinstance(node, ast.FunctionDef))
            ns.update(REPLAN_ROOT=replan, REPLAN_ARCHIVE=replan_archive, GRASP_INTERFACES_VERIFIED=True)
            exec(compile(ast.Module(body=[pair_function], type_ignores=[]), "replanning-pair", "exec"), ns)
            calls_before_pair = control["calls"]
            with redirect_stdout(io.StringIO()):
                ns["run_replanning_pair"]("view_15", 1002)
            self.assertEqual(control["calls"], calls_before_pair)
            checks = json.loads((replan_archive / "pair_checks/view_15_seed_1002.json").read_text())
            self.assertTrue(all(checks.values()))
            self.assertEqual(set(checks), {"initial_state_equal", "first_noise_equal", "language_tokens_equal", "camera_equal"})
            # 接触均为 False 的接口替身数据：仅测试用户函数回调与汇总接入，不提供练习答案。
            import csv
            ns.update(first_true_time=lambda time, mask: None, csv=csv)
            summary_source = "".join(next(c for c in notebook["cells"] if c["id"] == "camera-contact-summary")["source"])
            summary_functions = [node for node in ast.parse(summary_source).body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=summary_functions, type_ignores=[]), "summary-functions", "exec"), ns)
            replanning_summary = "".join(next(c for c in notebook["cells"] if c["id"] == "camera-replanning-summary")["source"])
            replanning_functions = [node for node in ast.parse(replanning_summary).body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=replanning_functions, type_ignores=[]), "replanning-summary", "exec"), ns)
            with redirect_stdout(io.StringIO()):
                records = ns["summarize_replanning_experiment"]()
            self.assertEqual(len(records), 2)
            self.assertEqual({row["n_action_steps"] for row in records}, {50, 10})
            self.assertEqual({row["chunk_calls"] for row in records}, {1, 3})
            self.assertTrue((replan_archive / "analysis/replanning_summary.csv").is_file())
            # 未通过配对核对的回合不能悄悄进入对照汇总。
            (replan / "pair_checks/view_15_seed_1002.json").write_text(json.dumps({"first_noise_equal": False}))
            with self.assertRaisesRegex(RuntimeError, "配对尚未验证通过"):
                ns["summarize_replanning_experiment"]()
            # 20 步预算探测只新增候选回合，读取已有 50 步结果作为基线。
            budget_source = "".join(next(c for c in notebook["cells"] if c["id"] == "camera-budget-probe")["source"])
            budget_source = budget_source.replace('    assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Drive"', '')
            budget_function = next(node for node in ast.parse(budget_source).body if isinstance(node, ast.FunctionDef))
            ns.update(BUDGET_ROOT=Path(directory) / "budget", BUDGET_ARCHIVE=Path(directory) / "budget-drive")
            exec(compile(ast.Module(body=[budget_function], type_ignores=[]), "budget-probe", "exec"), ns)
            calls_before_budget = control["calls"]
            with redirect_stdout(io.StringIO()):
                ns["run_budget_probe"](1002)
            self.assertEqual(control["calls"], calls_before_budget + 1)
            budget_audit = json.loads((ns["BUDGET_ARCHIVE"] / "pair_checks/view_15_seed_1002_50_vs_20.json").read_text())
            self.assertTrue(all(budget_audit["checks"].values()))
            self.assertTrue((ns["BUDGET_ARCHIVE"] / "analysis/view_15_seed_1002_50_vs_20.csv").is_file())
            # 冻结保留协议、50/20 配对、进度表与已完成回合的复用。
            heldout_source = "".join(next(c for c in notebook["cells"] if c["id"] == "camera-heldout-definitions")["source"])
            heldout_source = heldout_source.replace('    assert Path("/content/drive/MyDrive").is_dir(), "先挂载 Drive"', '')
            exec(compile(heldout_source, "heldout-definitions", "exec"), ns)
            ns.update(HELDOUT_ROOT=Path(directory) / "heldout", HELDOUT_ARCHIVE=Path(directory) / "heldout-drive")
            before_heldout = control["calls"]
            with redirect_stdout(io.StringIO()):
                ns["run_heldout_pair"]("view_10", 1000)
                ns["run_heldout_pair"]("view_10", 1000)
            self.assertEqual(control["calls"], before_heldout + 2)
            heldout_checks = json.loads((ns["HELDOUT_ARCHIVE"] / "pair_checks/view_10_seed_1000_50_vs_20.json").read_text())
            self.assertTrue(all(heldout_checks.values()))
            self.assertTrue(heldout_checks["actual_settings_match"])
            self.assertEqual(ns["controlled_camera_task"]("view_5")[0], 656)
            self.assertEqual(ns["controlled_camera_task"]("view_10")[0], 661)
            with self.assertRaises(ValueError):
                ns["run_heldout_pair"]("view_15", 1000)
            heldout_summary = "".join(next(c for c in notebook["cells"] if c["id"] == "camera-heldout-summary")["source"])
            summary_function = next(node for node in ast.parse(heldout_summary).body if isinstance(node, ast.FunctionDef))
            exec(compile(ast.Module(body=[summary_function], type_ignores=[]), "heldout-summary", "exec"), ns)
            with redirect_stdout(io.StringIO()):
                report = ns["summarize_heldout_validation"]()
            self.assertEqual(report["completed_pairs"], 1)
            self.assertEqual(report["expected_pairs"], 6)
            self.assertFalse(report["validation_complete"])
            self.assertEqual({g["n_action_steps"] for g in report["groups"]}, {50, 20})
            self.assertTrue((ns["HELDOUT_ARCHIVE"] / "analysis/heldout_summary_1pairs.csv").is_file())
            ns["FROZEN_HELDOUT_SEEDS"] = (1000,)
            with self.assertRaisesRegex(RuntimeError, "已有保留验证协议不同"):
                ns["freeze_heldout_protocol"]()
            self.assertEqual(tasks, originals)


if __name__ == "__main__":
    unittest.main()
