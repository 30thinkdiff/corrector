import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .batch_evaluation import make_plan, run_plan, set_recording_fps, summarize


class BatchEvaluationTests(unittest.TestCase):
    def test_budget_pauses_at_episode_boundary_and_resume_keeps_plan(self):
        plan=make_plan(['a'],[1,2],{'H50':{'video_policy':'none'}},{})
        clock=[0.];calls=[]
        def callback(case,directory):
            calls.append(case['id']);clock[0]+=2.
            return {'success':True,'video_paths':[]}
        with tempfile.TemporaryDirectory() as tmp,patch('studies.policy_generalization.corrector.batch_evaluation.time.perf_counter',side_effect=lambda:clock[0]):
            root=Path(tmp)
            first=run_plan(root,plan,{'H50':callback},max_seconds=1.)
            self.assertEqual(first[0]['completed'],1)
            self.assertEqual(json.loads((root/'execution.json').read_text())['reason'],'time_budget')
            second=run_plan(root,plan,{'H50':callback},max_seconds=10.)
            self.assertEqual(second[0]['completed'],2)
            self.assertEqual(len(calls),2)
            self.assertEqual(json.loads((root/'execution.json').read_text())['status'],'completed')

    def test_stop_request_and_disk_floor_do_not_count_as_robot_failure(self):
        from types import SimpleNamespace
        plan=make_plan(['a'],[1],{'H50':{'video_policy':'none'}},{})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stop=root/'STOP';stop.touch()
            callback=lambda *_: {'success':True,'video_paths':[]}
            rows=run_plan(root,plan,{'H50':callback},stop_file=stop)
            self.assertEqual(rows[0]['completed'],0)
            self.assertEqual(rows[0]['errors'],0)
            self.assertEqual(json.loads((root/'execution.json').read_text())['reason'],'operator_stop_file')
            stop.unlink()
            with patch('studies.policy_generalization.corrector.batch_evaluation.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
                rows=run_plan(root,plan,{'H50':callback},min_free_gib=1.)
            self.assertEqual(rows[0]['completed'],0)
            self.assertEqual(json.loads((root/'execution.json').read_text())['reason'],'disk_space')
            self.assertEqual(run_plan(root,plan,{'H50':callback})[0]['completed'],1)

    def test_optional_video_policy_keeps_failure_evidence_requirement(self):
        for policy,success,allowed in [('none',False,True),('failures',True,True),('failures',False,False)]:
            plan=make_plan(['v'],[1],{'H50':{'video_policy':policy}},{})
            with tempfile.TemporaryDirectory() as tmp:
                callback=lambda *_:{'success':success,'video_paths':[]}
                if allowed:
                    rows=run_plan(tmp,plan,{'H50':callback})
                    self.assertEqual(rows[0]['completed'],1)
                else:
                    with self.assertRaises(ValueError):run_plan(tmp,plan,{'H50':callback})
    def test_overall_rate_weights_episodes_and_records_coverage(self):
        plan = make_plan(["a", "b"], [1, 2, 3], {"H50": {}}, {})
        records = [{"case": plan["cases"][i], "status": "completed",
                    "result": {"success": value, "chunk_calls": 2}}
                   for i, value in [(0, True), (1, True), (2, False), (3, True)]]
        with tempfile.TemporaryDirectory() as tmp:
            summarize(records, tmp, plan)
            total = json.loads((Path(tmp) / "summary.json").read_text())["overall_by_method"][0]
            self.assertEqual(total["success_rate"], 3 / 4)
            self.assertEqual(total["completion_rate"], 4 / 6)
            self.assertEqual(total["mean_chunk_calls"], 2)
            self.assertIsNone(total["mean_monitor_s"])

    def test_recording_fps_only_changes_instance_metadata(self):
        class Env:
            metadata = {"render_fps": 80}
        env = Env()
        env.envs = [Env()]
        set_recording_fps(env, 20)
        self.assertEqual(env.metadata["render_fps"], 20)
        self.assertEqual(env.envs[0].metadata["render_fps"], 20)
        self.assertEqual(Env.metadata["render_fps"], 80)

    def test_error_is_not_a_robot_failure(self):
        plan = make_plan(["view_5"], [1, 2, 3], {"H50": {}}, {})
        records = [{"case": plan["cases"][0], "status": "completed", "result": {"success": False}},
                   {"case": plan["cases"][1], "status": "error", "error": "disconnect"}]
        with tempfile.TemporaryDirectory() as tmp:
            rows = summarize(records, tmp, plan)
            self.assertEqual(rows[0]["completed"], 1)
            self.assertEqual(rows[0]["errors"], 1)
            self.assertEqual(rows[0]["pending"], 1)
            self.assertEqual(rows[0]["success_rate"], 0.)

    def test_resume_skips_verified_trials_but_rejects_damaged_video(self):
        plan = make_plan(["view_5"], [1], {"H50": {}}, {"scope": "test fixture, not policy performance"})
        calls = []
        def callback(case, directory):
            calls.append(case["id"])
            video = directory / "fixture.mp4"
            video.write_bytes(b"not a real video; filesystem contract fixture only")
            return {"success": True, "video_paths": [str(video)]}
        with tempfile.TemporaryDirectory() as tmp:
            run_plan(tmp, plan, {"H50": callback})
            run_plan(tmp, plan, {"H50": callback})
            self.assertEqual(len(calls), 1)
            video = next(Path(tmp).rglob("fixture.mp4"))
            video.write_bytes(b"damaged")
            with self.assertRaises(RuntimeError):
                run_plan(tmp, plan, {"H50": callback})

    def test_interrupted_trial_is_retried_with_new_attempt(self):
        plan = make_plan(["v"], [1], {"H50": {}}, {})
        def interrupted(case, directory):
            (directory / "partial.log").write_text("some real progress would go here")
            raise RuntimeError("connection lost")
        def finished(case, directory):
            video = directory / "fixture.mp4"
            video.write_bytes(b"filesystem fixture, no real rollout")
            return {"success": False, "video_paths": [str(video)]}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_plan(tmp, plan, {"H50": interrupted})
            status = Path(tmp) / "trials" / f"{plan['cases'][0]['id']}.json"
            self.assertEqual(json.loads(status.read_text())["status"], "error")
            run_plan(tmp, plan, {"H50": finished})
            self.assertEqual(json.loads(status.read_text())["attempt"], 2)
            self.assertEqual(len(list(Path(tmp).rglob("partial.log"))), 1)

    def test_corrector_cannot_silently_fall_back_to_baseline(self):
        plan = make_plan(["v"], [1], {"event_replan": {"uses_corrector": True}}, {})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_plan(tmp, plan, {})
            with self.assertRaises(RuntimeError):
                run_plan(tmp, plan, {"event_replan": lambda *_: None})
            self.assertFalse((Path(tmp) / "plan.json").exists())

    def test_duplicate_records_rejected(self):
        plan = make_plan(["v"], [1], {"H50": {}}, {})
        record = {"case": plan["cases"][0], "status": "running"}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            summarize([record, record], tmp, plan)
