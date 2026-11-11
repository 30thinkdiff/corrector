"""队列交接、原版本续训与预算边界；不启动GPU。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import corrector_followup_queue as follow
import corrector_night_sweep as night


class FollowupTests(unittest.TestCase):
    def fixture(self,root):
        args=follow.parser().parse_args(['run','--output',str(root/'new'),
             '--predecessor-output',str(root/'old'),'--predecessor-code',str(root/'oldcode')])
        args.output.mkdir();args.predecessor_code.mkdir()
        script=args.predecessor_code/'corrector_night_sweep.py';script.write_text('# frozen old version')
        follow.write(args.predecessor_output/'sweep_plan.json',{'jobs':[{'name':'old_s42'}],
            'code_sha256':{script.name:follow.digest(script)},'settings':{'output':str(args.predecessor_output),
            'models':['transformer-h1','mlp','transformer-h3'],'seeds':[42,43,44],'no_eval':False}})
        follow.write(args.predecessor_output/'sweep_status.json',{'status':'paused'})
        return args

    def complete_old(self,args):
        follow.write(args.predecessor_output/'runs/old_s42/JOB_COMPLETE.json',{})
        follow.write(args.predecessor_output/'sweep_status.json',{'status':'completed'})

    def test_frozen_cli_roundtrip_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));cmd=follow.frozen_command(args.predecessor_code,args.predecessor_output,7)
            parsed=night.parser().parse_args(cmd[3:])
            self.assertEqual(parsed.max_compute_hours,7);self.assertEqual(parsed.seeds,[42,43,44])
            self.assertFalse(parsed.no_eval)
            (args.predecessor_code/'corrector_night_sweep.py').write_text('# changed')
            with self.assertRaises(RuntimeError):follow.frozen_command(args.predecessor_code,args.predecessor_output,7)

    def test_wait_resume_original_then_action_without_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));runner=follow.Followup(args);calls=[]
            def child(command,label):
                calls.append(label)
                if label=='resume_original_queue':self.complete_old(args)
                else:
                    self.assertTrue(follow.predecessor_complete(args.predecessor_output))
                    self.assertIn('transformer-a10-h1',command)
                    follow.write(args.output/'experiments/sweep_status.json',{'status':'completed'})
            with patch.object(follow,'predecessor_busy',side_effect=[True,False,False]),patch.object(follow.time,'sleep'), \
                 patch.object(runner,'child',side_effect=child):runner.run()
            self.assertEqual(calls,['resume_original_queue','action_experiments'])
            self.assertEqual(runner.state['status'],'completed')

    def test_completed_original_skipped_and_operator_stop_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));(args.predecessor_output/'STOP').touch()
            with patch.object(follow,'predecessor_busy',return_value=False):
                runner=follow.Followup(args);runner.run()
            self.assertEqual(runner.state['reason'],'predecessor_has_operator_STOP')
            self.assertTrue((args.predecessor_output/'STOP').exists())
            (args.predecessor_output/'STOP').unlink();self.complete_old(args)
            def child(command,label):
                self.assertEqual(label,'action_experiments')
                follow.write(args.output/'experiments/sweep_status.json',{'status':'paused','reason':'budget'})
            with patch.object(follow,'predecessor_busy',return_value=False),patch.object(follow.Followup,'child',side_effect=child):
                runner=follow.Followup(args);runner.run()
            self.assertEqual(runner.state['status'],'paused')

    def test_budget_expires_before_new_stage_no_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));runner=follow.Followup(args)
            with patch.object(runner,'remaining',return_value=0),patch.object(runner,'child') as child:runner.run()
            child.assert_not_called();self.assertEqual(runner.state['status'],'paused')

    def test_predecessor_lock_prevents_double_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            with follow.lock(args.predecessor_output/'.sweep.lock'):
                self.assertTrue(follow.predecessor_busy(args.predecessor_output,args.predecessor_code))
            self.assertFalse(follow.predecessor_busy(args.predecessor_output,args.predecessor_code))

    def test_action_jobs_keep_sample_anchors_and_action_cli(self):
        args=night.parser().parse_args(['plan','--models','transformer-a10-h1','transformer-a10-h3'])
        self.assertEqual(len(night.specifications(args)),6)
        for job in night.specifications(args):
            cmd=night.fit_command(args,job,{'batch':8,'workers':0},'inspect')
            self.assertEqual(cmd[cmd.index('--action-steps')+1],'10')
            self.assertEqual(cmd[cmd.index('--anchor-history')+1],'3')


if __name__=='__main__':unittest.main()
