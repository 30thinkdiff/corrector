import io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_progress_report import summarize_phase,report


class ProgressMetrics(unittest.TestCase):
    def make(self,duplicate=False):
        buf=io.BytesIO();prefix='plus/plus/'
        cases=[{'id':m,'method':m,'condition':'plus/test','seed':1,'config':{}} for m in ['H20','j_ogg']]
        with zipfile.ZipFile(buf,'w') as z:
            def write(n,v):z.writestr(prefix+n,json.dumps(v))
            write('plan.json',{'cases':cases,'protocol':{'first_due_step':24}})
            write('execution.json',{'status':'running','planned':2})
            for c in cases:
                if c['method']=='H20':r={'case':c,'status':'running','attempt':1}
                else:
                    r={'case':c,'status':'completed','attempt':1,'wall_s':3.,'result':{'success':True,'env_steps':26,'guidance_calls':1,'episode_wall_s':2.}}
                    write('attempts/j_ogg/attempt_001/actual_protocol.json',{'monitor_first_due_step':24})
                    rows=[{'step':24,'score':.3},{'step':25,'score':None},
                          {'step':24,'event':'joint_ogg','loss':.8,'gradient_norm':2.,'velocity_update_norm':.5,'candidate_action_update_norm':.25,'latency_s':.01}]
                    z.writestr(prefix+'attempts/j_ogg/attempt_001/monitor_steps.jsonl','\n'.join(json.dumps(r) for r in rows))
                write('trials/'+c['id']+'.json',r)
            if duplicate:write('trials/duplicate.json',r)
        buf.seek(0);return zipfile.ZipFile(buf)
    def test_running_is_not_failure_and_common_pairs_not_inferred(self):
        with self.make() as z:d=summarize_phase(z,'plus/plus/')
        h=next(r for r in d['methods'] if r['method']=='H20');j=next(r for r in d['methods'] if r['method']=='j_ogg')
        self.assertIsNone(h['success_rate']);self.assertIsNone(h['coverage']);self.assertEqual(d['common_paired_groups'],0)
        self.assertEqual(j['coverage'],.5);self.assertEqual(j['max_gap_steps'],1)
        self.assertEqual(j['mean_callback_wall_s'],3.);self.assertEqual(j['mean_episode_wall_s'],2.)
        self.assertEqual(j['mean_ogg_gradient_l2'],2.);self.assertEqual(j['mean_ogg_clean_action_update_l2'],.25)
        self.assertEqual(d['snapshot_running'],1)
    def test_duplicate_trial_is_rejected(self):
        with self.make(True) as z,self.assertRaisesRegex(ValueError,'duplicate'):summarize_phase(z,'plus/plus/')


@unittest.skipUnless((ROOT/'outputs/corrector_progress_snapshots/20261113T150640Z.zip').exists(),'requires real remote snapshot')
class RealProgress(unittest.TestCase):
    def test_completed_discovery_and_partial_replication_stay_separate(self):
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/'report';report(ROOT/'outputs/corrector_progress_snapshots/20261113T150640Z.zip',dest)
            d=json.loads((dest/'summary.json').read_text());main=d['phases']['plus/plus'];partial=d['phases']['plus/plus_replicates']
            self.assertEqual(main['snapshot_completed'],544);self.assertEqual(partial['snapshot_completed'],24)
            self.assertEqual(main['common_paired_groups'],32)
            methods={r['method']:r for r in main['methods']}
            self.assertEqual(methods['H20']['successes'],20);self.assertEqual(methods['j_ogg']['successes'],9)
            self.assertEqual(methods['H20']['completed'],32);self.assertEqual(methods['j_lvm']['coverage'],1.)
            self.assertLess(methods['transformer-a10-h1_ogg']['coverage'],.6)
            self.assertTrue(all(r['epochs_completed']==60 for r in d['training'] if r['status']=='completed'))
            self.assertGreater(next(r for r in d['training'] if r['run']=='J_s42')['mean_epoch_s'],200)
            self.assertTrue((dest/'training_layers.csv').is_file())
