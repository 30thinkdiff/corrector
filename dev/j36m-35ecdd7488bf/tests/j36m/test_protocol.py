import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from j36m.audit import Accumulator,feature_scale_audit
from j36m.loss import LossConfig
from j36m.plus import plus_plan,selected_task
from j36m.calibration import fit_calibration
from j36m.training import TransitionDataset,scope_loss,check_stats
from j36m.model import LAYERS
# Reuse real completion format fixtures and tests from the preserved MVP.
sys.path.insert(0,str(ROOT/'tests/multilayer_mvp'))
from test_mvp import fixture
from multilayer_mvp.data import validate


class ProtocolTests(unittest.TestCase):
    def test_full_eval_plan_no_view5_leak_into_development(self):
        dev=plus_plan('model','cal','development');held=plus_plan('model','cal','heldout')
        self.assertEqual(len(dev['cases'])+len(held['cases']),48)
        self.assertEqual({c['condition'] for c in dev['cases']},{'default','view15'})
        self.assertEqual({c['condition'] for c in held['cases']},{'view5','view10'})
        self.assertEqual({c['seed'] for c in dev['cases']},{1000,1001,1002})
        for plan in (dev,held):
            for case in plan['cases']:
                self.assertIn('H20',{c['method'] for c in plan['cases']})
                if case['method'].startswith('j_'):self.assertEqual(case['config']['corrector_sha256'],'model')
        self.assertEqual(len(plus_plan('m','c','development',monitor=True)['cases'])+len(plus_plan('m','c','heldout',monitor=True)['cases']),60)
        self.assertEqual({r['seed'] for r in plus_plan('m','c','calibration')['cases']},{9000,9001})
    def test_normal_calibration_freezes_only_independent_successful_runs(self):
        layer={k:{'valid':True,'predicted_rms':1.,'observed_rms':1.,'score':.1} for k in LAYERS}
        trials=[{'condition':'default','seed':s,'normal_success':True,'checkpoint_sha256':'m','training_manifest_sha256':'d',
                 'rows':[{'layer_scores':layer} for _ in range(20)]} for s in (9000,9001)]
        cal=fit_calibration(trials,'m','d');self.assertEqual(cal['status'],'frozen')
        trials[1]['seed']=1000
        with self.assertRaises(ValueError):fit_calibration(trials,'m','d')
    def test_exact_moments_sampled_quantiles_declared(self):
        a=Accumulator();x=np.arange(24,dtype='float32').reshape(2,3,4);a.add(x,0)
        r=a.report();self.assertAlmostEqual(r['rms'],float(np.square(x,dtype=np.float64).mean()**.5))
        self.assertAlmostEqual(r['mean'],float(x.mean()));self.assertIn('stratified',r['quantile_method'])
        self.assertEqual(len(r['per_camera_rms']),2);self.assertEqual(len(r['per_channel_rms']),4)
    def test_scale_audit_no_val_fit_and_single_layer_targets(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(Path(d));records=validate(cfg);out=Path(d)/'audit'
            r=feature_scale_audit(records,out)
            train=TransitionDataset(records,'train','S2');x=train[0]
            self.assertEqual(set(x['inputs']),{'block_02','action'})
            self.assertEqual(set(x['targets']),{'delta_block_02'})
            for k in LAYERS:self.assertAlmostEqual(r['split']['train'][k]['zero_normalized_mse'],1.)
            check_stats(r,records)
            bad=dict(r,source=[])
            with self.assertRaises(ValueError):check_stats(bad,records)
    def test_formal_stats_ignore_validation_values_and_S2_future_not_input(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(Path(d));records=validate(cfg)
            first=feature_scale_audit(records,Path(d)/'first')
            val=Path(cfg['episodes'][2]['raw_path'])/'block_02.fp16.npy'
            a=np.load(val,mmap_mode='r+');a[:]*=20;a.flush()
            second=feature_scale_audit(records,Path(d)/'second')
            self.assertEqual(first['train_only_input_rms'],second['train_only_input_rms'])
            self.assertEqual(first['train_only_delta_rms'],second['train_only_delta_rms'])
            ds=TransitionDataset(records,'train','S2');before=ds[0]
            a=np.load(Path(cfg['episodes'][0]['raw_path'])/'block_02.fp16.npy',mmap_mode='r+')
            a[10]+=10;a.flush();after=ds[0]
            np.testing.assert_array_equal(before['inputs']['block_02'],after['inputs']['block_02'])
            self.assertFalse(np.array_equal(before['targets']['delta_block_02'],after['targets']['delta_block_02']))

    def test_trace_coverage_counts_real_step_rows_and_skips_warmup(self):
        from j36m.reporting import trace_coverage
        rows=[{'step':i,'score':None if i<20 or i in (23,24) else .5} for i in range(30)]
        rows+=[{'step':23,'event':'new_action_plan'}]
        result=trace_coverage(rows)
        self.assertEqual(result['eligible_steps_after_warmup'],10)
        self.assertEqual(result['scored_steps'],8);self.assertEqual(result['longest_unscored_run_steps'],2)
        self.assertEqual(result['coverage_after_warmup'],.8)
        truncated=[r for r in rows if r.get('step')!=25]
        incomplete=trace_coverage(truncated,expected_steps=30)
        self.assertIsNone(incomplete['coverage_after_warmup']);self.assertIn(25,incomplete['missing_trace_steps'])

    def test_loss_choices_frozen_and_serializable(self):
        self.assertEqual(scope_loss('J').weights,(.25,.25,.5))
        self.assertEqual(scope_loss('JC').weights,(0.,0.,1.))
        self.assertEqual(scope_loss('J_equal').weights,(1/3,)*3)
        json.dumps(scope_loss('J').serializable(),allow_nan=False)

if __name__=='__main__':unittest.main()
