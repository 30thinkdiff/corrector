"""合成结果测试：覆盖率、配对分母、只读结果导出与完整性；不是GPU实验。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import corrector_results_bundle as bundle
import corrector_results_analysis as analysis


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(root):
    run=root/'runs/mlp_s42';run.mkdir(parents=True)
    (run/'best_model.pt').write_bytes(b'synthetic unit test weight placeholder')
    (run/'latest.pt').write_bytes(b'synthetic latest, not a model')
    digest=bundle.sha(run/'best_model.pt')
    arrays={'train':np.array([0,1],dtype='<i8'),'validation':np.array([2],dtype='<i8')}
    split={'mode':'episode','samples':{'train':2,'validation':1},'episodes':{'train':[1],'validation':[2]},
        'index_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
    np.savez(run/'split_indices.npz',**arrays)
    protocol={'model':{},'model_type':'mlp','h_window':1,'anchor_history':1,'k_step':1,'fps':10,
        'prediction_seconds':.1,'split':split,'seed':42,'batch_size':1,'accumulate':2,'source_sha':'synthetic',
        'encoder_policy_revision':'synthetic-policy'}
    val={'loss':1.,'mse':.8,'cosine_loss':.2,'zero_prediction_mse':2.}
    for name,value in {'protocol.json':protocol,'training_status.json':{'status':'completed'},
        'result.json':{'checkpoint_reload_verified':True,'best_validation':val,'parameters':3,'epochs_completed':1,'optimizer_steps':1},
        'history.json':[{'epoch':1,'step':1,'train_loss':1.1,'train_mse':.9,'validation':val,'seconds':1.}],
        'JOB_COMPLETE.json':{'checkpoint_sha256':digest}}.items():write(run/name,value)
    report=root/'reports/mlp_s42'/digest[:16]
    write(report/'summary.json',{'checkpoint_sha256':digest,'checkpoint_epoch':1,'split':split,'parameters':3,
        'metrics':{'mse':.8,'cosine':.8,'zero_mse':2.,'train_mean_mse':1.9}})
    evaluation=root/'evaluation/mlp_s42'/digest[:16]/'paired'
    cases=[]
    for seed in (1000,1001):
        for method in ('H50','H20','lvm','ogg'):
            case={'id':str(seed)+'_'+method,'method':method,'condition':'libero_object/task_0/init_0','seed':seed,
                'config':{'uses_corrector':method in {'lvm','ogg'},'corrector_sha256':digest if method in {'lvm','ogg'} else None,
                    'video_policy':'failures'}}
            cases.append(case);success=seed==1001 or method in {'H20','ogg'}
            result={'success':success,'env_steps':4,'chunk_calls':1,'policy_s':.2,'monitor_s':.1 if method in {'lvm','ogg'} else 0,
                'event_replans':int(method in {'lvm','ogg'}),'guidance_calls':int(method=='ogg')}
            record={'case':case,'status':'completed','attempt':1,'wall_s':1.,'result':result,'video_artifacts':[]}
            attempt=evaluation/'attempts'/case['id']/'attempt_001';attempt.mkdir(parents=True)
            if not success:
                video=attempt/'video.mp4';video.write_bytes(b'not a real video: test existence only')
                record['video_artifacts']=[{'path':str(video.relative_to(evaluation)),'sha256':bundle.sha(video)}]
            write(evaluation/'trials'/f"{case['id']}.json",record)
            write(attempt/'actual_protocol.json',{'monitor_horizon_steps':2,'monitor_first_due_step':2,
                'predictor_protocol':{'h_window':1,'prediction_seconds':.1,'records':['large source details excluded']}})
            if method in {'lvm','ogg'}:
                trace=[{'step':2,'score':.3,'trigger':False},{'step':3,'score':.4,'trigger':True}]
                if method=='ogg':trace.append({'step':3,'guidance_loss':.5})
                (attempt/'monitor_steps.jsonl').write_text('\n'.join(json.dumps(r) for r in trace))
    write(evaluation/'plan.json',{'cases':cases,'protocol':{'debug_forced_event':False,'fps':20,
        'policy_revision':'synthetic-policy','corrector_settings':{'stride':1},
        'conditions':{'libero_object/task_0/init_0':{'task_id':0,'language':'synthetic task'}}}})
    write(evaluation/'execution.json',{'status':'completed','completed':8,'planned':8,'errors':0})


class ResultsAuditTests(unittest.TestCase):
    def export(self,root,archive):
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            return bundle.Exporter(z,{'night':root}).collect()

    def test_roundtrip_recomputes_success_pairs_and_excludes_weights_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';fixture(source)
            before=bundle.sha(source/'runs/mlp_s42/best_model.pt')
            manifest=self.export(source,root/'bundle.zip')
            self.assertFalse(any(n.endswith(('.pt','.mp4')) for n in manifest['members']))
            result=analysis.analyze(root/'bundle.zip',root/'analysis',plot=False)
            self.assertEqual(result['counts']['completed_training_runs'],1)
            self.assertEqual(result['counts']['formal_completed_trials'],8)
            ogg=next(r for r in result['closed_loop'] if r['method']=='ogg')
            self.assertEqual(ogg['successes'],2);self.assertEqual(ogg['completed'],2)
            paired=next(r for r in result['paired'] if r['baseline']=='lvm')
            self.assertEqual(paired['candidate_only_success'],1)
            self.assertEqual(result['monitor_coverage'][0]['coverage'],1.)
            self.assertFalse(any(f['severity']=='error' for f in result['findings']))
            self.assertEqual(bundle.sha(source/'runs/mlp_s42/best_model.pt'),before)

    def test_mse_reconstruction_matches_vector_identity(self):
        predicted=np.array([3.,4.]);target=np.array([0.,5.])
        cosine=float(predicted@target/(np.linalg.norm(predicted)*np.linalg.norm(target)))
        mse=analysis.residual_mse_from_norms(float(np.linalg.norm(predicted)),float(np.linalg.norm(target)),1-cosine,2)
        self.assertAlmostEqual(mse,float(np.mean((predicted-target)**2)))

    def test_warmup_excluded_and_19_step_gap_is_point_95_seconds(self):
        trace=[{'step':i} for i in list(range(24,51))+list(range(70,100))]
        result=analysis.coverage_stats(trace,24,100,1,20)
        self.assertEqual(result['eligible_score_steps'],76);self.assertEqual(result['scores'],57)
        self.assertEqual(result['max_gap_steps'],19);self.assertEqual(result['max_gap_seconds'],.95)

    def test_duplicate_and_early_scores_do_not_inflate_coverage(self):
        result=analysis.coverage_stats([{'step':1},{'step':2},{'step':2},{'step':3}],2,4,1,20)
        self.assertEqual(result['scores'],2);self.assertEqual(result['duplicate_scores'],1)
        self.assertEqual(result['out_of_schedule_scores'],1)
        self.assertEqual(analysis.coverage_stats([],24,20,1,20)['eligible_score_steps'],0)

    def test_saved_indices_corruption_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'source')
            np.savez(root/'source/runs/mlp_s42/split_indices.npz',train=np.array([0,1]),validation=np.array([1]))
            self.export(root/'source',root/'bundle.zip')
            result=analysis.analyze(root/'bundle.zip',root/'analysis',False)
            self.assertTrue(any(f['code']=='index_file_hash' for f in result['findings']))
            self.assertTrue(any(f['code']=='sample_leak' for f in result['findings']))

    def test_stream_zip_with_login_banner_can_be_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'source');self.export(root/'source',root/'bundle.zip')
            path=root/'banner.zip';path.write_bytes(b'SSH banner\n'+(root/'bundle.zip').read_bytes())
            self.assertEqual(len(bundle.validate_bundle(path)['runs']),1)

    def test_modified_payload_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'source');self.export(root/'source',root/'bundle.zip')
            with zipfile.ZipFile(root/'bundle.zip') as source,zipfile.ZipFile(root/'bad.zip','w') as target:
                for info in source.infolist():
                    raw=source.read(info.filename)
                    if info.filename.endswith('/result.json'):raw=b'{}'
                    target.writestr(info.filename,raw)
            with self.assertRaises(ValueError):bundle.validate_bundle(root/'bad.zip')


if __name__=='__main__':unittest.main()
