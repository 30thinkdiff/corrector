import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_final_review import seed_review,review


def fixtures():
    def row(m,s):return {'case':{'condition':'condition1','method':m,'seed':s,'id':m+str(s)},'status':'completed','attempt':1,'result':{'success':m=='H20'}}
    first=[row(m,1000) for m in ['H20','j_ogg']]
    repeat=[row(m,s) for s in [1001,1002] for m in ['H20','j_ogg']]
    selected=[{'condition':'condition1','seed':1000,'outcomes':{'H20':True,'j_ogg':False}}]
    return first,repeat,selected


class SeedTests(unittest.TestCase):
    def test_repetitions_are_one_condition_with_three_eval_seeds(self):
        first,repeat,selected=fixtures();d=seed_review(first,repeat,selected)
        self.assertEqual(len(d['cases']),2)
        self.assertTrue(all(r['conditions']==1 and r['new_seed_trials']==2 for r in d['summary']))
        self.assertEqual(d['replay_candidates'][0]['a_only_success_seeds'],[1000,1001,1002])
    def test_missing_or_running_outcomes_never_become_failure(self):
        first,repeat,selected=fixtures()
        with self.assertRaisesRegex(ValueError,'incomplete'):seed_review(first,repeat[:-1],selected)
        repeat[-1]['status']='running'
        with self.assertRaisesRegex(ValueError,'completed'):seed_review(first,repeat,selected)
    def test_duplicate_or_changed_selection_is_rejected(self):
        first,repeat,selected=fixtures()
        with self.assertRaisesRegex(ValueError,'duplicate'):seed_review(first+[first[0]],repeat,selected)
        selected[0]['outcomes']['H20']=False
        with self.assertRaisesRegex(ValueError,'selection differs'):seed_review(first,repeat,selected)


@unittest.skipUnless((ROOT/'outputs/corrector_progress_snapshots/20261113T161432Z.zip').exists(),'requires verified real final snapshot')
class RealFinalTests(unittest.TestCase):
    def test_real_final_counts_and_case_pairing(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'review'
            d=review(ROOT/'outputs/corrector_progress_snapshots/20261113T161432Z.zip',
                     ROOT/'studies/policy_generalization/results/20261114/expanded_snapshot_20261113T161434Z',
                     ROOT/'outputs/corrector_progress_snapshots/final_integrity_20261114.json',out)
            self.assertEqual(sum(d['counts'].values()),957)
            self.assertEqual(sum(r['matched_j_pairs'] for r in d['j_ogg_pairs']),108)
            self.assertTrue(all(r['different_j_lvm_ogg_outcomes']==0 for r in d['j_ogg_pairs']))
            rows={r['method']:r for r in d['seed_review']}
            self.assertEqual(rows['mlp_ogg']['new_seed_successes'],13)
            self.assertEqual(rows['j_ogg']['new_seed_successes'],4)
            self.assertEqual(rows['H20']['conditions'],8)
            self.assertTrue((out/'replay_candidates.json').exists())
