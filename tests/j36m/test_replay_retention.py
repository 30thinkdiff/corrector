import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from replay_retention import select_groups


def record(c,s,m,out,status='completed'):
    return {'case':{'condition':c,'seed':s,'method':m},'status':status,'result':{'success':out}}


class SelectionTests(unittest.TestCase):
    def test_missing_and_error_never_become_failure(self):
        rows=[record('c',1,'H50',True),record('c',1,'H20',None),record('d',1,'H20',False,'error')]
        self.assertEqual(select_groups(rows,['H50','H20'])['eligible_groups'],0)
    def test_distinct_physical_states_and_bounded_diagnostic_selection(self):
        rows=[record(c,s,m,m=='H20') for c in ['a','b','c'] for s in [1,2] for m in ['H50','H20']]
        d=select_groups(rows,['H50','H20'],max_groups=8,per_pattern=2)
        self.assertEqual(len(d['selected_groups']),2)
        self.assertEqual(len({r['condition'] for r in d['selected_groups']}),2)
        self.assertTrue(all(r['pair_flips']==[['H50','H20']] for r in d['selected_groups']))
        self.assertFalse(d['deletion_performed'])
    def test_duplicate_outcome_does_not_silently_overwrite(self):
        r=record('c',1,'H50',True)
        with self.assertRaises(ValueError):select_groups([r,r],['H50'])
