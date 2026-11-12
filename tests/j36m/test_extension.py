"""Safety contracts for the independent follow-up; no real GPU or remote calls."""
import json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import corrector_j36m_extension as e

class Followup(unittest.TestCase):
 def fixture(self,root,status='paused'):
  primary=root/'primary';primary.mkdir();code=root/'code';code.mkdir()
  (code/'DEPLOYMENT_MANIFEST.json').write_text('{}')
  e.write(primary/'plan.json',{'deadline_epoch':time.time()+3600,'budget_hours':32})
  e.write(primary/'status.json',{'status':status})
  argv=['followup','--primary',str(primary),'--root',str(root/'followup'),'--data-root',str(root/'data'),'--code',str(code)]
  return primary,code,argv
 def test_primary_pause_never_spawns_GPU(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);primary,code,argv=self.fixture(root)
   with patch.object(sys,'argv',argv),patch.object(e.subprocess,'Popen') as popen:e.main()
   popen.assert_not_called();self.assertEqual(e.read(root/'followup/status.json')['status'],'blocked')
 def test_operator_STOP_cancels_followup(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);primary,code,argv=self.fixture(root,'running');(primary/'STOP').touch()
   with patch.object(sys,'argv',argv),patch.object(e.subprocess,'Popen') as popen:e.main()
   popen.assert_not_called();self.assertEqual(e.read(root/'followup/status.json')['status'],'paused')
 def test_changed_frozen_deadline_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);primary,code,argv=self.fixture(root)
   with patch.object(sys,'argv',argv):e.main()
   data=e.read(primary/'plan.json');data['deadline_epoch']+=1;e.write(primary/'plan.json',data)
   with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'plan changed'):e.main()
 def test_JC_gate_and_matched_budget_before_replication(self):
  jobs=e.proposed(Path('/code'),Path('/state'),Path('/data'),'/python',Path('/primary'))
  self.assertEqual(jobs[0]['name'],'gpu_JC');self.assertEqual(len(jobs),9)
  self.assertEqual({x['name'] for x in jobs[1:4]},{'JC_s42','JC_s43','JC_s44'})
  for j in jobs[1:]:
   c=j['command'];self.assertEqual(c[c.index('--epochs')+1],'60');self.assertEqual(c[c.index('--accumulate')+1],'4')
   self.assertIn('--data-min-free-gib',c)

if __name__=='__main__':unittest.main()
