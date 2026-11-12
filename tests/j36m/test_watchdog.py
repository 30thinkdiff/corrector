import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from corrector_j36m_watchdog import may_restart
class Watchdog(unittest.TestCase):
 def test_dead_supervisor_can_resume_child_from_frozen_state(self):self.assertTrue(may_restart({'status':'running'},False,False,0))
 def test_never_duplicate_live_supervisor(self):self.assertFalse(may_restart({'status':'running'},True,False,0))
 def test_never_retry_program_errors_or_manual_pause(self):
  for status in ('error','paused','blocked','completed'):self.assertFalse(may_restart({'status':status},False,False,0))
  self.assertFalse(may_restart({'status':'running'},False,True,0))
 def test_no_unlimited_restart(self):self.assertFalse(may_restart({'status':'running'},False,False,2))
if __name__=='__main__':unittest.main()
