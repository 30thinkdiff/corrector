"""代理恢复只针对匹配的下载器；测试不发信号、不联网、不启动任务。"""
import importlib.util
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('recover_download',ROOT/'scripts/corrector_recover_download.py')
recover=importlib.util.module_from_spec(spec);spec.loader.exec_module(recover)


class RecoveryTests(unittest.TestCase):
    def command(self,root):
        return ['python','-u',str(root/'training-code-netfix/corrector_prefetch.py'),
                '--manifest',str(root/'object10/manifest.json'),
                '--local-dir',str(root/'object10-work/source'),'--max-new-files','0',
                '--proxy-url','http://127.0.0.1:17890','--workers','3']

    def prepare(self,root):
        for name in ['runtime/venv/bin/python','training-code-netfix/corrector_prefetch.py',
                     'object10/manifest.json']:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')

    def test_only_expected_full_download_matches(self):
        root=Path('/fixture')
        self.assertTrue(recover.is_target(self.command(root),root))
        consumer=self.command(root)
        consumer[2]=str(root/'local-pipeline/corrector_stream_workbench.py')
        self.assertFalse(recover.is_target(consumer,root))
        self.assertFalse(recover.is_target(self.command(Path('/other')),root))

    def test_changed_pid_does_not_receive_signal(self):
        root=Path('/fixture')
        with patch.object(recover,'targets',return_value={}),patch.object(recover.os,'kill') as kill:
            with self.assertRaises(RuntimeError):recover.stop_target(111,self.command(root),root)
            kill.assert_not_called()

    def test_graceful_exit_needs_no_force(self):
        root=Path('/fixture');command=self.command(root)
        with patch.object(recover,'targets',side_effect=[{111:command},{}]), \
             patch.object(recover.os,'kill') as kill:
            self.assertFalse(recover.stop_target(111,command,root))
            kill.assert_called_once_with(111,signal.SIGTERM)

    def test_force_after_grace_still_checks_same_command(self):
        root=Path('/fixture');command=self.command(root)
        with patch.object(recover,'targets',side_effect=[{111:command},{111:command},{}]), \
             patch.object(recover.time,'monotonic',side_effect=[0.,16.]), \
             patch.object(recover.os,'kill') as kill:
            self.assertTrue(recover.stop_target(111,command,root))
            self.assertEqual([c.args for c in kill.call_args_list],[(111,signal.SIGTERM),(111,signal.SIGKILL)])

    def test_failed_new_proxy_keeps_current_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.prepare(root)
            with patch.object(recover.subprocess,'run',return_value=SimpleNamespace(returncode=28,stdout='000')), \
                 patch.object(recover.os,'kill') as kill,patch.object(recover.subprocess,'Popen') as launch, \
                 patch('sys.argv',['recover','--root',str(root)]):
                with self.assertRaises(SystemExit):recover.main()
                kill.assert_not_called();launch.assert_not_called()

    def test_restart_reuses_paths_changes_only_proxy_and_leaves_gpu_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.prepare(root);command=self.command(root)
            with patch.object(recover.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='302')), \
                 patch.object(recover,'targets',side_effect=[{111:command},{}]), \
                 patch.object(recover,'stop_target',return_value=False) as stop, \
                 patch.object(recover.subprocess,'Popen',return_value=SimpleNamespace(pid=222)) as launch, \
                 patch('sys.argv',['recover','--root',str(root)]):
                recover.main()
                stop.assert_called_once_with(111,command,root)
                actual=launch.call_args.args[0]
                self.assertEqual(recover.option(actual,'--proxy-url'),'http://127.0.0.1:17891')
                self.assertEqual(recover.option(actual,'--local-dir'),str(root/'object10-work/source'))
                self.assertTrue(launch.call_args.kwargs['start_new_session'])


if __name__=='__main__':unittest.main()
