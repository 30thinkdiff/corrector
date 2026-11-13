import json,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import corrector_j36m as cli
import corrector_j36m_pipeline as queue
import package_j36m as package

class Workflow(unittest.TestCase):
    def test_deployment_hash_no_overwrite_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'source.zip';payload=b'print("source fixture")\n'
            def fixture(name):
                with zipfile.ZipFile(archive,'w') as z:
                    z.writestr(name,payload)
                    z.writestr('DEPLOYMENT_MANIFEST.json',json.dumps({'format':'j36m_source_v1','git_commit':'fixture',
                        'files':{name:package.sha(payload)}}))
                return package.sha(archive.read_bytes())
            digest=fixture('scripts/tool.py')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):package.install(archive,'0'*64,root/'bad')
            self.assertFalse((root/'bad').exists())
            with patch('builtins.print'):package.install(archive,digest,root/'installed')
            self.assertEqual((root/'installed/scripts/tool.py').read_bytes(),payload)
            with self.assertRaises(FileExistsError):package.install(archive,digest,root/'installed')
            digest=fixture('../escaped.py')
            with self.assertRaisesRegex(ValueError,'unsafe member'):package.install(archive,digest,root/'unsafe')
            self.assertFalse((root/'unsafe').exists());self.assertFalse((root/'escaped.py').exists())

    def test_receipt_rejects_changed_evidence_and_missing_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args=SimpleNamespace(receipts=root/'receipts');e=root/'evidence.json';e.write_text('{"real":true}')
            cli.publish(args,'cpu',e);self.assertEqual(cli.receipt(args,'cpu')['status'],'passed')
            e.write_text('{"real":false}')
            with self.assertRaises(ValueError):cli.receipt(args,'cpu')
            with self.assertRaises(FileNotFoundError):cli.receipt(args,'gpu_J')
    def test_plan_freezes_balanced_config_and_default_stops_at_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            with patch.object(sys,'argv',['queue','plan','--root',str(root)]):
                with patch('builtins.print'):queue.main()
            index=json.loads((root/'experiment_index.json').read_text())
            self.assertEqual(index['jobs'][0]['mode'],'preflight')
            self.assertEqual(index['jobs'][3]['mode'],'station-audit')
            self.assertEqual(len([j for j in index['jobs'] if j['mode']=='gpu-smoke']),5)
            self.assertEqual(len(index['config_files']),3)
            # Failed preflight must stop immediately; no later GPU job is spawned.
            child=SimpleNamespace(pid=123456,poll=lambda:1,returncode=1)
            with patch.object(sys,'argv',['queue','run','--root',str(root)]),patch.object(queue.subprocess,'Popen',return_value=child) as popen:
                queue.main()
            self.assertEqual(popen.call_count,1)
            status=json.loads((root/'pipeline_status.json').read_text())
            self.assertEqual(status['status'],'paused_or_error');self.assertEqual(status['until'],'smoke')

if __name__=='__main__':unittest.main()
