"""远程Git接入的本地临时仓库测试；不连接GitHub或GPU。"""
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
import json
import io
from contextlib import redirect_stderr
import corrector_remote_git_setup as setup
import corrector_repository_export as export


def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.STDOUT)


class RepositorySetupTests(unittest.TestCase):
    def source(self,root):
        source=root/'source';source.mkdir();git(source,'init','-b','main')
        (source/'runtime').mkdir();(source/'runtime/code.py').write_text('# published source\n')
        (source/'README.md').write_text('Chinese project description\n');(source/'.gitignore').write_text('data/\n*.pt\n')
        git(source,'add','.')
        git(source,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','test snapshot')
        sha=git(source,'rev-parse','HEAD').decode().strip();bundle=root/'source.bundle';git(source,'bundle','create',str(bundle),'main')
        return source,bundle,sha

    def test_initialize_preserves_working_code_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,bundle,sha=self.source(root);target=root/'remote';target.mkdir()
            (target/'runtime').mkdir();(target/'runtime/code.py').write_text('# newer remote edit\n')
            (target/'data').mkdir();(target/'data/model.pt').write_bytes(b'valuable checkpoint')
            result=setup.initialize(target,bundle,sha)
            self.assertEqual(result['head'],sha);self.assertEqual((target/'runtime/code.py').read_text(),'# newer remote edit\n')
            self.assertEqual((target/'data/model.pt').read_bytes(),b'valuable checkpoint')
            self.assertTrue((target/'README.md').is_file());self.assertIn('M runtime/code.py',result['status'])
            self.assertNotIn('data/',result['status'])
            self.assertEqual(git(target,'config','user.email').decode().strip(),setup.PUBLIC_EMAIL)
            self.assertEqual(setup.initialize(target,bundle,sha)['head'],sha)

    def test_existing_different_history_is_not_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,bundle,sha=self.source(root)
            (source/'README.md').write_text('newer local history\n');git(source,'add','README.md')
            git(source,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','newer')
            head=git(source,'rev-parse','HEAD')
            with self.assertRaises(RuntimeError):setup.initialize(source,bundle,sha)
            self.assertEqual(git(source,'rev-parse','HEAD'),head)

    def test_export_omits_credentials_data_and_environments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'remote';base.mkdir();(base/'runtime').mkdir()
            (base/'runtime/code.py').write_text('# owned script')
            (base/'.env').write_text('SECRET=not_for_export')
            (base/'runtime/model.pt').write_bytes(b'model')
            (base/'runtime/venv').mkdir();(base/'runtime/venv/private.py').write_text('not owned source')
            (base/'night-code-demo').mkdir();(base/'night-code-demo/train.py').write_text('# frozen script')
            path=root/'snapshot.zip'
            with redirect_stderr(io.StringIO()):export.export(path,base)
            with zipfile.ZipFile(path) as z:
                names=set(z.namelist());self.assertIn('runtime/code.py',names);self.assertIn('night-code-demo/train.py',names)
                self.assertNotIn('.env',names);self.assertNotIn('runtime/model.pt',names);self.assertNotIn('runtime/venv/private.py',names)


if __name__=='__main__':unittest.main()
