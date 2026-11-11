"""预下载限量、目录边界和无模型调用检查；不访问网络。"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[3] / 'scripts/corrector_prefetch.py'
spec = importlib.util.spec_from_file_location('corrector_prefetch', SCRIPT)
prefetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefetch)


class PrefetchTests(unittest.TestCase):
    def manifest(self, names):
        return {'dataset': 'HuggingFaceVLA/libero', 'revision': 'a' * 40,
                'records': [{'files': [{'path': name, 'bytes': 4} for name in names]}]}

    def test_limit_counts_new_files_and_deduplicates_shared_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data').mkdir()
            (root / 'data/a.parquet').write_bytes(b'abcd')
            m = self.manifest(['data/a.parquet', 'data/b.parquet',
                               'data/a.parquet', 'data/c.parquet'])
            files = prefetch.selected_files(m, root, 1)
            self.assertEqual(files, [('data/a.parquet', 4, True),
                                     ('data/b.parquet', 4, False)])

    def test_rejects_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'
            root.mkdir()
            with self.assertRaises(ValueError):
                prefetch.selected_files(self.manifest(['data/../../a.parquet']), root, 3)
            (root / 'data').symlink_to(Path(tmp), target_is_directory=True)
            with self.assertRaises(ValueError):
                prefetch.selected_files(self.manifest(['data/a.parquet']), root, 3)

    def run_cli(self, folder, *extra, env_overrides=None):
        folder = Path(folder)
        manifest = folder / 'manifest.json'
        manifest.write_text(json.dumps(self.manifest([
            'data/a.parquet', 'data/b.parquet', 'data/c.parquet'])))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(folder)
        env.update(env_overrides or {})
        return subprocess.run([sys.executable, str(SCRIPT), '--manifest', str(manifest),
                               '--local-dir', str(folder / 'source'), '--min-free-gib', '0',
                               *extra], env=env, text=True, capture_output=True, timeout=10)

    def test_cli_downloads_only_requested_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'huggingface_hub.py').write_text(
                'from pathlib import Path\n'
                'def hf_hub_download(repo, name, **kw):\n'
                '    assert kw["token"] is False\n'
                '    p=Path(kw["local_dir"])/name\n'
                '    p.parent.mkdir(parents=True,exist_ok=True)\n'
                '    p.write_bytes(b"abcd")\n'
                '    return str(p)\n')
            result = self.run_cli(folder, '--max-new-files', '2', '--workers', '2')
            self.assertEqual(result.returncode, 0, result.stderr)
            final = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(final['status'], 'completed')
            self.assertEqual(final['completed_files'], 2)
            self.assertFalse((folder / 'source/data/c.parquet').exists())

    def test_explicit_auth_uses_sdk_credentials_without_printing_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'huggingface_hub.py').write_text(
                'from pathlib import Path\n'
                'def get_token(): return "synthetic-secret-never-print"\n'
                'def hf_hub_download(repo, name, **kw):\n'
                '    assert kw["token"] is True\n'
                '    p=Path(kw["local_dir"])/name\n'
                '    p.parent.mkdir(parents=True,exist_ok=True)\n'
                '    p.write_bytes(b"abcd")\n'
                '    return str(p)\n')
            result = self.run_cli(folder, '--use-hf-token', '--max-new-files', '1',
                                  env_overrides={'HF_ENDPOINT': 'https://huggingface.co'})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('synthetic-secret-never-print', result.stdout + result.stderr)

    def test_auth_without_credentials_fails_before_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'huggingface_hub.py').write_text(
                'def get_token(): return None\n'
                'def hf_hub_download(*a, **kw): raise RuntimeError("must not download")\n')
            result = self.run_cli(folder, '--use-hf-token',
                                  env_overrides={'HF_ENDPOINT': 'https://huggingface.co'})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('未找到HF_TOKEN', result.stderr)
            self.assertNotIn('must not download', result.stderr)

    def test_mirror_cannot_use_official_account_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'huggingface_hub.py').write_text(
                'def get_token(): raise RuntimeError("must not read credentials")\n'
                'def hf_hub_download(*a, **kw): raise RuntimeError("must not download")\n')
            result = self.run_cli(folder, '--use-hf-token',
                                  env_overrides={'HF_ENDPOINT': 'https://hf-mirror.com'})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('仅用于https://huggingface.co', result.stderr)
            self.assertNotIn('must not read credentials', result.stderr)

    def test_stop_and_space_guard_precede_sdk_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'huggingface_hub.py').write_text('raise RuntimeError("must not import")\n')
            (folder / 'PREFETCH_STOP').touch()
            result = self.run_cli(folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout.splitlines()[-1])['status'], 'paused')
            (folder / 'PREFETCH_STOP').unlink()
            result = self.run_cli(folder, '--min-free-gib', '999999999')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('磁盘不足', result.stderr)
            self.assertNotIn('must not import', result.stderr)


if __name__ == '__main__':
    unittest.main()
