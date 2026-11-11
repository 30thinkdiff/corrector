"""准备入口的CPU合同检查；不冒充真实SmolVLA GPU训练。"""
import ast
import importlib.util
import json
import inspect
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/smolvla_training_workbench.py"
spec = importlib.util.spec_from_file_location("training_workbench", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrainingWorkbenchTests(unittest.TestCase):
    def test_cli_backend_does_not_inherit_notebook_inline_and_parent_is_unchanged(self):
        with patch.dict(module.os.environ, {"MPLBACKEND": "module://matplotlib_inline.backend_inline"}):
            env = module.runtime_env(Path('/tmp/backend-contract'))
            self.assertEqual(env['MPLBACKEND'], 'Agg')
            self.assertEqual(module.os.environ['MPLBACKEND'], 'module://matplotlib_inline.backend_inline')

    def test_supported_torch_range_keeps_lerobot_constraint(self):
        self.assertTrue(module.torch_version_supported('2.11.0+cu130'))
        self.assertTrue(module.torch_version_supported('2.7.1'))
        self.assertFalse(module.torch_version_supported('2.12.0+cu130'))
        self.assertFalse(module.torch_version_supported('2.6.0'))

    def test_incompatible_host_dispatches_to_new_interpreter_before_linking(self):
        from unittest.mock import Mock
        fake = SimpleNamespace(__version__='2.14.0+cu130', __file__='/host/torch/__init__.py',
                               version=SimpleNamespace(cuda='13.0'),
                               cuda=SimpleNamespace(is_available=lambda: True))
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(workdir=Path(directory), profile='gpu', allow_managed_torch=True)
            with patch.dict(sys.modules, {'torch': fake}), patch.object(module, 'prepare_with_managed_torch') as managed, \
                    patch.object(module, 'ensure_environment') as environment, \
                    patch.object(module.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 2**30)):
                module.prepare(args, Mock())
                managed.assert_called_once()
                environment.assert_not_called()
            self.assertEqual(fake.__version__, '2.14.0+cu130')

    def test_managed_torch_installs_only_inside_isolated_environment_and_no_recursive_flag(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / 'torch-runtime-2.11-cu128/bin/python'
            args = SimpleNamespace(workdir=root, skip_assets=True)
            log = Mock(directory=root / 'logs')
            with patch.object(module, 'ensure_environment', return_value=python), \
                    patch.object(module.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='missing torch')), \
                    patch.object(module.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 2**30)):
                module.prepare_with_managed_torch(args, log)
            commands = [list(call.args[0]) for call in log.run.call_args_list]
            self.assertEqual(commands[0][0], python)
            self.assertIn(module.MANAGED_TORCH_INDEX, commands[0])
            self.assertIn('torch==2.11.0', commands[0])
            self.assertEqual(commands[-1][0], python)
            self.assertIn('--skip-assets', commands[-1])
            self.assertNotIn('--allow-managed-torch', commands[-1])

    def test_inspection_exports_latest_frame_for_five_dimensional_images(self):
        import torch
        from PIL import Image
        images = torch.zeros(1, 2, 3, 2, 2)
        images[:, -1] = 1
        original = images.clone()
        with tempfile.TemporaryDirectory() as directory:
            names = module.save_inspection_images({'observation.images.image': images,
                                                   'observation.images.image2': torch.zeros(1, 3, 2, 2),
                                                   'observation.images.image_is_pad': torch.zeros(1, 2, dtype=torch.bool)}, directory)
            self.assertEqual(len(names), 2)
            with Image.open(Path(directory) / 'observation_images_image.png') as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            with Image.open(Path(directory) / 'observation_images_image2.png') as image:
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
            torch.testing.assert_close(images, original)

    def make_dataset_fixture(self, root, *, wrong_file=False, wrong_task=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        (root / 'data/chunk-000').mkdir(parents=True)
        (root / 'meta/episodes/chunk-000').mkdir(parents=True)
        records = []
        for episode, length, start, physical_file in [(813, 2, 100, 213), (818, 3, 200, 214), (825, 1, 300, 215)]:
            records.append({'episode_index': episode, 'length': length, 'dataset_from_index': start,
                            'dataset_to_index': start + length, 'data/file_index': 37})
            rows = [{'episode_index': 90 if wrong_file else episode,
                     'task_index': 5 if wrong_task else 24,
                     'frame_index': i, 'index': start + i} for i in range(length)]
            pq.write_table(pa.Table.from_pylist(rows), root / f'data/chunk-000/file-{physical_file:03d}.parquet')
        pq.write_table(pa.Table.from_pylist(records), root / 'meta/episodes/chunk-000/file-000.parquet')

    def test_dataset_validation_uses_physical_episode_rows_not_metadata_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset_fixture(root)
            report = module.validate_dataset_files(root)
            self.assertEqual(report['813']['frames'], 2)
            self.assertEqual(report['818']['files'], ['data/chunk-000/file-214.parquet'])

    def test_dataset_missing_selected_episodes_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset_fixture(root, wrong_file=True)
            with self.assertRaisesRegex(RuntimeError, '813.*0帧'):
                module.validate_dataset_files(root)

    def test_dataset_wrong_task_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset_fixture(root, wrong_task=True)
            with self.assertRaisesRegex(RuntimeError, '任务'):
                module.validate_dataset_files(root)

    def test_config_loads_cached_sha_without_main_reference(self):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            repo = cache / 'models--lerobot--smolvla_base'
            snapshot = repo / 'snapshots' / module.BASE_SHA
            snapshot.mkdir(parents=True)
            config = snapshot / 'config.json'
            config.write_text('{"scope":"offline config regression"}')
            self.assertFalse((repo / 'refs/main').exists())

            def loader(model_id, **kwargs):
                return hf_hub_download(model_id, 'config.json', cache_dir=cache, **kwargs)

            # 真实Hub缓存解析：默认main找不到，即使SHA目录的文件已存在。
            with self.assertRaises(LocalEntryNotFoundError):
                loader(module.BASE_ID, local_files_only=True)
            resolved = module.load_policy_config(loader, module.BASE_ID)
            self.assertEqual(Path(resolved), config)
            self.assertFalse((repo / 'refs/main').exists())

    def test_config_revision_conflict_rejected_and_local_checkpoint_untouched(self):
        loader = __import__('unittest.mock', fromlist=['Mock']).Mock(return_value='config')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            module.load_policy_config(loader, module.BASE_ID, revision='a-different-version')
        loader.assert_not_called()
        self.assertEqual(module.load_policy_config(loader, Path('/tmp/checkpoint'), revision='other'), 'config')
        loader.assert_called_once_with(Path('/tmp/checkpoint'), revision='other')

    def test_repairs_python_without_pip_and_reuses_ready_environment(self):
        # 模拟Colab失败留下的状态：Python存在、pip不存在；无需网络。
        import ensurepip
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'venv'
            subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(target)], check=True)
            missing = subprocess.run([str(target / 'bin/python'), '-m', 'pip', '--version'], capture_output=True)
            self.assertNotEqual(missing.returncode, 0)
            env = module.runtime_env(root)
            env['PIP_NO_INDEX'] = '1'
            env['PIP_FIND_LINKS'] = str(Path(ensurepip.__file__).parent / '_bundled')
            log = module.Log(root / 'logs')
            python = module.ensure_environment(target, log, env)
            result = subprocess.run([str(python), '-m', 'pip', '--version'], text=True, capture_output=True, check=True)
            self.assertIn(str(target), result.stdout)
            self.assertEqual(module.ensure_environment(target, log, env), python)
            saved = log.path.read_text()
            bootstrap_count = saved.count('"install", "pip==25.0.1"') + saved.count('"ensurepip", "--upgrade", "--default-pip"')
            self.assertEqual(bootstrap_count, 1)

    def test_environment_that_points_outside_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bin').mkdir()
            (root / 'bin/python').symlink_to(sys.executable)
            with self.assertRaisesRegex(RuntimeError, '其他环境'):
                module.ensure_environment(root, module.Log(root / 'logs'), module.runtime_env(root))

    def test_drive_directory_without_actual_mount_is_rejected(self):
        with patch.object(module.os.path, 'ismount', return_value=False):
            with self.assertRaisesRegex(RuntimeError, '挂载'):
                module.check_persistent_path(Path('/content/drive/MyDrive/experiment'))
        module.check_persistent_path(Path('/tmp/server-local-experiment'))

    def make_update_auditor(self, directory):
        import torch
        tree = ast.parse(SCRIPT.read_text())
        train_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        auditor = next(n for n in train_node.body if isinstance(n, ast.FunctionDef) and n.name == "audited_update")
        evidence, holder = {}, {}

        def original_update(train_metrics, policy, batch, optimizer, accelerator):
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return train_metrics, {}

        namespace = {"torch": torch, "inspect": inspect, "patch": patch,
                     "math": __import__('math'), "BASE_SHA": module.BASE_SHA,
                     "shutil": __import__('shutil'),
                     "args": SimpleNamespace(inspect_only=False), "BatchInspectionComplete": module.BatchInspectionComplete,
                     "practice_step": None, "PracticeStepComplete": module.PracticeStepComplete,
                     "original_update": original_update, "evidence": evidence,
                     "holder": holder, "output": Path(directory), "write_json": module.write_json,
                     "log": SimpleNamespace(say=lambda value: None)}
        evidence['steps'] = []
        exec(compile(ast.Module(body=[auditor], type_ignores=[]), str(SCRIPT), 'exec'), namespace)
        return namespace['audited_update'], evidence, namespace

    def test_auditor_observes_before_zero_grad_and_keeps_real_update(self):
        import torch

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
                self.adapter = torch.nn.Parameter(torch.tensor(0.))
                self.peft_config = {'default': SimpleNamespace(revision=None)}

            def forward(self, batch):
                return (((self.frozen + self.adapter) * batch['x'] - batch['target']) ** 2).mean(), {}

        with tempfile.TemporaryDirectory() as directory:
            auditor, evidence, _ = self.make_update_auditor(directory)
            policy = Layer()
            optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
            accelerator = SimpleNamespace(num_processes=1, mixed_precision='no', unwrap_model=lambda x: x)
            with patch.object(torch.cuda, 'max_memory_allocated', return_value=0), patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
                auditor(None, policy, {'x': torch.ones(2), 'target': torch.ones(2)}, optimizer, accelerator)
            self.assertEqual(policy.frozen.item(), 2.)
            self.assertIsNone(policy.adapter.grad)
            self.assertTrue(evidence['steps'][0]['any_nonzero_gradient'])
            self.assertTrue(evidence['steps'][0]['any_trainable_parameter_changed'])
            self.assertEqual(evidence['steps'][0]['loss'], 1.)
            self.assertEqual(policy.peft_config['default'].revision, module.BASE_SHA)

    def test_auditor_rejects_nan_loss_before_optimizer_changes(self):
        import torch

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = torch.nn.Parameter(torch.tensor(1.))
                self.peft_config = {'default': SimpleNamespace(revision=None)}

            def forward(self, batch):
                return self.adapter * torch.tensor(float('nan')), {}

        with tempfile.TemporaryDirectory() as directory:
            auditor, evidence, _ = self.make_update_auditor(directory)
            policy = Layer()
            optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
            accelerator = SimpleNamespace(num_processes=1, mixed_precision='no', unwrap_model=lambda x: x)
            with self.assertRaisesRegex(RuntimeError, 'loss'):
                auditor(None, policy, {'x': torch.ones(1)}, optimizer, accelerator)
            self.assertEqual(policy.adapter.item(), 1.)

    def test_opt_in_practice_calls_user_function_once_and_saves_separately(self):
        import torch
        from studies.policy_generalization.train_one_batch import train_one_batch

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = torch.nn.Parameter(torch.tensor(0.))
                self.peft_config = {'default': SimpleNamespace(revision=None)}

            def forward(self, batch):
                return ((self.adapter - batch['target']) ** 2).mean(), {}

            def save_pretrained(self, path):
                path.mkdir()
                torch.save(self.state_dict(), path / 'test_weights.pt')

        with tempfile.TemporaryDirectory() as directory:
            auditor, evidence, namespace = self.make_update_auditor(directory)
            policy = Layer()
            policy.eval()
            policy.adapter.grad = torch.tensor(999.)
            optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
            calls = []

            def user_step(model, batch, opt):
                calls.append(model)
                return train_one_batch(model, batch, opt)

            namespace['practice_step'] = user_step
            namespace['args'].practice_step_file = Path(__file__)
            accelerator = SimpleNamespace(num_processes=1, mixed_precision='no', unwrap_model=lambda x: x)
            with patch.object(torch.cuda, 'max_memory_allocated', return_value=0), patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
                with self.assertRaises(module.PracticeStepComplete):
                    auditor(None, policy, {'target': torch.ones(1)}, optimizer, accelerator)
            self.assertEqual(len(calls), 1)
            self.assertTrue(policy.training)
            self.assertAlmostEqual(policy.adapter.item(), .2, places=6)
            self.assertTrue(evidence['steps'][0]['any_trainable_parameter_changed'])
            self.assertEqual(evidence['steps'][0]['optimizer_step_calls'], 1)
            self.assertTrue((Path(directory) / 'optimizer.pt').is_file())
            self.assertFalse((Path(directory) / 'checkpoints').exists())

    def test_archive_complete_idempotent_and_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "archive"
            (source / "pretrained_model").mkdir(parents=True)
            (source / "training_state").mkdir()
            (source / "pretrained_model/adapter.bin").write_bytes(b"adapter")
            (source / "training_state/optimizer.bin").write_bytes(b"optimizer")
            module.archive_checkpoint(source, destination)
            module.archive_checkpoint(source, destination)
            self.assertTrue((destination / "COMPLETE.json").is_file())
            (destination / "training_state/optimizer.bin").write_bytes(b"different")
            with self.assertRaises(RuntimeError):
                module.archive_checkpoint(source, destination)
            self.assertEqual((source / "training_state/optimizer.bin").read_bytes(), b"optimizer")

    def test_partial_archive_with_foreign_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "pretrained_model").mkdir(parents=True)
            (source / "training_state").mkdir()
            (source / "pretrained_model/weights").write_bytes(b"weights")
            partial = root / "target.partial"
            partial.mkdir()
            (partial / "old-run").write_text("preserve")
            with self.assertRaises(RuntimeError):
                module.archive_checkpoint(source, root / "target")
            self.assertTrue((partial / "old-run").exists())

    def test_single_cell_contains_current_standalone_script(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepare.ipynb"
            module.export_notebook(SimpleNamespace(notebook=path))
            notebook = json.loads(path.read_text())
            self.assertEqual(len(notebook["cells"]), 1)
            cell = notebook["cells"][0]
            tree = ast.parse("".join(cell["source"]))
            source = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "SOURCE" for t in n.targets))
            self.assertEqual(source, SCRIPT.read_text())
            self.assertFalse(cell["outputs"])

    def test_notebook_export_preserves_user_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(notebook=Path(directory) / 'prepare.ipynb')
            module.export_notebook(args)
            original = json.loads(args.notebook.read_text())
            user = {'cell_type': 'code', 'id': 'user-gpu-check', 'metadata': {},
                    'source': ['print("my GPU")'], 'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': ['Tesla T4']}],
                    'execution_count': 2}
            original['cells'].append(user)
            args.notebook.write_text(json.dumps(original))
            module.export_notebook(args)
            self.assertEqual(json.loads(args.notebook.read_text())['cells'][1], user)

    def test_failure_log_keeps_context_and_redacts_common_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            log = module.Log(directory)
            token = "hf_" + "a" * 30
            with self.assertRaises(RuntimeError):
                log.run([sys.executable, "-c", f"print('{token}');raise SystemExit(7)"])
            saved = log.path.read_text()
            self.assertNotIn(token, saved)
            self.assertIn("TOKEN_REDACTED", saved)
            self.assertIn("SystemExit(7)", saved)


if __name__ == "__main__":
    unittest.main()
