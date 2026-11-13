"""Small synthetic caches only. Never presented as real 4090 validation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from corrector_data_workbench import digest
import corrector_multilayer_train as train
from multilayer_mvp.data import (CAMERAS, IMAGES, POLICY, WEIGHT, MultiLayerTransitionDataset,
                                fit_scales, load_config, validate)
from multilayer_mvp.metrics import Metrics, loss
from multilayer_mvp.model import ModelConfig, MultiLayerCorrector
from multilayer_mvp.runtime import ResourcePause, safe_output, resources
import package_multilayer_mvp as packaging


def dump(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    cfg = {'features_root': str(root/'features'), 'policy_revision': POLICY,
           'policy_weight_sha256': WEIGHT, 'raw_shape': [2, 4, 8], 'connector_shape': [4, 12],
           'fps': 10., 'k_step': 10, 'episodes': []}
    rng = np.random.default_rng(51)
    for ep, split in [(813, 'train'), (814, 'train'), (815, 'validation')]:
        n = 14
        raw, old = root/f'raw_{ep}', root/'features'/f'episode_{ep}'
        raw.mkdir(parents=True)
        old.mkdir(parents=True)
        cfg['episodes'].append({'episode': ep, 'split': split, 'frames': n, 'task_id': ep-800,
                                'raw_path': str(raw)})
        values = {
            'z_q': rng.integers(-50, 50, (n, 4, 12), dtype='int8'),
            'z_scale': np.full((n, 4), .02, dtype='float16'),
            'actions': rng.normal(size=(n, 7)).astype('float16'),
            'frame_index': np.arange(n), 'episode_index': np.full(n, ep),
            'timestamp': np.arange(n, dtype='float32') / 10.,
        }
        for key, value in values.items():
            np.save(old/f'{key}.npy', value)
        meta = {'format': 'siglip_frame_cache_v2', 'num_frames': n, 'fps': 10., 'split': split,
                'camera_keys': CAMERAS, 'image_keys': IMAGES, 'policy_id': 'lerobot/smolvla_libero',
                'policy_revision': POLICY, 'action_representation': 'checkpoint_processor_normalized',
                'l_tokens': 4, 'd_dim': 12, 'action_dim': 7, 'dtype_actions': 'float16',
                'dtype_z_q': 'int8', 'dtype_z_scale': 'float16', 'raw_provenance': {
                    'episode': ep, 'split': split, 'num_frames': n, 'fps': 10., 'task_id': ep-800,
                    'sources': [{'sha256': 'a'*64}]}}
        dump(old/'metadata.json', meta)
        dump(old/'COMPLETE.json', {'files': {p.name: digest(p) for p in old.iterdir()}})
        for name in ('block_02', 'block_08'):
            np.save(raw/f'{name}.fp16.npy', rng.normal(size=(n, 2, 4, 8)).astype('float16'))
        marker = {'format': 'multilayer_raw_fp16_v1', 'episode': ep, 'num_frames': n,
                  'fps': 10., 'layer_indices_zero_based': [2, 8], 'camera_keys': CAMERAS,
                  'source_image_keys': IMAGES, 'dtype': 'float16', 'pooling': 'none',
                  'shape_per_layer': [n, 2, 4, 8], 'policy_revision': POLICY,
                  'policy_weight_sha256': WEIGHT, 'k_data_frames': 10,
                  'deep_cache_path': str(old), 'deep_cache_complete_sha256': digest(old/'COMPLETE.json'),
                  'frame_index_and_timestamp_source': 'deep_cache_path',
                  'action_source': 'deep_cache_path/actions.npy; checkpoint_processor_normalized',
                  'source_parquet_sha256': 'a'*64,
                  'files': {p.name: {'sha256': digest(p), 'size_bytes': p.stat().st_size} for p in raw.iterdir()}}
        dump(raw/'COMPLETE.json', marker)
    return cfg


class MVPTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = fixture(self.root)
        # Tiny temporary fixtures need <1MiB; retain the real resource implementation,
        # but do not require the production 2GiB reserve on this full local laptop.
        real_resources = train.resources
        self.guard = patch.object(train, 'resources', side_effect=lambda *a, **kw:
                                  real_resources(*a, **dict(kw, reserve_gib=.001)))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def records(self):
        return validate(self.cfg)

    def republish_deep(self, ep=813):
        old = Path(self.cfg['features_root']) / f'episode_{ep}'
        dump(old/'COMPLETE.json', {'files': {p.name: digest(p) for p in old.iterdir() if p.name != 'COMPLETE.json'}})
        raw = Path(next(r['raw_path'] for r in self.cfg['episodes'] if r['episode'] == ep))
        marker = json.loads((raw/'COMPLETE.json').read_text())
        marker['deep_cache_complete_sha256'] = digest(old/'COMPLETE.json')
        dump(raw/'COMPLETE.json', marker)

    def args(self, output, **kwargs):
        args = SimpleNamespace(output=output, device='cpu', precision='fp32', seed=42,
                               cpu_threads=1, variant='joint', batch_size=2, epochs=2,
                               patience=0, lr=.003, weight_decay=.0001, width=8, depth=1,
                               resume=False, overfit_steps=40)
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args

    def test_mmap_counts_boundaries_and_dequantization(self):
        records = self.records()
        ds = MultiLayerTransitionDataset(records, 'train')
        self.assertEqual(len(ds), 8)
        self.assertEqual(ds[3]['episode'], 813)
        self.assertEqual(ds[4]['episode'], 814)
        self.assertEqual(ds[4]['t'], 0)
        self.assertIsInstance(records[0]['arrays']['block_02'], np.memmap)
        self.assertFalse(records[0]['arrays']['block_02'].flags.writeable)
        x = ds[0]
        a = records[0]['arrays']
        np.testing.assert_array_equal(x['inputs']['connector'], a['z_q'][0].astype('float32') * a['z_scale'][0].astype('float32')[:, None])
        np.testing.assert_array_equal(x['targets']['delta_block_02'], a['block_02'][10].astype('float32') - a['block_02'][0].astype('float32'))
        self.assertEqual(x['inputs']['action'].shape, (7,))
        with self.assertRaises(ValueError):
            ds[len(ds)]

    def test_no_future_feature_or_action_in_inputs(self):
        records = self.records()
        ds = MultiLayerTransitionDataset(records, 'train')
        before = ds[0]
        raw = Path(self.cfg['episodes'][0]['raw_path'])
        for name in ('block_02', 'block_08'):
            a = np.load(raw/f'{name}.fp16.npy', mmap_mode='r+')
            a[10] += 50
            a.flush()
        a = np.load(Path(self.cfg['features_root'])/'episode_813/actions.npy', mmap_mode='r+')
        a[1:] += 10
        a.flush()
        after = ds[0]
        for k in before['inputs']:
            np.testing.assert_array_equal(before['inputs'][k], after['inputs'][k])
        self.assertFalse(np.array_equal(before['targets']['delta_block_02'], after['targets']['delta_block_02']))

    def test_split_overlap_rejected(self):
        self.cfg['episodes'].append(dict(self.cfg['episodes'][0], split='validation'))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.records()

    def test_validation_only_scale_fitting_rejected(self):
        ds = MultiLayerTransitionDataset(self.records(), 'validation')
        with self.assertRaisesRegex(ValueError, 'train only'):
            fit_scales(ds)

    def test_validation_mutation_does_not_affect_train_scales(self):
        records = self.records()
        ds = MultiLayerTransitionDataset(records, 'train')
        a = fit_scales(ds)
        raw = Path(self.cfg['episodes'][2]['raw_path'])
        data = np.load(raw/'block_02.fp16.npy', mmap_mode='r+')
        data[:] *= 10
        data.flush()
        self.assertEqual(a, fit_scales(ds))

    def test_sha_corruption_rejected(self):
        path = Path(self.cfg['episodes'][0]['raw_path'])/'block_02.fp16.npy'
        with path.open('r+b') as f:
            f.seek(-1, 2)
            f.write(b'X')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.records()

    def test_missing_complete_rejected(self):
        (Path(self.cfg['episodes'][0]['raw_path'])/'COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.records()

    def test_camera_revision_pooling_and_shape_mismatch(self):
        path = Path(self.cfg['episodes'][0]['raw_path'])/'COMPLETE.json'
        original = json.loads(path.read_text())
        for key, value in [('camera_keys', CAMERAS[::-1]), ('policy_revision', 'wrong'),
                           ('pooling', 'mean8'), ('shape_per_layer', [14, 2, 1, 8])]:
            dump(path, dict(original, **{key: value}))
            with self.assertRaises(ValueError):
                self.records()
        dump(path, original)

    def test_frame_time_and_episode_mismatch_even_with_valid_sha(self):
        old = Path(self.cfg['features_root'])/'episode_813'
        for name in ('frame_index', 'timestamp', 'episode_index'):
            path = old/f'{name}.npy'
            original = np.load(path).copy()
            bad = original.copy()
            bad[2] += 1
            np.save(path, bad)
            self.republish_deep()
            with self.assertRaises(ValueError):
                self.records()
            np.save(path, original)
            self.republish_deep()

    def test_normalized_actions_not_replaced_by_raw_actions(self):
        old = Path(self.cfg['features_root'])/'episode_813'
        np.save(old/'raw_actions.npy', np.ones((14, 7), dtype='float16')*100)
        self.republish_deep()
        ds = MultiLayerTransitionDataset(self.records(), 'train')
        np.testing.assert_array_equal(ds[0]['inputs']['action'], np.load(old/'actions.npy')[0].astype('float32'))

    def test_production_config_counts_and_strict_shapes(self):
        path = ROOT/'studies/policy_generalization/corrector/multilayer_mvp/config.json'
        cfg = load_config(path)
        self.assertEqual(sum(s['frames']-10 for s in cfg['episodes'] if s['split']=='train'), 557)
        self.assertEqual(sum(s['frames']-10 for s in cfg['episodes'] if s['split']=='validation'), 142)
        cfg['raw_shape'] = [2, 64, 768]
        dump(self.root/'bad.json', cfg)
        with self.assertRaises(ValueError):
            load_config(self.root/'bad.json')

    def test_full_real_shapes_shared_gradients_and_input_dependency(self):
        model = MultiLayerCorrector(ModelConfig(width=8, depth=1))
        inputs = {'block_02': torch.randn(1, 2, 1024, 768),
                  'block_08': torch.randn(1, 2, 1024, 768),
                  'connector': torch.randn(1, 128, 960), 'action': torch.randn(1, 7)}
        for value in inputs.values():
            value.requires_grad_()
        pred = model(inputs)
        self.assertEqual(tuple(pred['delta_block_02'].shape), (1, 2, 1024, 768))
        self.assertEqual(tuple(pred['delta_connector'].shape), (1, 128, 960))
        cross = torch.autograd.grad(pred['delta_connector'].square().mean(),
                                    inputs['block_02'], retain_graph=True)[0]
        self.assertGreater(float(cross.norm()), 0)  # Connector head truly uses intermediate features.
        cross = torch.autograd.grad(pred['delta_block_02'].square().mean(),
                                    inputs['connector'], retain_graph=True)[0]
        self.assertGreater(float(cross.norm()), 0)
        loss(pred, {k: torch.zeros_like(v) for k, v in pred.items()}, {k: 1. for k in pred}).backward()
        for name, value in inputs.items():
            self.assertGreater(float(value.grad.norm()), 0, name)
        for name, module in model.heads.items():
            self.assertGreater(float(module.weight.grad.norm()), 0, name)
        self.assertGreater(float(model.backbone[0].net[1].weight.grad.norm()), 0)

    def test_metrics_zero_vectors_and_layer_weighting(self):
        metrics = Metrics({'delta_connector': 2.})
        x = torch.zeros(2, 4, 12)
        metrics.add({'delta_connector': x}, {'delta_connector': x})
        row = metrics.result()['delta_connector']
        self.assertIsNone(row['cosine'])
        self.assertEqual(row['zero_targets'], 2)
        self.assertIsNone(row['relative_mse_vs_zero'])
        value = loss({'a': torch.ones(1, 100), 'b': torch.ones(1, 2)*2},
                     {'a': torch.zeros(1, 100), 'b': torch.zeros(1, 2)}, {'a': 1., 'b': 2.})
        self.assertEqual(float(value), 1.)

    def test_single_batch_overfit_smoke_and_reload(self):
        report = train.smoke(self.args(self.root/'smoke'), self.cfg, self.records())
        self.assertLess(report['overfit_final'], .95*report['overfit_initial'])
        self.assertTrue(any(report['parameters_changed'].values()))
        model = MultiLayerCorrector(ModelConfig('joint', (2,4,8), (4,12), 8, 1))
        batch = next(iter(torch.utils.data.DataLoader(MultiLayerTransitionDataset(self.records(),'train'), batch_size=2)))
        p = model(batch['inputs'])
        train.atomic_save(self.root/'model.pt', model.state_dict())
        other = MultiLayerCorrector(model.cfg)
        other.load_state_dict(torch.load(self.root/'model.pt', weights_only=True))
        for key, value in p.items():
            torch.testing.assert_close(value, other(batch['inputs'])[key], rtol=0, atol=0)

    def test_short_ab_training_same_steps_and_resume_exact(self):
        records = self.records()
        args = self.args(self.root/'joint')
        first = train.fit(args, self.cfg, records)
        self.assertTrue(first['checkpoint_reload_verified'])
        args.resume = True
        resumed = train.fit(args, self.cfg, records)
        self.assertEqual(first['best_validation'], resumed['best_validation'])
        baseline = train.fit(self.args(self.root/'connector', variant='connector'), self.cfg, records)
        self.assertEqual(first['optimizer_steps'], baseline['optimizer_steps'])
        args.lr = .1
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            train.fit(args, self.cfg, records)

    def test_epoch_boundary_pause_and_recovery_is_deterministic(self):
        records = self.records()
        reference = train.fit(self.args(self.root/'reference'), self.cfg, records)
        real_write = train.write_json
        def write_and_pause(path, data):
            real_write(path, data)
            if Path(path).name == 'status.json' and data.get('epoch') == 1:
                (Path(path).parent/'STOP').touch()
        with patch.object(train, 'write_json', side_effect=write_and_pause):
            with self.assertRaises(ResourcePause):
                train.fit(self.args(self.root/'paused'), self.cfg, records)
        (self.root/'paused/STOP').unlink()
        resumed = train.fit(self.args(self.root/'paused', resume=True), self.cfg, records)
        self.assertEqual(reference['best_validation'], resumed['best_validation'])
        self.assertEqual(reference['optimizer_steps'], resumed['optimizer_steps'])

    def test_readonly_sources_output_protection_and_existing_output(self):
        records = self.records()
        paths = [p for p in self.root.rglob('*') if p.is_file()]
        before = {str(p): digest(p) for p in paths}
        train.fit(self.args(self.root/'run'), self.cfg, records)
        self.assertEqual(before, {str(p): digest(p) for p in paths})
        with self.assertRaises(FileExistsError):
            train.fit(self.args(self.root/'run'), self.cfg, records)
        with self.assertRaises(ValueError):
            safe_output(Path(self.cfg['features_root'])/'bad', self.cfg)
        with self.assertRaises(ValueError):
            safe_output('/tmp/night-code-test/new', self.cfg)

    def test_oom_recovery_restarts_last_complete_epoch(self):
        records = self.records()
        expected = train.fit(self.args(self.root/'complete'), self.cfg, records)
        with patch.object(train, 'reference_update', side_effect=torch.cuda.OutOfMemoryError('synthetic OOM')):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                train.fit(self.args(self.root/'oom'), self.cfg, records)
        state = torch.load(self.root/'oom/latest.pt', weights_only=True)
        self.assertEqual(state['epoch'], 0)
        self.assertEqual(json.loads((self.root/'oom/status.json').read_text())['status'], 'paused')
        recovered = train.fit(self.args(self.root/'oom', resume=True), self.cfg, records)
        self.assertEqual(expected['best_validation'], recovered['best_validation'])

    def test_low_disk_and_gpu_occupied_stop(self):
        with patch('multilayer_mvp.runtime.shutil.disk_usage', return_value=SimpleNamespace(free=1024)):
            with self.assertRaises(ResourcePause):
                resources([self.root], enforce=True)
        with patch('multilayer_mvp.runtime.subprocess.check_output', side_effect=['0, 4090', '999, other.py, 1024']):
            with self.assertRaisesRegex(ResourcePause, 'compute processes'):
                resources([self.root], gpu=True, enforce=True, reserve_gib=0)

    def test_package_sha_and_no_overwrite(self):
        archive = self.root/'source.zip'
        packaging.build(archive)
        sha = packaging.sha256(archive.read_bytes())
        dest = self.root/'deploy'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            packaging.install(archive, '0'*64, dest)
        self.assertFalse(dest.exists())
        packaging.install(archive, sha, dest)
        self.assertTrue((dest/'scripts/corrector_multilayer_train.py').is_file())
        with self.assertRaises(FileExistsError):
            packaging.install(archive, sha, dest)

    def test_bad_shapes_nonfinite_loss_and_mixed_metadata(self):
        with self.assertRaises(ValueError):
            loss({'x': torch.ones(1,2)}, {'x': torch.ones(1,3)}, {'x':1.})
        with self.assertRaises(FloatingPointError):
            loss({'x': torch.ones(1,2)*float('nan')}, {'x': torch.ones(1,2)}, {'x':1.})
        path = Path(self.cfg['features_root'])/'episode_813/metadata.json'
        meta = json.loads(path.read_text())
        meta['split'] = 'validation'
        dump(path, meta)
        self.republish_deep()
        with self.assertRaisesRegex(ValueError, 'split mismatch'):
            self.records()

    def test_paired_cli_common_stop_and_pending_second_variant(self):
        root = self.root/'pair'
        argv = ['corrector_multilayer_train.py', 'paired', '--config', str(self.root/'config.json'),
                '--output', str(root), '--device', 'cpu', '--epochs', '2', '--batch-size', '2',
                '--width', '8', '--depth', '1', '--cpu-threads', '1']
        real_write = train.write_json
        def pause_a(path, data):
            real_write(path, data)
            if Path(path) == root/'connector/status.json' and data.get('epoch') == 1:
                (root/'STOP').touch()
        with patch.object(sys, 'argv', argv), patch.object(train, 'load_config', return_value=self.cfg), \
             patch.object(train, 'write_json', side_effect=pause_a):
            with self.assertRaises(ResourcePause):
                train.main()
        self.assertFalse((root/'joint').exists())
        (root/'STOP').unlink()
        with patch.object(sys, 'argv', argv+['--resume']), patch.object(train, 'load_config', return_value=self.cfg):
            train.main()
        comparison = json.loads((root/'comparison.json').read_text())
        self.assertTrue(comparison['matched_steps'])
        self.assertEqual(comparison['results']['joint']['optimizer_steps'], 8)
        # A complete + B complete is idempotent; no new gradient steps.
        with patch.object(sys, 'argv', argv+['--resume']), patch.object(train, 'load_config', return_value=self.cfg):
            train.main()
        self.assertEqual(json.loads((root/'comparison.json').read_text())['results']['joint']['optimizer_steps'], 8)

    def test_gpu_guard_allows_only_own_process_for_sequential_ab(self):
        import os
        with patch('multilayer_mvp.runtime.subprocess.check_output', side_effect=['0, 4090', f'{os.getpid()}, python, 100']):
            report = resources([self.root], gpu=True, enforce=True, reserve_gib=0)
        self.assertEqual(report['other_gpu_processes'], [])


if __name__ == '__main__':
    unittest.main()
