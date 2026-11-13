"""Strict read-only adapter for multilayer_raw_fp16_v1 + siglip_frame_cache_v2.

Only current-frame inputs leave this module. No VLA, image decoding, or downloads.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from corrector_data_workbench import digest

POLICY = 'a24f1f380461b665158d52acd140152c79e8b36d'
WEIGHT = '9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8'
CAMERAS = ['observation.images.camera1', 'observation.images.camera2']
IMAGES = ['observation.images.image', 'observation.images.image2']
LAYERS = ('block_02', 'block_08', 'connector')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_marker(directory, raw=False):
    marker = read_json(directory / 'COMPLETE.json')
    require(isinstance(marker.get('files'), dict) and marker['files'], 'missing completion files')
    for name, entry in marker['files'].items():
        require(Path(name).name == name and name not in ('.', '..'), 'unsafe marker path')
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f'missing/symlink file: {path}')
        sha = entry['sha256'] if raw else entry
        require(digest(path) == sha, f'SHA256 mismatch: {path}')
        if raw:
            require(path.stat().st_size == entry['size_bytes'], f'size mismatch: {path}')
    return marker


def load_config(path):
    cfg = read_json(path)
    require(cfg['k_step'] == 10 and cfg['fps'] == 10.0, 'MVP requires k10 at data 10Hz')
    require(cfg['policy_revision'] == POLICY and cfg['policy_weight_sha256'] == WEIGHT,
            'frozen policy revision/weight mismatch')
    require(cfg['raw_shape'] == [2, 1024, 768] and cfg['connector_shape'] == [128, 960],
            'production requires unpooled full spatial supervision')
    ids = [r['episode'] for r in cfg['episodes']]
    require(len(ids) == len(set(ids)), 'duplicate episode / split leakage')
    require({r['split'] for r in cfg['episodes']} == {'train', 'validation'}, 'missing/unknown split')
    require(all(r['split'] in ('train', 'validation') for r in cfg['episodes']), 'unknown split')
    return cfg


class MultiLayerTransitionDataset(Dataset):
    """One split, already validated mmap handles; targets never passed as inputs."""
    def __init__(self, records, split, k_step=10, connector_only=False):
        self.records = [r for r in records if r['spec']['split'] == split]
        require(bool(self.records), f'empty split: {split}')
        self.k_step, self.connector_only = k_step, connector_only
        self.ends = np.cumsum([r['spec']['frames'] - k_step for r in self.records]).tolist()
        require(all(r['spec']['frames'] > k_step for r in self.records), 'no legal transition')

    def __len__(self):
        return self.ends[-1]

    def locate(self, index):
        require(0 <= index < len(self), 'sample index out of range')
        e = bisect.bisect_right(self.ends, index)
        t = index - (self.ends[e - 1] if e else 0)
        return self.records[e], t

    def __getitem__(self, index):
        record, t = self.locate(index)
        a, k = record['arrays'], self.k_step
        now = {'connector': a['z_q'][t].astype('float32') * a['z_scale'][t].astype('float32')[:, None]}
        future = {'connector': a['z_q'][t+k].astype('float32') * a['z_scale'][t+k].astype('float32')[:, None]}
        if not self.connector_only:
            for name in LAYERS[:2]:
                now[name] = a[name][t].astype('float32')
                future[name] = a[name][t+k].astype('float32')
        inputs = dict(now, action=a['actions'][t].astype('float32'))
        targets = {f'delta_{name}': future[name] - now[name] for name in now}
        require(all(np.isfinite(x).all() for x in (*inputs.values(), *targets.values())),
                f'nonfinite sample: episode {record["spec"]["episode"]} t={t}')
        return {'inputs': inputs, 'targets': targets, 'episode': record['spec']['episode'], 't': t}


def validate(cfg):
    """Hash all published files once per command; return only read-only memmaps."""
    records = []
    ids = [s['episode'] for s in cfg['episodes']]
    require(len(ids) == len(set(ids)), 'duplicate episode / split leakage')
    for spec in cfg['episodes']:
        ep, n = spec['episode'], spec['frames']
        old = Path(cfg['features_root']) / f'episode_{ep}'
        raw = Path(spec['raw_path'])
        require(old.resolve() != raw.resolve(), 'raw/deep directories must differ')
        deep_complete = verify_marker(old)
        published = set(deep_complete['files'])
        require({'metadata.json', 'z_q.npy', 'z_scale.npy', 'actions.npy', 'episode_index.npy',
                 'frame_index.npy', 'timestamp.npy'} <= published, 'incomplete Connector manifest')
        meta, marker = read_json(old / 'metadata.json'), verify_marker(raw, raw=True)
        checks = {
            'format': 'multilayer_raw_fp16_v1', 'episode': ep, 'num_frames': n,
            'fps': cfg['fps'], 'layer_indices_zero_based': [2, 8], 'camera_keys': CAMERAS,
            'source_image_keys': IMAGES, 'dtype': 'float16', 'pooling': 'none',
            'shape_per_layer': [n, *cfg['raw_shape']], 'policy_revision': cfg['policy_revision'],
            'policy_weight_sha256': cfg['policy_weight_sha256'], 'k_data_frames': cfg['k_step'],
            'deep_cache_complete_sha256': digest(old / 'COMPLETE.json'),
            'frame_index_and_timestamp_source': 'deep_cache_path',
            'action_source': 'deep_cache_path/actions.npy; checkpoint_processor_normalized',
        }
        for key, expected in checks.items():
            require(marker.get(key) == expected, f'raw {ep}: {key} mismatch')
        # Relocation is allowed; content binding is the deep COMPLETE SHA, not a machine path.
        for key, expected in {
            'format': 'siglip_frame_cache_v2', 'num_frames': n, 'fps': cfg['fps'],
            'split': spec['split'], 'camera_keys': CAMERAS, 'image_keys': IMAGES,
            'policy_id': 'lerobot/smolvla_libero', 'policy_revision': cfg['policy_revision'],
            'action_representation': 'checkpoint_processor_normalized',
            'l_tokens': cfg['connector_shape'][0], 'd_dim': cfg['connector_shape'][1], 'action_dim': 7,
            'dtype_actions': 'float16', 'dtype_z_q': 'int8', 'dtype_z_scale': 'float16',
        }.items():
            require(meta.get(key) == expected, f'Connector {ep}: {key} mismatch')
        p = meta['raw_provenance']
        require((p['episode'], p['split'], p['num_frames'], p['fps'], p['task_id']) ==
                (ep, spec['split'], n, cfg['fps'], spec['task_id']), 'raw provenance mismatch')
        require(marker['source_parquet_sha256'] in [s['sha256'] for s in p['sources']],
                'raw/Connector source parquet mismatch')
        arrays = {}
        files = {'z_q': ('int8', (n, *cfg['connector_shape'])),
                 'z_scale': ('float16', (n, cfg['connector_shape'][0])),
                 'actions': ('float16', (n, 7)), 'frame_index': (None, (n,)),
                 'episode_index': (None, (n,)), 'timestamp': (None, (n,))}
        for key, (dtype, shape) in files.items():
            value = np.load(old / f'{key}.npy', mmap_mode='r', allow_pickle=False)
            require(value.shape == shape and (dtype is None or value.dtype == dtype), f'bad {key} shape/dtype')
            arrays[key] = value
        require(np.issubdtype(arrays['frame_index'].dtype, np.integer) and
                np.issubdtype(arrays['episode_index'].dtype, np.integer), 'noninteger frame/episode IDs')
        require(np.array_equal(arrays['frame_index'], np.arange(n)), 'noncontinuous frame index')
        require(np.all(arrays['episode_index'] == ep), 'mixed episode array')
        require(np.allclose(arrays['timestamp'], np.arange(n) / cfg['fps'], atol=2e-5, rtol=0),
                'timestamp/data FPS mismatch')
        require(np.isfinite(arrays['actions']).all() and np.isfinite(arrays['z_scale']).all()
                and (arrays['z_scale'] > 0).all(), 'invalid action/scale')
        require(set(marker['files']) == {'block_02.fp16.npy', 'block_08.fp16.npy'}, 'wrong raw layers')
        for name in LAYERS[:2]:
            value = np.load(raw / f'{name}.fp16.npy', mmap_mode='r', allow_pickle=False)
            require(value.shape == (n, *cfg['raw_shape']) and value.dtype == 'float16', f'bad {name}')
            arrays[name] = value
        records.append({'spec': spec, 'arrays': arrays, 'provenance': {
            'episode': ep, 'split': spec['split'], 'frames': n, 'task_id': spec['task_id'],
            'deep_complete_sha256': digest(old / 'COMPLETE.json'),
            'raw_complete_sha256': digest(raw / 'COMPLETE.json'),
        }})
    require({r['spec']['split'] for r in records} == {'train', 'validation'}, 'missing split')
    return records


def fit_scales(dataset):
    """Streaming train-only RMS(delta); each layer gets one scalar, no validation data."""
    require(all(r['spec']['split'] == 'train' for r in dataset.records), 'scale fitting must use train only')
    sums, counts = {}, {}
    for i in range(len(dataset)):
        for name, target in dataset[i]['targets'].items():
            sums[name] = sums.get(name, 0.) + float(np.square(target, dtype=np.float64).sum())
            counts[name] = counts.get(name, 0) + target.size
    return {name: max((sums[name] / counts[name]) ** .5, 1e-6) for name in sums}
