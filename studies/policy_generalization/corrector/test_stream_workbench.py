"""流式下载协议/页脚定位/逐轨迹恢复；小构造Parquet，不模拟真实视觉推理。"""
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('stream_workbench',ROOT/'scripts/corrector_stream_workbench.py')
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)


class StreamTests(unittest.TestCase):
    def test_explicit_proxy_is_set_before_clients_and_dead_port_stops_early(self):
        import os
        import socket
        with patch.dict(os.environ,{},clear=True),patch.object(socket,'create_connection') as connect:
            stream.configure_proxy('http://127.0.0.1:17890')
            self.assertEqual(os.environ['HTTPS_PROXY'],'http://127.0.0.1:17890')
            self.assertEqual(os.environ['NO_PROXY'],'localhost,127.0.0.1,::1')
            connect.assert_called_once_with(('127.0.0.1',17890),timeout=5)
        with patch.object(socket,'create_connection',side_effect=ConnectionRefusedError()):
            with self.assertRaises(ConnectionError):stream.configure_proxy('http://127.0.0.1:17890')

    def test_verified_template_requires_no_hub_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(archive=root,workdir=root,task_ids=[24],limit=44,seed=42,feature_budget_gib=4.,
                  manifest_template=ROOT/'studies/policy_generalization/corrector/task24_manifest.json')
            with patch('huggingface_hub.HfApi') as api:
                manifest=stream.make_manifest(args)
                api.assert_not_called()
            self.assertEqual(len(manifest['records']),44)
            self.assertEqual(manifest['frames'],6867)
            self.assertEqual(manifest['records'][0]['files'][0]['path'],'data/chunk-000/file-213.parquet')
    def test_split_is_stable_when_probe_subset_expands(self):
        rows=[{'episode_index':e,'length':155,'tasks':['task'],
               'stats/task_index/min':[24],'stats/task_index/max':[24]} for e in [813,818,825,850,852,857,860,876]]
        small=stream.choose_records(rows,{24},3)
        full=stream.choose_records(rows,{24},0)
        self.assertEqual([r['episode'] for r in small],[813,818,825])
        self.assertEqual({r['episode']:r['split'] for r in small},
                         {r['episode']:r['split'] for r in full if r['episode'] in {813,818,825}})

    def test_footer_resolver_includes_episode_crossing_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            files=[{'path':str(i),'bytes':1} for i in range(4)]
            bounds=[(0,2),(2,4),(5,7),(8,10)]
            r=stream.FooterResolver(files,None,Path(tmp)/'cache.json',reader=lambda p:bounds[int(p)])
            self.assertEqual([f['path'] for f in r.locate(2)],['0','1'])
            with self.assertRaises(RuntimeError):r.locate(11)

    def test_stream_persists_before_cleanup_and_resume_needs_no_old_raw(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';archive=root/'archive'
            work.mkdir();archive.mkdir()
            rows=[{'episode_index':e,'frame_index':i,'task_index':24,'action':[0.]*7} for e in [813,825] for i in range(2)]
            fixture=root/'source.parquet';pq.write_table(pa.Table.from_pylist(rows),fixture)
            manifest={'records':[{'episode':e,'frames':2,'task_id':24,'task':'task','split':s,
                       'files':[{'path':'data/chunk-000/file-213.parquet','bytes':fixture.stat().st_size}]} for e,s in [(813,'train'),(825,'validation')]]}
            (archive/'manifest.json').write_text(json.dumps(manifest))
            args=SimpleNamespace(workdir=work,archive=archive,source_root=root,max_new_episodes=1,
                                 minutes=0,keep_raw=False,min_free_gib=0.)
            interrupt_next = False
            def download(*a,**kw):
                p=Path(kw['local_dir'])/a[1];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fixture,p);return str(p)
            def encode(options):
                nonlocal interrupt_next
                for ep in options.episodes:
                    options.before_episode(ep)
                    raw=options.raw_root/f'episode_{ep}'
                    prov=json.loads((raw/'provenance.json').read_text())
                    local=work/'features'/f'episode_{ep}';local.mkdir(parents=True,exist_ok=True)
                    if interrupt_next:
                        interrupt_next = False
                        # 模拟当前轨迹提取被打断：原始归档完整，特征还没发布。
                        (local/'metadata.json').write_text('{}')
                        raise KeyboardInterrupt()
                    (local/'metadata.json').write_text(json.dumps({'policy_revision':stream.data.POLICY_SHA,
                        'corrector_revision':stream.data.CORRECTOR_SHA,'split':prov['split'],'num_frames':2}))
                    final=archive/'features'/f'episode_{ep}';final.parent.mkdir(exist_ok=True)
                    stream.data.publish(local,final)
                    options.on_complete(ep,{'frames':2,'extraction_seconds':.1,'publish_seconds':.01,'feature_bytes':1})
            with patch('huggingface_hub.hf_hub_download',side_effect=download),patch.object(stream.data,'extract',side_effect=encode):
                stream.run_stream(args,manifest)
                self.assertTrue(stream.data.verified(archive/'features/episode_813'))
                self.assertFalse((work/'raw/episode_813').exists())
                self.assertTrue((work/'source/data/chunk-000/file-213.parquet').is_file())
                interrupt_next = True
                with self.assertRaises(KeyboardInterrupt):
                    stream.run_stream(args,manifest)
                interrupted = json.loads((archive/'stream_report.json').read_text())['attempts'][-1]
                self.assertEqual(interrupted['status'], 'interrupted_or_error')
                self.assertEqual(interrupted['error'], 'KeyboardInterrupt: ')
                self.assertTrue(stream.data.verified(archive/'features/episode_813'))
                self.assertFalse((archive/'features/episode_825/COMPLETE.json').exists())
                stream.run_stream(args,manifest)
            self.assertTrue(stream.data.verified(archive/'features/episode_825'))
            self.assertFalse((work/'source/data/chunk-000/file-213.parquet').exists())
            report=json.loads((archive/'stream_report.json').read_text())
            self.assertTrue(all(v['completed'] for v in report['episodes'].values()))

    def test_release_only_unreferenced_manifest_files(self):
        from collections import Counter
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir()
            for name in ('shared.parquet','finished.parquet','unrelated.parquet'):
                (root/'data'/name).write_bytes(b'fixture')
            released=stream.release_source_files(root,Counter({'data/shared.parquet':1}),
                                                {'data/shared.parquet','data/finished.parquet'})
            self.assertEqual(released,len(b'fixture'))
            self.assertTrue((root/'data/shared.parquet').exists())
            self.assertTrue((root/'data/unrelated.parquet').exists())
            with self.assertRaises(ValueError):
                stream.release_source_files(root,Counter(),{'../unsafe.parquet'})

    def test_stop_file_pauses_before_raw_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';archive=root/'archive'
            work.mkdir();archive.mkdir();(archive/'STOP').touch()
            manifest={'records':[{'episode':1,'frames':2,'files':[{'path':'data/a.parquet','bytes':1}]}]}
            args=SimpleNamespace(workdir=work,archive=archive,source_root=root,
                                 max_new_episodes=0,minutes=0,keep_raw=False)
            def extract(options):options.before_episode(1)
            with patch.object(stream.data,'extract',side_effect=extract),patch('huggingface_hub.hf_hub_download') as download:
                stream.run_stream(args,manifest)
                download.assert_not_called()
            attempt=json.loads((archive/'stream_report.json').read_text())['attempts'][-1]
            self.assertEqual(attempt['stop_reason'],'operator_stop_file')

    def test_disk_guard_pauses_without_downloading_or_deleting_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';archive=root/'archive'
            work.mkdir();archive.mkdir()
            manifest={'records':[{'episode':1,'frames':2,'files':[{'path':'data/a.parquet','bytes':1}]}]}
            args=SimpleNamespace(workdir=work,archive=archive,source_root=root,
                                 max_new_episodes=0,minutes=0,keep_raw=False)
            def extract(options):options.before_episode(1)
            with patch.object(stream.data,'extract',side_effect=extract), \
                 patch('huggingface_hub.hf_hub_download') as download, \
                 patch.object(stream.shutil,'disk_usage',return_value=SimpleNamespace(free=1024)):
                stream.run_stream(args,manifest)
                download.assert_not_called()
            attempt=json.loads((archive/'stream_report.json').read_text())['attempts'][-1]
            self.assertEqual(attempt['stop_reason'],'disk_space')

    def test_incomplete_or_wrong_episode_rows_cannot_be_published(self):
        spec={'episode':813,'frames':2,'task_id':24}
        with self.assertRaises(RuntimeError):stream.validate_rows([],spec)
        bad=[{'episode_index':813,'frame_index':i,'task_index':24,'action':[0.]*7} for i in [0,0]]
        with self.assertRaises(RuntimeError):stream.validate_rows(bad,spec)
