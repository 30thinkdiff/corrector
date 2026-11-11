"""并发消费者只读已发布Hub缓存；构造文件和编码回调，不访问网络/GPU。"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('pipeline_stream', ROOT/'scripts/corrector_stream_workbench.py')
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


def complete_source(root, name, content):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    metadata = root/'.cache/huggingface/download'/(name+'.metadata')
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(stream.data.DATA_SHA+'\n'+stream.data.digest(path)+'\n1.0\n')
    return {'path':name, 'bytes':len(content)}, path, metadata


class LocalPipelineTests(unittest.TestCase):
    def test_requires_final_file_metadata_commit_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, path, metadata = complete_source(root, 'data/a.parquet', b'1234')
            checked = {}
            self.assertTrue(stream.local_shard_ready(root, spec, stream.data.DATA_SHA, checked))
            with patch.object(stream.data, 'digest', side_effect=AssertionError('重复hash')):
                self.assertTrue(stream.local_shard_ready(root, spec, stream.data.DATA_SHA, checked))
            path.write_bytes(b'123')
            self.assertFalse(stream.local_shard_ready(root, spec, stream.data.DATA_SHA, checked))
            path.write_bytes(b'5678')
            with self.assertRaisesRegex(RuntimeError, 'SHA256不符'):
                stream.local_shard_ready(root, spec, stream.data.DATA_SHA, checked)
            metadata.write_text('b'*40+'\n'+'a'*64+'\n1.0\n')
            with self.assertRaisesRegex(RuntimeError, 'revision不同'):
                stream.local_shard_ready(root, spec, stream.data.DATA_SHA, {})
            metadata.unlink()
            self.assertFalse(stream.local_shard_ready(root, spec, stream.data.DATA_SHA, {}))
            path.rename(path.with_suffix('.parquet.incomplete'))
            self.assertFalse(stream.local_shard_ready(root, spec, stream.data.DATA_SHA, {}))

    def test_missing_first_episode_does_not_block_ready_one_then_arrival(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b, _, _ = complete_source(root, 'data/b.parquet', b'bbbb')
            a = {'path':'data/a.parquet', 'bytes':4}
            specs = {1:{'files':[a]}, 2:{'files':[b]}}
            messages = []
            def arrival(_):
                complete_source(root, 'data/a.parquet', b'aaaa')
            def waiting(count, elapsed): messages.append(count)
            with patch.object(stream.time, 'sleep', side_effect=arrival):
                result = list(stream.ready_episodes([1,2], specs, root, stream.data.DATA_SHA,
                    check_stop=lambda:None, on_wait=waiting, poll_seconds=.01))
            self.assertEqual(result, [2,1])
            self.assertEqual(messages, [1])

    def test_idle_timeout_and_stop_do_not_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {1:{'files':[{'path':'data/missing.parquet','bytes':4}]}}
            with patch.object(stream.time, 'monotonic', side_effect=[0.,61.]), \
                    patch.object(stream.time, 'sleep') as sleep:
                with self.assertRaisesRegex(stream.BudgetStop, 'waiting_for_local_source_timeout'):
                    list(stream.ready_episodes([1],specs,Path(tmp),stream.data.DATA_SHA,
                         check_stop=lambda:None,on_wait=lambda *a:None,idle_minutes=1))
                sleep.assert_not_called()
            def stop(): raise stream.BudgetStop('operator_stop_file')
            with self.assertRaisesRegex(stream.BudgetStop,'operator_stop_file'):
                next(stream.ready_episodes([1],specs,Path(tmp),stream.data.DATA_SHA,
                     check_stop=stop,on_wait=lambda *a:None))

    def test_cross_shard_episode_waits_for_all_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a,_,_=complete_source(root,'data/a.parquet',b'aaaa')
            b={'path':'data/b.parquet','bytes':4}
            specs={1:{'files':[a,b]},2:{'files':[a]}}
            with patch.object(stream.time,'sleep',side_effect=lambda _:complete_source(root,'data/b.parquet',b'bbbb')):
                result=list(stream.ready_episodes([1,2],specs,root,stream.data.DATA_SHA,
                    check_stop=lambda:None,on_wait=lambda *a:None))
            self.assertEqual(result,[2,1])

    def test_same_archive_accepts_only_one_feature_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with stream.feature_writer_lock(Path(tmp)):
                with self.assertRaisesRegex(RuntimeError, '已有特征提取进程'):
                    with stream.feature_writer_lock(Path(tmp)):
                        self.fail('不应获得第二把锁')
            with stream.feature_writer_lock(Path(tmp)):
                pass

    def test_watch_consumes_local_only_preserves_source_and_resumes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); work=root/'work'; archive=root/'archive'
            work.mkdir(); archive.mkdir()
            fixture=root/'fixture.parquet'
            rows=[{'episode_index':ep,'frame_index':i,'task_index':24,'action':[0.]*7}
                  for ep in [813,825] for i in range(2)]
            pq.write_table(pa.Table.from_pylist(rows), fixture)
            shard, original, _ = complete_source(work/'source', 'data/a.parquet', fixture.read_bytes())
            manifest={'revision':stream.data.DATA_SHA, 'records':[
                {'episode':ep,'frames':2,'task_id':24,'task':'task','split':split,'files':[shard]}
                for ep,split in [(813,'train'),(825,'validation')]]}
            (archive/'manifest.json').write_text(json.dumps(manifest))
            args=SimpleNamespace(workdir=work,archive=archive,source_root=root,watch_local=True,
                                 max_new_episodes=0,minutes=0,keep_raw=False,min_free_gib=0.)
            calls=[]
            def encode(options):
                self.assertTrue(options.cache_only)
                for ep in options.episode_source():
                    options.before_episode(ep);calls.append(ep)
                    local=work/'features'/f'episode_{ep}';local.mkdir(parents=True)
                    split='train' if ep==813 else 'validation'
                    (local/'metadata.json').write_text(json.dumps({'policy_revision':stream.data.POLICY_SHA,
                        'corrector_revision':stream.data.CORRECTOR_SHA,'split':split,'num_frames':2}))
                    final=archive/'features'/f'episode_{ep}';final.parent.mkdir(exist_ok=True)
                    stream.data.publish(local,final)
                    options.on_complete(ep,{'frames':2,'extraction_seconds':.1})
            with patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('禁止下载')) as hub, \
                    patch.object(stream,'release_source_files',side_effect=AssertionError('禁止清理producer文件')), \
                    patch.object(stream.data,'extract',side_effect=encode) as extract:
                stream.run_stream(args,manifest)
                self.assertEqual(extract.call_count,1)
                hub.assert_not_called()
                stream.run_stream(args,manifest)
            self.assertEqual(calls,[813,825])
            self.assertTrue(original.is_file())
            report=json.loads((archive/'stream_report.json').read_text())
            self.assertEqual(report['attempts'][0]['released_source_bytes'],0)
            self.assertEqual(report['attempts'][0]['status'],'finished_selected_batch')


if __name__ == '__main__':
    unittest.main()
