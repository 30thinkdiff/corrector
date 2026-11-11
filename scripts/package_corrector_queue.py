"""为已安装环境生成独立队列部署包；不包含数据、模型或账号信息。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

from sync_corrector_notebook import ROOT, SOURCES


def build(output, night=False, followup=False):
    files = {}
    for key, relative in SOURCES.items():
        name = 'corrector_online_practice.py' if key == 'ONLINE_PRACTICE_SOURCE' else Path(relative).name
        files[name] = (ROOT/relative).read_bytes()
    files['corrector_training_queue.py'] = (ROOT/'scripts/corrector_training_queue.py').read_bytes()
    if night or followup:
        for name in ('corrector_night_sweep.py','corrector_temporal_audit.py'):
            files[name]=(ROOT/'scripts'/name).read_bytes()
    if followup:
        files['corrector_followup_queue.py']=(ROOT/'scripts/corrector_followup_queue.py').read_bytes()
    for name, content in files.items():
        if name.endswith('.py'):
            ast.parse(content)
    files['launch_queue.sh'] = b'''#!/usr/bin/env bash
set -euo pipefail
queue_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
queue_base=/root/autodl-tmp/corrector
queue_launch_log="$queue_base/object10-queue-launch.log"
nohup "$queue_base/bootstrap/bin/python" -u "$queue_code_dir/corrector_training_queue.py" run "$@" >> "$queue_launch_log" 2>&1 < /dev/null &
queue_pid=$!
echo "queue PID=$queue_pid; launch log=$queue_launch_log"
sleep 1
if ! kill -0 "$queue_pid" 2>/dev/null; then
    tail -n 20 "$queue_launch_log"
    exit 1
fi
'''
    if night:
        files['launch_queue.sh']=files['launch_queue.sh'].replace(b'corrector_training_queue.py',b'corrector_night_sweep.py').replace(b'object10-queue-launch.log',b'object10-night-launch.log')
    if followup:
        files['launch_queue.sh']=files['launch_queue.sh'].replace(b'corrector_training_queue.py',b'corrector_followup_queue.py').replace(b'object10-queue-launch.log',b'object10-action-followup-launch.log')
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
    files['queue_bundle_manifest.json'] = json.dumps({'code_id': identity, 'files': hashes}, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    print(json.dumps({'output': str(output), 'code_id': identity,
                      'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))
    return identity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--night',action='store_true')
    parser.add_argument('--followup',action='store_true')
    args=parser.parse_args()
    build(args.output,night=args.night,followup=args.followup)
