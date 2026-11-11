"""只打包标准库36h监督器；已有训练部署包不改动。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def build(output):
    script=(Path(__file__).parent/'corrector_guardian.py').read_bytes();ast.parse(script)
    files={'corrector_guardian.py':script,'launch_guardian.sh':b'''#!/usr/bin/env bash
set -euo pipefail
guardian_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /root/corrector_supervision_36h
nohup bash "$guardian_code_dir/guardian_watch.sh" "$@" >> /root/corrector_supervision_36h/launch.log 2>&1 < /dev/null &
guardian_pid=$!
echo "guardian PID=$guardian_pid; log=/root/corrector_supervision_36h/launch.log"
sleep 1
if ! kill -0 "$guardian_pid" 2>/dev/null; then tail -n 20 /root/corrector_supervision_36h/launch.log; exit 1; fi
'''}
    files['guardian_watch.sh']=b'''#!/usr/bin/env bash
set -uo pipefail
guardian_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
guardian_args=("$@")
for guardian_attempt in 1 2 3; do
    if /root/autodl-tmp/corrector/bootstrap/bin/python -u "$guardian_code_dir/corrector_guardian.py" run "${guardian_args[@]}"; then exit 0; else guardian_rc=$?; fi
    if [[ "$guardian_rc" != 137 && "$guardian_rc" != 139 ]] || [[ "$guardian_attempt" == 3 ]]; then exit "$guardian_rc"; fi
    echo "guardian interrupted: exit=$guardian_rc; bounded restart attempt=$guardian_attempt"
    guardian_next_args=()
    guardian_skip_next=0
    for guardian_arg in "${guardian_args[@]}"; do
        if [[ "$guardian_skip_next" == 1 ]]; then guardian_skip_next=0; continue; fi
        if [[ "$guardian_arg" == --extend-hours ]]; then guardian_skip_next=1; continue; fi
        if [[ "$guardian_arg" == --extend-hours=* ]]; then continue; fi
        guardian_next_args+=("$guardian_arg")
    done
    guardian_args=("${guardian_next_args[@]}")
    sleep 60
done
'''
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:10]
    files['guardian_bundle.json']=json.dumps({'code_id':identity,'files':hashes},indent=2).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in files.items():archive.writestr(name,content)
    print(json.dumps({'file':str(output),'code_id':identity,'bytes':output.stat().st_size,
                      'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/autodl_setup/corrector_guardian_36h.zip'))
    build(p.parse_args().output)
