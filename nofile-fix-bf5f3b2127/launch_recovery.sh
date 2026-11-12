#!/usr/bin/env bash
set -euo pipefail
fix_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /root/corrector_supervision_36h
nohup /root/autodl-tmp/corrector/bootstrap/bin/python -u "$fix_code_dir/corrector_recover_nofile.py" "$@" >> /root/corrector_supervision_36h/nofile-recovery-launch.log 2>&1 < /dev/null &
fix_pid=$!
echo "recovery PID=$fix_pid; log=/root/corrector_supervision_36h/nofile-recovery-launch.log"
sleep 1
if ! kill -0 "$fix_pid" 2>/dev/null; then tail -n 25 /root/corrector_supervision_36h/nofile-recovery-launch.log; exit 1; fi
