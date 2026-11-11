#!/usr/bin/env bash
set -euo pipefail
guardian_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /root/corrector_supervision_36h
nohup bash "$guardian_code_dir/guardian_watch.sh" "$@" >> /root/corrector_supervision_36h/launch.log 2>&1 < /dev/null &
guardian_pid=$!
echo "guardian PID=$guardian_pid; log=/root/corrector_supervision_36h/launch.log"
sleep 1
if ! kill -0 "$guardian_pid" 2>/dev/null; then tail -n 20 /root/corrector_supervision_36h/launch.log; exit 1; fi
