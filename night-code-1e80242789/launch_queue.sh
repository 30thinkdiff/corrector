#!/usr/bin/env bash
set -euo pipefail
queue_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
queue_base=/root/autodl-tmp/corrector
queue_launch_log="$queue_base/object10-night-launch.log"
nohup "$queue_base/bootstrap/bin/python" -u "$queue_code_dir/corrector_night_sweep.py" run "$@" >> "$queue_launch_log" 2>&1 < /dev/null &
queue_pid=$!
echo "queue PID=$queue_pid; launch log=$queue_launch_log"
sleep 1
if ! kill -0 "$queue_pid" 2>/dev/null; then
    tail -n 20 "$queue_launch_log"
    exit 1
fi
