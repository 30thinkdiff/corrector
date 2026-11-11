#!/usr/bin/env bash
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
