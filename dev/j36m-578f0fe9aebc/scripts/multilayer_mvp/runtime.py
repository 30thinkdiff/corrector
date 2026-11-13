"""Resource checks and immutable provenance; no environment/package modification."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

from corrector_data_workbench import digest


class ResourcePause(RuntimeError):
    pass


def source_manifest():
    scripts = Path(__file__).resolve().parents[1]
    paths = [*sorted((scripts / 'multilayer_mvp').glob('*.py')),
             *(scripts / name for name in ('corrector_multilayer_train.py', 'corrector_train.py',
                                           'corrector_models.py', 'corrector_data_workbench.py'))]
    files = {str(p.relative_to(scripts)): digest(p) for p in paths}
    try:
        sha = subprocess.check_output(['git', '-C', str(scripts), 'rev-parse', 'HEAD'],
                                      stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = None
    build = scripts.parent / 'DEPLOYMENT_MANIFEST.json'
    build_sha = json.loads(build.read_text()).get('git_commit') if build.exists() else sha
    return {'build_git_sha': build_sha, 'git_sha_at_execution': sha, 'files': files,
            'source_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def resources(paths, gpu=False, enforce=False, reserve_gib=2):
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'disks': {}}
    for raw in paths:
        p = Path(raw).resolve()
        while not p.exists():
            p = p.parent
        free = shutil.disk_usage(p).free / 2**30
        report['disks'][str(p)] = {'free_gib': free}
        if enforce and free < reserve_gib:
            raise ResourcePause(f'disk free {free:.2f}GiB < reserve {reserve_gib}: {p}')
    if Path('/proc/meminfo').exists():
        fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        report['host_available_ram_gib'] = int(fields['MemAvailable'].split()[0]) / 2**20
    cgroup = Path('/sys/fs/cgroup')
    for name in ('memory.max', 'memory.current', 'cpu.max'):
        p = cgroup / name
        if p.exists():
            report['cgroup_' + name.replace('.', '_')] = p.read_text().strip()
    ceiling = report.get('cgroup_memory_max')
    current = report.get('cgroup_memory_current')
    if ceiling and ceiling != 'max' and current:
        report['effective_available_ram_gib'] = min(
            report.get('host_available_ram_gib', float('inf')), (int(ceiling)-int(current))/2**30)
        if enforce and report['effective_available_ram_gib'] < 1.:
            raise ResourcePause('less than 1GiB effective RAM available')
    try:
        report['cpu_affinity_count'] = len(os.sched_getaffinity(0))
    except AttributeError:
        report['cpu_affinity_count'] = os.cpu_count()
    if gpu:
        report['gpu'] = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        processes = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'], text=True).strip()
        report['gpu_processes'] = processes
        external = [row for row in processes.splitlines() if row.split(',', 1)[0].strip() != str(os.getpid())]
        report['other_gpu_processes'] = external
        if enforce and external:
            raise ResourcePause('GPU has compute processes; do not start concurrent experiment: ' + '; '.join(external))
    return report


def safe_output(output, cfg):
    out = Path(output).resolve()
    protected = [Path(cfg['features_root']).resolve(),
                 *(Path(r['raw_path']).resolve() for r in cfg['episodes'])]
    # Also keep outputs away from frozen deployment and model/cache trees.
    frozen = ('night-code-', 'action-code-', 'guardian-code-', 'queue-code-', 'training-code')
    if any(out == p or p in out.parents or out in p.parents for p in protected):
        raise ValueError('output intersects read-only feature data')
    if any(x.startswith(frozen) or x in ('hf', 'venv', 'sim-venv') for x in out.parts):
        raise ValueError('output lies inside frozen code/environment/weights')
    return out
