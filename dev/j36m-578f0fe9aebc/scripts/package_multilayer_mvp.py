"""Build/verify a source-only deployment. Stdlib only; never includes runtime data."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOC = 'studies/policy_generalization/corrector/multilayer_mvp'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_paths():
    paths = [*sorted((ROOT/'scripts/multilayer_mvp').glob('*.py')),
             *sorted((ROOT/'tests/multilayer_mvp').glob('*.py')),
             *sorted((ROOT/DOC).glob('*'))]
    paths += [ROOT/'scripts'/name for name in (
        'corrector_multilayer_train.py', 'corrector_train.py', 'corrector_models.py',
        'corrector_data_workbench.py', 'package_multilayer_mvp.py')]
    paths += [ROOT/'AGENTS.md', ROOT/'THIRD_PARTY.md']
    return [p for p in paths if p.is_file()]


def build(output):
    paths = source_paths()
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
    if any(Path(n).suffix not in ('.py', '.md', '.json') for n in files):
        raise ValueError('source-only allowlist violated')
    build_manifest = ROOT/'DEPLOYMENT_MANIFEST.json'
    git_sha = (json.loads(build_manifest.read_text())['git_commit'] if build_manifest.exists() else
               subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
    manifest = {'format': 'multilayer_mvp_deployment_v1', 'git_commit': git_sha,
                'files': {n: sha256(v) for n, v in files.items()},
                'raw_data_included': False, 'weights_included': False}
    if output.exists():
        raise FileExistsError('package already exists; use a new path')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, value in files.items():
            z.writestr(name, value)
        z.writestr('DEPLOYMENT_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    checksum = sha256(output.read_bytes())
    output.with_suffix(output.suffix+'.sha256').write_text(f'{checksum}  {output.name}\n')
    print(json.dumps({'zip': str(output), 'sha256': checksum, 'files': len(files),
                      'bytes': output.stat().st_size, 'git_commit': git_sha}))


def install(archive, expected, destination):
    if sha256(archive.read_bytes()) != expected:
        raise ValueError('ZIP SHA256 mismatch; nothing extracted')
    if destination.exists():
        raise FileExistsError('destination exists; never overwrite deployment')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive members')
        manifest = json.loads(z.read('DEPLOYMENT_MANIFEST.json'))
        if manifest.get('format') != 'multilayer_mvp_deployment_v1':
            raise ValueError('invalid package format')
        if set(names) != set(manifest['files']) | {'DEPLOYMENT_MANIFEST.json'}:
            raise ValueError('unexpected archive member')
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('unsafe archive path/type')
            if info.file_size > 2**20:
                raise ValueError('not a source-only package')
        for name, digest in manifest['files'].items():
            if sha256(z.read(name)) != digest:
                raise ValueError(f'package member SHA mismatch: {name}')
        destination.mkdir(parents=True, exist_ok=False)
        for name in names:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write(z.read(name))
    print(json.dumps({'installed': str(destination), 'git_commit': manifest['git_commit'],
                      'experiments_started': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['build', 'install'])
    p.add_argument('--output', type=Path)
    p.add_argument('--archive', type=Path)
    p.add_argument('--sha256')
    p.add_argument('--destination', type=Path)
    args = p.parse_args()
    if args.mode == 'build':
        if args.output is None:
            p.error('build needs --output')
        build(args.output)
    else:
        if None in (args.archive, args.sha256, args.destination):
            p.error('install needs --archive --sha256 --destination')
        install(args.archive, args.sha256, args.destination)
