"""在现有4090项目根目录初始化Git：离线接收bundle，不覆盖现有工作文件。"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

DEFAULT_ROOT=Path('/root/autodl-tmp/corrector')
REPOSITORY='https://github.com/30thinkdiff/corrector.git'
PUBLIC_EMAIL='154583857+30thinkdiff@users.noreply.github.com'


def run_git(root,*args):
    return subprocess.check_output(['git','-c','core.hooksPath=/dev/null','-C',str(root),*args],stderr=subprocess.STDOUT)


def initialize(root,bundle,expected,repository=REPOSITORY):
    if not re.fullmatch('[0-9a-f]{40}',expected):raise ValueError('需要完整40位提交SHA')
    if not root.is_dir():raise RuntimeError('项目目录不存在，不创建替代目录')
    existed=(root/'.git').exists()
    if existed:
        try:head=run_git(root,'rev-parse','--verify','HEAD').decode().strip()
        except subprocess.CalledProcessError:head=None
        if head and head!=expected:raise RuntimeError('已有不同Git历史，不重置、不覆盖')
        if not head and run_git(root,'ls-files').strip():raise RuntimeError('已有暂存文件，保留index，不自动初始化')
    else:
        run_git(root,'init','-b','main')
        head=None
    try:origin=run_git(root,'remote','get-url','origin').decode().strip()
    except subprocess.CalledProcessError:origin=None
    if origin and origin not in {repository,'git@github.com:30thinkdiff/corrector.git'}:
        raise RuntimeError('已有不同origin，不自动替换')
    run_git(root,'bundle','verify',str(bundle.resolve()))
    run_git(root,'fetch',str(bundle.resolve()),'refs/heads/main')
    fetched=run_git(root,'rev-parse','FETCH_HEAD').decode().strip()
    if fetched!=expected:raise RuntimeError('bundle提交与已发布SHA不符')
    if not head:
        run_git(root,'symbolic-ref','HEAD','refs/heads/main')
        run_git(root,'update-ref','refs/heads/main',expected,'0'*40)
        run_git(root,'read-tree',expected)  # 不带-u：不覆盖已存在的工作文件。
    added=[];preserved=[]
    tree=run_git(root,'ls-tree','-rz','HEAD')
    for entry in tree.split(b'\0'):
        if not entry:continue
        metadata,name=entry.split(b'\t',1);mode,kind,blob=metadata.decode().split();relative=Path(os.fsdecode(name))
        if relative.is_absolute() or '..' in relative.parts or kind!='blob' or mode=='120000':raise RuntimeError('拒绝不安全的Git树项')
        path=root/relative
        if path.exists() or path.is_symlink():preserved.append(str(relative));continue
        path.parent.resolve().relative_to(root.resolve())
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(run_git(root,'cat-file','blob',blob))
        if mode=='100755':path.chmod(0o755)
        added.append(str(relative))
    if not origin:run_git(root,'remote','add','origin',repository)
    run_git(root,'update-ref','refs/remotes/origin/main',expected)
    run_git(root,'branch','--set-upstream-to=origin/main','main')
    run_git(root,'config','core.filemode','false')
    run_git(root,'config','user.name','30thinkdiff')
    run_git(root,'config','user.email',PUBLIC_EMAIL)
    status=run_git(root,'status','--short').decode()
    return {'root':str(root),'head':expected,'origin':repository,'preserved_existing_files':len(preserved),
            'materialized_missing_files':len(added),'status':status,'existing_data_and_environments_untouched':True}


def apply(args):
    import json
    cache=Path('/root/corrector_git_import');cache.mkdir(parents=True,exist_ok=True)
    target=cache/(args.commit+'.bundle');partial=target.with_suffix('.bundle.partial')
    h=hashlib.sha256();count=0
    with partial.open('wb') as file:
        while True:
            block=sys.stdin.buffer.read(2**20)
            if not block:break
            count+=len(block)
            if count>32*2**20:raise RuntimeError('导入bundle超过32MiB，未初始化项目')
            h.update(block);file.write(block)
    if h.hexdigest()!=args.bundle_sha256:raise RuntimeError('bundle校验失败，未初始化项目')
    partial.replace(target)
    result=initialize(args.root,target,args.commit)
    (cache/'last_setup.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def install(args):
    if not args.bundle.is_file():raise FileNotFoundError(args.bundle)
    if not args.commit:
        heads=subprocess.check_output(['git','bundle','list-heads',str(args.bundle),'refs/heads/main'],text=True).splitlines()
        if len(heads)!=1:raise RuntimeError('bundle没有唯一main分支')
        args.commit=heads[0].split()[0]
    digest=hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    source=Path(__file__).read_text()
    remote=['/root/autodl-tmp/corrector/bootstrap/bin/python','-c',source,'apply',
            '--root',str(args.root),'--commit',args.commit,'--bundle-sha256',digest]
    command=['ssh','-T','-o','ConnectTimeout=15']
    if args.ssh_port:command+=['-p',str(args.ssh_port)]
    command += [args.ssh_host,shlex.join(remote)]
    print('请在终端完成SSH认证；在现有项目根目录初始化Git，不下载数据或覆盖已有代码。',flush=True)
    with args.bundle.open('rb') as file:result=subprocess.run(command,stdin=file)
    if result.returncode:raise RuntimeError('远程Git初始化未完成；请保留输出定位，未强制覆盖工作文件')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('apply');p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--commit',required=True)
    p.add_argument('--bundle-sha256',required=True)
    p=sub.add_parser('install');p.add_argument('--ssh-host',default='connect.nmb1.seetacloud.com');p.add_argument('--ssh-port',type=int)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--commit')
    args=parser.parse_args()
    if args.mode=='apply':apply(args)
    else:install(args)

if __name__=='__main__':main()
