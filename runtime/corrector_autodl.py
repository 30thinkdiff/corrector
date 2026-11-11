"""AutoDL命令行启动器：复用现有准备/评测实现，不依赖Colab或Google Drive挂载。

prepare只准备依赖；night顺序执行准备→真实权重检查→强制事件检查→自然事件批次。
代码与结果放数据盘；在tmux中执行，已完成回合可按原计划恢复。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import zipfile

import smolvla_training_workbench as wb

BEST_SHA = "19f73f07cc75a32500aaaeb6f059f30b11ce8cc5b3626eff84a33f4b8ca7492c"


def verify_bundle(directory):
    path = directory / 'bundle_manifest.json'
    if not path.is_file():
        return  # 仓库内开发运行不要求打包清单。
    manifest = json.loads(path.read_text())
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('部署包清单含非法路径')
        file = directory / name
        if not file.is_file() or wb.digest(file) != expected:
            raise RuntimeError(f'部署文件与包内SHA不符：{name}；修改后请重新打包')


def diagnostics(args):
    """只收集运行日志/统计；不打包权重、图像、视频或任何登录配置。"""
    args.workdir.mkdir(parents=True, exist_ok=True)
    output = args.workdir / ('diagnostics_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.zip')
    omitted, total = [], 0
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for label, root in [('logs',args.workdir/'logs'), ('results',args.output)]:
            for path in sorted(root.rglob('*')):
                if path.is_symlink() or not path.is_file() or path.suffix not in {'.log','.txt','.json','.csv','.jsonl'}:
                    continue
                name = str(Path(label) / path.relative_to(root))
                if total + path.stat().st_size > 64 * 2**20:
                    omitted.append(name);continue
                data = wb.redact(path.read_text(errors='replace')).encode()
                archive.writestr(name, data);total += len(data)
        archive.writestr('collection.json', json.dumps({'scope':'redacted logs/statistics only; no credentials/models/videos',
                                                       'omitted_budget':omitted}, ensure_ascii=False, indent=2))
    print('诊断日志包：', output)
    return output


def check_checkpoint(args):
    if not args.checkpoint or not args.checkpoint.is_file():
        raise FileNotFoundError('请先从Drive取回已训练的best_model.pt，并用--checkpoint指定；不需要44条训练特征')
    actual = wb.digest(args.checkpoint)
    if args.checkpoint_sha256 and actual != args.checkpoint_sha256:
        raise ValueError('预测器SHA与已选模型不同；请核对文件，使用其他权重时显式指定--checkpoint-sha256')
    return actual


def prepare(args, log):
    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError('请使用已创建的Python3.12 bootstrap环境运行本入口')
    root = args.workdir
    environment = root / 'environment.json'
    if environment.is_file():
        saved = json.loads(environment.read_text())
        if saved.get('profile') != 'gpu' or saved.get('lerobot_revision') != wb.LEROBOT_SHA:
            raise ValueError('已有环境版本不同；请另选workdir，不覆盖已有环境')
    else:
        # bootstrap只需Python/pip；直接调用现有独立Torch准备器，兼容未安装Torch的conda。
        options = argparse.Namespace(workdir=root, profile='gpu', skip_assets=True)
        wb.prepare_with_managed_torch(options, log)
    python = root / 'venv/bin/python'
    probe = ("import torch,torchvision;from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy;"
             "assert torch.cuda.is_available();assert torch.ones(2,device='cuda').sum().item()==2;"
             "torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device='cuda'),torch.tensor([1.],device='cuda'),.5);"
             "print('CUDA与torchvision算子通过：',torch.__version__,torch.cuda.get_device_name(0))")
    log.run([python, '-c', probe], env=wb.runtime_env(root))
    # 原prepare-sim会检查固定版本、真实reset/EGL与视频编码；可复用已完成环境。
    log.run([python, Path(wb.__file__), 'prepare-sim', '--workdir', root,
             '--log-root', log.directory / 'simulation'], env=wb.runtime_env(root))
    log.say('依赖及实际渲染检查已完成；没有训练或重新提取示范特征。')


def check_runtime(args, log):
    root = args.workdir
    for name in ['environment.json', 'simulation.json']:
        if not (root / name).is_file():
            raise RuntimeError('先执行prepare，缺少：' + name)
    if not json.loads((root / 'simulation.json').read_text()).get('ready'):
        raise RuntimeError('仿真准备未完成，请检查prepare日志')
    sha = check_checkpoint(args)
    code = ("from pathlib import Path;from corrector_online_runtime import load_predictor,decisions;"
            f"model,protocol=load_predictor(Path({str(args.checkpoint)!r}),Path({str(args.source_root)!r}),'cpu');"
            "decisions('student');assert len(protocol['records'])==44;"
            "assert protocol['prediction_seconds']==1.;"
            "print('真实44轨迹预测器检查通过：',sum(p.numel() for p in model.parameters()),'参数')")
    # 子进程cwd不一定是代码目录；显式插入受控脚本路径，不修改环境PYTHONPATH。
    code = f"import sys;sys.path.insert(0,{str(Path(__file__).resolve().parent)!r});" + code
    log.run([root / 'sim-venv/bin/python', '-c', code], env=wb.runtime_env(root))
    log.say('预测器SHA256：' + sha)


def evaluation_command(args, mode):
    code = Path(__file__).resolve().parent
    output = args.output / ('debug_ogg_001' if mode == 'smoke' else 'paired_001')
    command = [args.workdir / 'sim-venv/bin/python', code / 'corrector_closed_loop_eval.py',
               '--workdir', args.workdir, '--output', output, '--suite', args.suite,
               '--task-ids', *map(str, args.task_ids), '--corrector-checkpoint', args.checkpoint,
               '--source-root', args.source_root, '--control-implementation', 'student']
    if mode == 'smoke':
        # 只选第一个任务，不让强制事件进入正式计划。
        start = command.index('--task-ids') + 1
        end = command.index('--corrector-checkpoint')
        command[start:end] = [str(args.task_ids[0])]
        command += ['--methods', 'ogg', '--initial-state-ids', str(args.initial_state_ids[0]),
                    '--seeds', str(args.seeds[0]), '--max-steps', '60', '--debug-force-step', '24',
                    '--video-policy', 'all', '--audit-mode', 'all']
    else:
        command += ['--methods', *args.methods, '--initial-state-ids', *map(str,args.initial_state_ids),
                    '--seeds', *map(str,args.seeds), '--video-policy', args.video_policy,
                    '--audit-mode', 'failures', '--trace-mode', 'all']
    return command, output


def verify_smoke(args):
    _, root = evaluation_command(args, 'smoke')
    files = list((root / 'trials').glob('*.json'))
    records = [json.loads(p.read_text()) for p in files]
    if len(records) != 1 or records[0]['status'] != 'completed':
        raise RuntimeError('必须先成功完成单回合接线检查，再启动正式批次')
    record = records[0]
    result = record['result']
    if result.get('event_replans', 0) < 1 or result.get('guidance_calls', 0) < 1:
        raise RuntimeError('单回合没有实际执行中止与OGG梯度引导')
    directory = root / 'attempts' / record['case']['id'] / f"attempt_{record['attempt']:03d}"
    actual = json.loads((directory / 'actual_protocol.json').read_text())
    if actual['monitor_horizon_steps'] != 20:
        raise RuntimeError('1秒监测时间应为20个控制步')
    if record['case']['config']['corrector_sha256'] != check_checkpoint(args):
        raise RuntimeError('单回合检查与本次选择的权重不同，请使用新output重新检查')
    plan = json.loads((root / 'plan.json').read_text())
    for name, file in [('runtime_source_sha256','corrector_online_runtime.py'),
                       ('learner_core_sha256','corrector_online_practice.py')]:
        path = Path(__file__).resolve().parent / file
        if file == 'corrector_online_practice.py' and not path.is_file():
            path = Path(__file__).resolve().parents[1] / 'studies/policy_generalization/corrector/online_practice.py'
        if plan['protocol'][name] != wb.digest(path):
            raise RuntimeError('在线核心代码在单回合后改变；请用新output检查')
    return root


def evaluate(args, mode, log):
    command, root = evaluation_command(args, mode)
    env = wb.runtime_env(args.workdir)
    if mode == 'batch':
        verify_smoke(args)
        if args.offline_batch:
            # smoke已成功加载完整模型、分词器和场景；批次阶段不依赖本机代理长久在线。
            env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    log.run(command, env=env)
    if mode == 'smoke':
        verify_smoke(args)
        log.say('单回合接线通过；带强制事件的结果不计入正式成功率。')
    log.say((root / 'analysis/summary.json').read_text())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare','check','smoke','batch','night','summary','diagnostics'])
    p.add_argument('--workdir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path)
    p.add_argument('--source-root', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-sha256', default=BEST_SHA)
    p.add_argument('--suite', choices=['libero_object','libero_spatial','libero_goal','libero_10'], default='libero_object')
    p.add_argument('--task-ids', type=int, nargs='+', default=[0])
    p.add_argument('--initial-state-ids', type=int, nargs='+', default=[0,1,2,3,4])
    p.add_argument('--seeds', type=int, nargs='+', default=[1000,1001])
    p.add_argument('--methods', nargs='+', choices=['H50','H20','monitor','lvm','ogg'],
                   default=['H50','H20','monitor','lvm','ogg'])
    p.add_argument('--video-policy', choices=['all','failures','none'], default='failures')
    p.add_argument('--offline-batch', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--proxy-url', help='可选：只为本启动器及其子进程设置HTTP(S)代理，例如http://127.0.0.1:17890')
    return p


def main():
    args = parser().parse_args()
    if args.proxy_url:
        from urllib.parse import urlsplit
        proxy = urlsplit(args.proxy_url)
        if proxy.scheme not in {'http','https'} or not proxy.hostname:
            raise ValueError('--proxy-url应为HTTP(S)代理地址')
        os.environ.update(http_proxy=args.proxy_url, https_proxy=args.proxy_url,
                          HTTP_PROXY=args.proxy_url, HTTPS_PROXY=args.proxy_url,
                          no_proxy='localhost,127.0.0.1,::1', NO_PROXY='localhost,127.0.0.1,::1')
    args.workdir = args.workdir.absolute()
    args.output = (args.output or args.workdir / 'results/night_001').absolute()
    args.source_root = (args.source_root or args.workdir / 'vla-corrector-reference').absolute()
    args.checkpoint = (args.checkpoint or args.workdir / 'checkpoints/best_model.pt').absolute()
    verify_bundle(Path(__file__).resolve().parent)
    if args.mode == 'diagnostics':
        diagnostics(args)
        return
    if args.mode == 'summary':
        for p in sorted(args.output.glob('*/analysis/summary.json')):
            print(p);print(p.read_text())
        return
    if args.mode != 'prepare':
        check_checkpoint(args)
    args.workdir.mkdir(parents=True,exist_ok=True)
    log = wb.Log(args.workdir / 'logs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + args.mode))
    wb.write_json(log.directory / 'launch.json', {
        'mode':args.mode,'python':sys.executable,'workdir':str(args.workdir),'output':str(args.output),
        'checkpoint':str(args.checkpoint),'checkpoint_sha256':args.checkpoint_sha256,
        'methods':args.methods,'task_ids':args.task_ids,'initial_state_ids':args.initial_state_ids,'seeds':args.seeds,
        'proxy_configured':bool(os.environ.get('HTTPS_PROXY')),
        'scope':'original LIBERO adaptation; single-task development evaluation, not paper table reproduction'})
    try:
        if args.mode in {'prepare','night'}:
            prepare(args,log)
        if args.mode in {'check','smoke','batch','night'}:
            check_runtime(args,log)
        if args.mode in {'smoke','night'}:
            evaluate(args,'smoke',log)
        if args.mode in {'batch','night'}:
            evaluate(args,'batch',log)
    except BaseException as error:
        log.say(traceback.format_exc())
        wb.write_json(log.directory/'status.json',{'ok':False,'type':type(error).__name__,'error':wb.redact(str(error))})
        raise
    wb.write_json(log.directory/'status.json',{'ok':True,'mode':args.mode})


if __name__ == '__main__':
    main()
