"""把本地脚本快照写入Corrector Notebook；不执行单元格、不清除本地实验输出。

本地.py修改后先运行本命令，再在远程内核执行Notebook首个同步格。
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "MODEL_API_SOURCE": "scripts/corrector_models.py",
    "WORKBENCH_SOURCE": "scripts/smolvla_training_workbench.py",
    "DATA_SOURCE": "scripts/corrector_data_workbench.py",
    "SIM_SOURCE": "scripts/smolvla_libero_tools.py",
    "BASELINE_SOURCE": "scripts/smolvla_baseline_benchmark.py",
    "BATCH_SOURCE": "studies/policy_generalization/corrector/batch_evaluation.py",
    "INVENTORY_SOURCE": "scripts/inspect_robot_datasets.py",
    "TRAIN_SOURCE": "scripts/corrector_train.py",
    "PREDICTION_SOURCE": "scripts/corrector_prediction_report.py",
    "STREAM_SOURCE": "scripts/corrector_stream_workbench.py",
    "STREAM_MANIFEST_SOURCE": "studies/policy_generalization/corrector/task24_manifest.json",
    "ONLINE_RUNTIME_SOURCE": "scripts/corrector_online_runtime.py",
    "ONLINE_PRACTICE_SOURCE": "studies/policy_generalization/corrector/online_practice.py",
    "CLOSED_LOOP_SOURCE": "scripts/corrector_closed_loop_eval.py",
}

RECOVERY_SOURCE = '''# 仅在旧环境格长时间无进展时使用；全新T4运行时跳过本格。
# 先停止旧单元格。默认只检查；确认要终止残留准备进程时改为True。
STOP_STALLED_PREPARE = False
from pathlib import Path
import os, signal, subprocess, time
recovery_work = Path(globals().get('WORKDIR', '/content/smolvla-training'))
processes = {}
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        argv = [x.decode() for x in (entry / 'cmdline').read_bytes().split(b'\\0') if x]
        parent = int(next(x for x in (entry / 'status').read_text().splitlines()
                          if x.startswith('PPid:')).split()[1])
        processes[int(entry.name)] = (parent, argv)
    except (OSError, ValueError, StopIteration, UnicodeError):
        continue
roots = set()
for pid, (_, argv) in processes.items():
    if (str(recovery_work / 'smolvla_training_workbench.py') in argv and 'prepare' in argv
            and '--workdir' in argv and argv[argv.index('--workdir') + 1:][:1] == [str(recovery_work)]):
        roots.add(pid)
targets = set(roots)
while True:
    extra = {pid for pid, (parent, _) in processes.items() if parent in targets} - targets
    if not extra:
        break
    targets.update(extra)
print('本项目残留prepare及其子进程：', sorted(targets))
if targets and not STOP_STALLED_PREPARE:
    raise RuntimeError('先停止旧格；若仍残留，将STOP_STALLED_PREPARE改为True后重跑本恢复格。不会终止训练/提取。')
for pid in targets:
    try:
        current = [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\\0') if x]
        if current == processes[pid][1]:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, FileNotFoundError):
        pass
if targets:
    time.sleep(1)
    for pid in targets:
        status = Path(f'/proc/{pid}/status')
        if status.exists() and 'State:\\tZ' not in status.read_text():
            raise RuntimeError(f'PID {pid}尚未退出，保留现场；不要并行重跑准备格。')
managed_python = recovery_work / 'torch-runtime-2.11-cu128/bin/python'
if managed_python.is_file():
    subprocess.run([str(managed_python), '-c',
                    "import torch; print('已安装Torch：',torch.__version__); print('CUDA：',torch.cuda.is_available()); print(torch.ones(1,device='cuda').sum().item())"],
                   check=True, timeout=60)
venv_python = recovery_work / 'venv/bin/python'
if venv_python.is_file():
    try:
        subprocess.run([str(venv_python), '-c', 'import sys; print(sys.prefix)'],
                       check=True, timeout=20)
    except (subprocess.SubprocessError, OSError):
        backup = recovery_work / ('venv_incomplete_' + str(time.time_ns()))
        (recovery_work / 'venv').rename(backup)
        print('仅备份启动失败的工作venv：', backup)
print('检查完成。现在执行下一环境准备格；模型、数据、已装Torch和pip缓存均未删除。')
'''


def repair_setup_cells(value):
    """恢复操作格，同时迁移旧模板的Drive逐行准备日志；保留所有运行输出。"""
    cells = value['cells']
    setup = next(c for c in cells if c.get('id') == 'corrector-data-1')
    source = ''.join(setup['source'])
    # 单元测试的小型源码快照模板没有环境启动部分。
    if 'log = wb.Log(ARCHIVE / \'setup_logs\')' not in source and 'LOCAL_SETUP_LOGS' not in source:
        return
    source = source.replace("log = wb.Log(ARCHIVE / 'setup_logs')",
                            "LOCAL_SETUP_LOGS = WORKDIR / 'setup_logs_local'\nlog = wb.Log(LOCAL_SETUP_LOGS)")
    source = source.replace("'--log-root', ARCHIVE / 'setup_logs'", "'--log-root', LOCAL_SETUP_LOGS")
    if 'SETUP_LOG_ZIP =' not in source:
        source += ("\n# 准备期间全部写临时盘；环境就绪后仅归档一次，避免逐行远程写入。\n"
                   "print('环境已就绪；现在压缩并备份准备日志。')\n"
                   "import time\n"
                   "SETUP_LOG_ZIP = shutil.make_archive(str(WORKDIR / ('setup_logs_' + str(time.time_ns()))), 'zip', LOCAL_SETUP_LOGS)\n"
                   "SETUP_ARCHIVE = ARCHIVE / 'setup_logs'\n"
                   "SETUP_ARCHIVE.mkdir(parents=True, exist_ok=True)\n"
                   "shutil.copy2(SETUP_LOG_ZIP, SETUP_ARCHIVE / Path(SETUP_LOG_ZIP).name)\n"
                   "print('准备日志已备份：', SETUP_ARCHIVE / Path(SETUP_LOG_ZIP).name)\n")
    source = source.replace(
        "SETUP_LOG_ZIP = shutil.make_archive(str(log.path.parent / ('setup_' + log.path.name)), 'zip', LOCAL_SETUP_LOGS)",
        "import time\nSETUP_LOG_ZIP = shutil.make_archive(str(WORKDIR / ('setup_logs_' + str(time.time_ns()))), 'zip', LOCAL_SETUP_LOGS)")
    setup['source'] = source.splitlines(keepends=True)
    recovery = next((c for c in cells if c.get('id') == 'setup-recovery'), None)
    if recovery is None:
        recovery = dict(id='setup-recovery', cell_type='code', metadata={},
                        execution_count=None, outputs=[], source=[])
    else:
        cells.remove(recovery)
    recovery['source'] = RECOVERY_SOURCE.splitlines(keepends=True)
    index = cells.index(setup)
    cells.insert(index, recovery)
    if not any(c.get('id') == 'setup-start-guide' for c in cells):
        cells.insert(index, dict(id='setup-start-guide', cell_type='markdown', metadata={}, source=[
            '## 从这里启动：新T4执行环境格；旧运行时卡住才用恢复格\n',
            '\n',
            '新连接不一定代表运行时已重置。下面的恢复格默认只检查残留准备进程；全新运行时直接跳过它。环境准备格以 `CLOSED_LOOP_SOURCE =` 开头（前面是随格同步的脚本），不需要先执行顶部的Torch查询格。\n',
            '\n',
            '准备日志先写 `/content/smolvla-training/setup_logs_local`，结束后压缩备份云盘。出现“环境已就绪”说明安装检查完成；之后才是日志归档。完成后执行“只更新代码”格，再进入“今日主线”1～4，暂不运行旧pilot与独立基线。\n']))
    if not any(c.get('id') == 'stream-code-refresh' for c in cells):
        cells.insert(cells.index(setup) + 1, dict(id='stream-code-refresh', cell_type='code',
                     metadata={}, execution_count=None, outputs=[], source=[]))


def sync(notebook, root=ROOT):
    value = json.loads(notebook.read_text())
    repair_setup_cells(value)
    cell = next(c for c in value["cells"] if c.get("id") == "corrector-data-1")
    source = "".join(cell["source"])
    replacements, found = [], set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in SOURCES:
                text = (root / SOURCES[name]).read_text()
                if SOURCES[name].endswith('.json'): json.loads(text)
                else: ast.parse(text)
                replacements.append((node.lineno - 1, node.end_lineno, f"{name} = {text!r}\n"))
                found.add(name)
    if found != set(SOURCES):
        raise ValueError(f"Notebook模板缺少源码字段：{set(SOURCES) - found}")
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    cell["source"] = "".join(lines).splitlines(keepends=True)
    refresh=next((c for c in value['cells'] if c.get('id')=='stream-code-refresh'),None)
    if refresh is not None:
        names={'ONLINE_PRACTICE_SOURCE':'corrector_online_practice.py'}
        bundle={names.get(key,Path(path).name):(root/path).read_text() for key,path in SOURCES.items()}
        refresh['source']=(
            "# 环境准备完成后运行：只更新代码，不重新安装。提取/训练子进程运行时不要执行此格。\n"
            f"REFRESH_FILES = {bundle!r}\n"
            "import hashlib\n"
            "assert WORKDIR.is_dir(), '先完成环境准备格'\n"
            "for filename, content in REFRESH_FILES.items():\n"
            "    target = WORKDIR / filename\n"
            "    target.write_text(content)\n"
            "    print(filename, hashlib.sha256(target.read_bytes()).hexdigest()[:16])\n"
            "print('代码同步完成；环境未重装。现在进入今日主线1～4。')\n"
        ).splitlines(keepends=True)
    temporary = notebook.with_name(notebook.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    temporary.replace(notebook)
    for name, path in SOURCES.items():
        print(path, hashlib.sha256((root / path).read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "scripts/corrector_data_prepare.ipynb")
    sync(parser.parse_args().notebook)
