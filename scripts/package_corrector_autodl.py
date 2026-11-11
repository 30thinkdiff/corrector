"""将当前工作区实际代码打包；不依赖尚未push的GitHub版本，不包含凭据/模型/实验输出。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

from sync_corrector_notebook import ROOT, SOURCES


def build(output, root=ROOT, training_manifest=None):
    files = {}
    for key, relative in SOURCES.items():
        name = 'corrector_online_practice.py' if key == 'ONLINE_PRACTICE_SOURCE' else Path(relative).name
        text = (root / relative).read_text()
        if name.endswith('.json'):json.loads(text)
        else:ast.parse(text)
        if name in files:raise ValueError('部署文件名冲突：' + name)
        files[name] = text.encode()
    runner = 'corrector_autodl.py'
    files[runner] = (root / 'scripts' / runner).read_bytes()
    ast.parse(files[runner])
    files['START_HERE.txt'] = '''AutoDL部署包（没有任何账号、令牌或SSH配置）
1. 解压到/root/autodl-tmp/corrector/runtime；Python入口使用已有bootstrap环境。
2. prepare只装依赖和检查真实渲染，不需要权重，不训练、不提取特征。
   /root/autodl-tmp/corrector/bootstrap/bin/python corrector_autodl.py prepare --proxy-url http://127.0.0.1:17890
3. 从Drive的corrector_stream_task24_001/predictor/paper_width_001下载best_model.pt，
   上传为runtime/checkpoints/best_model.pt；SHA256应为
   19f73f07cc75a32500aaaeb6f059f30b11ce8cc5b3626eff84a33f4b8ca7492c
4. 在tmux中执行：
   /root/autodl-tmp/corrector/bootstrap/bin/python corrector_autodl.py night --proxy-url http://127.0.0.1:17890
   默认原始LIBERO object/task0，5真实初始状态×2种子×5方法=50正式回合。
   之前另有1个强制事件单回合及至多10个小规模自然事件回合，分别记录，不计入长批次正式成功率。
   当前建议先小规模：在night命令后加
   --initial-state-ids 0 1 --seeds 1000 --output results/pilot_001
   得到5方法×2初始化×1种子=10回合；检查视频/触发/耗时后再用新output扩大。
5. stdout和所有阶段完整日志在runtime/logs；结果、视频、分数在runtime/results/night_001。
   正式批次成功前自动检查smoke；默认HF离线，失败视频留存，成功率按真实完成回合统计。
6. 中断后，原机器/原代码/原参数重跑同一night命令；已完成回合验视频SHA后跳过。
   改方法/种子/初始状态/保留规则请用新--output；不要编辑已有冻结计划。
7. 只看结果不需要GPU：python corrector_autodl.py summary
8. 环境/实验出错时收集小日志包：python corrector_autodl.py diagnostics
   diagnostics_*.zip不含权重/视频/账号目录，可取回本地进行定位。
9. 当前prepare仍在运行时，将新版包解压到另一代码目录longrun-code，不能覆盖运行中的旧代码。
   用新目录的入口指定--workdir /root/autodl-tmp/corrector/runtime，复用旧安装与权重。
   night加--wait-ready-minutes 240会只等待原prepare成功结束，不并行安装；失败会停止等待。
   --initial-state-count 50 --seed-count 20得到5000个有限自然事件回合；
   --max-wall-hours 10为每次启动正式批次的墙钟预算，回合完整结束后暂停，可能略超预算。
   安装/等待/单回合检查不计入该10小时，脚本不会自动关机。
   这里只在1个任务上重复真实初始化与随机种子，不表示完成多任务或视角泛化评测。
10. 暂停：touch /root/autodl-tmp/corrector/runtime/results/overnight_001/STOP
    （output为overnight_001时）；当前完整回合结束后停止。
    续跑前删除这个STOP文件，再用batch模式和原output/原计划参数继续；不必重复prepare。
    完成回合验SHA跳过；被强制中断的当前回合在新attempt目录重做。
    修改时间预算允许续跑；改变方法/初始状态/种子/代码须新output。
11. plan模式只预览有限队列，不需要GPU/模型；execution.json记录当前暂停原因/完成数，
    analysis/overall.csv包含任务成功率、策略/监控/回合墙钟耗时及Torch分配显存峰值均值。
    磁盘剩余不足5GiB时默认暂停，不会删除原始结果。计划提前完成则退出，不无意义循环。
联网下载阶段依赖SSH反向代理时，保持本地隧道开启；tmux只保证任务不因SSH退出而结束，
不会让本地代理永久在线。night中通过smoke后，默认批次阶段不再请求HF网络。
如果127.0.0.1:17890报Connection refused，先在本地重建ssh -R反向代理并保持开启，
远端curl --proxy返回200后重跑prepare；不要删除半成品venv或误判成pip版本不存在。
本包是准备好的入口，尚未证明AutoDL上的真实完整实验已通过。
'''.encode()
    if training_manifest is not None:
        selected = json.loads(Path(training_manifest).read_text())
        if (selected.get('dataset') != 'HuggingFaceVLA/libero'
                or selected.get('revision') != '86958911c0f959db2bbbdb107eb3e17c5f9c798e'
                or selected.get('selection') != {'task_ids':list(range(20,30)), 'limit':0, 'seed':42}
                or len(selected['records']) != 454 or selected['frames'] != 66984):
            raise ValueError('训练包需要已核对的Object10完整454轨迹清单')
        files['object10_manifest.json'] = Path(training_manifest).read_bytes()
        files['corrector_prefetch.py'] = (root / 'scripts/corrector_prefetch.py').read_bytes()
        ast.parse(files['corrector_prefetch.py'])
        files['START_HERE.txt'] = '''本包转向预测器训练，不自动启动5000回合评测。
先在远端touch runtime/results/overnight_001/STOP，等待paired/execution.json为paused、评测进程退出。
解压本包到新目录/root/autodl-tmp/corrector/training-code，不覆盖原longrun-code或旧runtime。
已有环境无需再prepare。完整命令见START_TRAINING.txt；本包没有模型、凭据或数据图像。
'''.encode()
        files['START_TRAINING.txt'] = '''# 逐节复制执行，不要把整份文件当脚本一口气运行。
# 以下在远程终端执行；先确认旧闭环评测已暂停，不并行抢占同一GPU。
# 先cd到本包实际解压目录（例如training-code或training-code-netfix），不要切回旧代码。
# 环境与HF缓存继续复用旧runtime。
# 以下修复命令使用training-code-netfix的绝对路径，不依赖终端当前目录。
export HF_HOME=/root/autodl-tmp/corrector/runtime/hf
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
# 下列提取命令显式使用127.0.0.1:17890；先恢复隧道并验证。
# curl --proxy只对curl生效；这里通过--proxy-url配置Python本次进程。
# 不复制特征到local-cache，直接使用数据盘归档，以免双倍占空间。

# 1. 冻结清单：只发布元信息，不下载图像、不加载GPU模型。
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/training-code-netfix/corrector_stream_workbench.py plan \\
 --archive /root/autodl-tmp/corrector/object10 \\
 --workdir /root/autodl-tmp/corrector/object10-work \\
 --source-root /root/autodl-tmp/corrector/runtime/vla-corrector-reference \\
 --task-ids 20 21 22 23 24 25 26 27 28 29 --limit 0 --feature-budget-gib 12 \\
 --manifest-template /root/autodl-tmp/corrector/training-code-netfix/object10_manifest.json

# 2. 先提取3条真实轨迹测速（含train和validation），输出保存到日志。
# 正常完成后查object10/stream_report.json的downloads/extraction_seconds/feature_bytes。
/root/autodl-tmp/corrector/runtime/venv/bin/python -u /root/autodl-tmp/corrector/training-code-netfix/corrector_stream_workbench.py run \\
 --archive /root/autodl-tmp/corrector/object10 \\
 --workdir /root/autodl-tmp/corrector/object10-work \\
 --source-root /root/autodl-tmp/corrector/runtime/vla-corrector-reference \\
 --proxy-url http://127.0.0.1:17890 \\
 --max-new-episodes 3 --minutes 30 --min-free-gib 5 \\
 >> /root/autodl-tmp/corrector/object10-probe-netfix.log 2>&1

# 3. 确认测速/剩余空间后，在screen内执行下列完整提取；不要求一次在线完成。
# 重跑同命令会验SHA跳过已完成轨迹。touch object10/STOP可在完整轨迹边界暂停。
/root/autodl-tmp/corrector/runtime/venv/bin/python -u /root/autodl-tmp/corrector/training-code-netfix/corrector_stream_workbench.py run \\
 --archive /root/autodl-tmp/corrector/object10 \\
 --workdir /root/autodl-tmp/corrector/object10-work \\
 --source-root /root/autodl-tmp/corrector/runtime/vla-corrector-reference \\
 --proxy-url http://127.0.0.1:17890 \\
 --max-new-episodes 0 --minutes 0 --min-free-gib 5 \\
 >> /root/autodl-tmp/corrector/object10-extract.log 2>&1

# 4. 检查全部454条完成；--manifest拒绝只对半成品缓存启动全量训练。
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/training-code-netfix/corrector_train.py inspect \\
 --features /root/autodl-tmp/corrector/object10/features \\
 --manifest /root/autodl-tmp/corrector/object10/manifest.json \\
 --source-root /root/autodl-tmp/corrector/runtime/vla-corrector-reference \\
 --output /root/autodl-tmp/corrector/object10/predictor_001 \\
 --device cuda --profile paper-width --split-mode episode \\
 --batch-size 32 --accumulate 16 --loader-workers 2 --pin-memory \\
 --cpu-threads 4 --epochs 30 --snapshot-every 5

# 5. 把上条命令中的inspect改成fit，再加--stop-after-epoch 1，先完成一轮真实测速。
# 批32/累积16为有效批512；不是直接把512份token激活同时装进24GB显存。
# 查看train.log、history.json中的seconds和result.json的peak_cuda_gib。
# 保持其他参数不变，去掉--stop-after-epoch 1并加--resume，继续原计划。
# 每轮验证；latest保存模型/AdamW/调度器/随机状态；best在验证改善时发布。
# 中断的半个epoch会重做；改变数据、batch、源码或协议需新output，不冒充精确续训。
# 若32仍OOM，使用新的训练output尝试16/累积32，不能修改同一已启动实验的协议。
# touch predictor_001/STOP会在完整epoch边界暂停；续跑前删除该STOP文件。
# 当前364训练/90验证episode，没有独立test集；不是论文MetaWorld结果的等价复现。
# 训练结束先做离线预测报告，只选best开展小批次闭环；不要对每份快照运行5000回合。
'''.encode()
        files['START_TRAINING.txt'] = ('''# 可选并发预下载：完成下面第1步plan后，可先单独运行这一行。
# 默认先3个新分片测速；确认完成/吞吐后改--max-new-files 0。不要与run提取同时运行。
# 该进程只下载，复用source缓存，不训练。下载结束后再进行第3步特征提取。
HF_HOME=/root/autodl-tmp/corrector/runtime/hf nohup /root/autodl-tmp/corrector/runtime/venv/bin/python -u /root/autodl-tmp/corrector/training-code-netfix/corrector_prefetch.py --manifest /root/autodl-tmp/corrector/object10/manifest.json --local-dir /root/autodl-tmp/corrector/object10-work/source --proxy-url http://127.0.0.1:17890 --workers 3 --max-new-files 3 --min-free-gib 5 >> /root/autodl-tmp/corrector/object10-prefetch.log 2>&1 < /dev/null &
# 下载时保持本地代理在线。当前正在运行的任务无需重新部署此包。

'''.encode() + files['START_TRAINING.txt'])
    manifest = {'format':1,'scope':'current source snapshot, no model or credentials',
                'files':{name:hashlib.sha256(content).hexdigest() for name,content in files.items()}}
    files['bundle_manifest.json'] = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    output = Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+'.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():z.writestr(name,content)
    temporary.replace(output)
    print(output.resolve(),output.stat().st_size,'bytes')
    print('SHA256:',hashlib.sha256(output.read_bytes()).hexdigest())
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/autodl_setup/corrector_autodl.zip')
    p.add_argument('--training-manifest',type=Path,help='加入已核对的Object10清单与分阶段训练命令')
    args=p.parse_args()
    build(args.output, training_manifest=args.training_manifest)
