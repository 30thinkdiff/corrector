"""复用 lerobot_eval.main()；只增加 --trace-dir 和 --check-env。"""
import argparse
import importlib.util
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import patch

import torch

if __package__:
    from .policy_trace import PolicyTrace
else:
    from policy_trace import PolicyTrace


def environment_report():
    packages = {}
    for name in ["lerobot", "torch", "transformers", "tokenizers", "mujoco"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    spec = importlib.util.find_spec("lerobot")
    source = spec.origin if spec is not None else None
    revision = None
    if source:
        for parent in Path(source).resolve().parents:
            if (parent / ".git").exists():
                result = subprocess.run(["git", "-C", str(parent), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, check=False)
                revision = result.stdout.strip() if result.returncode == 0 else None
                break
    return {
        "python": sys.executable, "packages": packages, "lerobot_source": source,
        "lerobot_git_revision": revision, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_with_trace(eval_module, trace, arguments):
    """短生命周期包装工厂函数；任何异常都会恢复原函数和实例方法。"""
    make_policy = eval_module.make_policy
    make_processors = eval_module.make_pre_post_processors

    def make_traced_policy(*args, **kwargs):
        policy = make_policy(*args, **kwargs)
        trace.install(policy)
        print("[Trace] Actual policy config:", trace.report["metadata"]["config"], flush=True)
        return policy

    def make_traced_processors(*args, **kwargs):
        pre, post = make_processors(*args, **kwargs)
        trace.observe_processors(pre, post)
        return pre, post

    try:
        with patch.object(eval_module, "make_policy", make_traced_policy), \
             patch.object(eval_module, "make_pre_post_processors", make_traced_processors), \
             patch.object(sys, "argv", [sys.argv[0], *arguments]):
            eval_module.main()
        if not trace.report["complete"]:
            raise RuntimeError("No real action chunk captured; check the evaluation log")
    except BaseException as error:
        trace.report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trace-dir", default="outputs/smolvla_trace")
    parser.add_argument("--check-env", action="store_true")
    options, eval_arguments = parser.parse_known_args()
    environment = environment_report()
    print(json.dumps(environment, indent=2, ensure_ascii=False), flush=True)
    if options.check_env:
        return  # 本地无 GPU、无仿真依赖时也能运行到这里。

    # 延迟导入：--check-env 不需要 transformers、LIBERO 或完整 checkpoint。
    from lerobot.scripts import lerobot_eval

    trace = PolicyTrace(options.trace_dir)
    trace.report["environment"] = environment
    trace.report["eval_arguments"] = eval_arguments
    run_with_trace(lerobot_eval, trace, eval_arguments)
    print(f"[Trace] Saved: {trace.output_dir / 'trace.json'}", flush=True)


if __name__ == "__main__":
    main()
