"""只观察首次生成：包装实例方法，不修改 LeRobot 安装包，不另做一次推理。"""
import inspect
import json
from functools import wraps
from pathlib import Path

import torch

if __package__:
    from .tensor_summary import summarize_tensor
else:
    from tensor_summary import summarize_tensor


class PolicyTrace:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "trace.json").exists():
            raise FileExistsError("Trace already exists; choose a new TRACE_DIR")
        self.report = {"complete": False, "metadata": {}, "tensors": {}, "queue_steps": []}
        self.active = False
        self.policy = None
        self._undo = []
        self._select_calls = 0
        self._chunk_calls = 0

    def record(self, name, value):
        """记录结构中实际存在的 Tensor；不把整个张量打印到终端。"""
        if isinstance(value, torch.Tensor):
            self.report["tensors"][name] = summarize_tensor(name, value, verbose=False)
        elif isinstance(value, dict):
            for key, item in value.items():
                self.record(f"{name}.{getattr(key, 'value', key)}", item)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self.record(f"{name}.{index}", item)

    def flush(self):
        (self.output_dir / "trace.json").write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _replace(self, owner, name, replacement):
        """恢复时区分原有实例属性与类上的方法，避免留下方法覆盖。"""
        had_own_attribute = name in vars(owner)
        previous = vars(owner).get(name)
        setattr(owner, name, replacement)

        def undo():
            if had_own_attribute:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)

        self._undo.append(undo)

    def _observe_output(self, owner, method_name, label, *, per_call=False, save_as=None):
        original = getattr(owner, method_name)
        calls = 0

        @wraps(original)
        def wrapper(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            if self.active and (per_call or calls == 0):
                name = f"{label}.{calls}" if per_call else label
                self.record(name, result)
                if save_as is not None:
                    torch.save(result.detach().cpu().clone(), self.output_dir / save_as)
                calls += 1
            return result  # 原对象、原数值，不重新计算。

        self._replace(owner, method_name, wrapper)

    def _queue_length(self):
        queue = getattr(self.policy, "_queues", {}).get("action")
        return len(queue) if queue is not None else None

    def _ready(self):
        return not self.report["complete"] and self._queue_length() in (None, 0)

    def install(self, policy):
        """由官方 make_policy() 返回真实实例后调用。"""
        if self.policy is not None:
            raise RuntimeError("Trace supports one policy and one sequential evaluation")
        required = ["prepare_images", "prepare_state", "_get_action_chunk", "select_action"]
        for name in required:
            if not callable(getattr(policy, name, None)):
                raise TypeError(f"Expected SmolVLA method: {name}")
        model = getattr(policy, "model", None)
        for name in ["sample_actions", "sample_noise", "embed_prefix", "embed_suffix"]:
            if not callable(getattr(model, name, None)):
                raise TypeError(f"Expected VLAFlowMatching method: {name}")

        self.policy = policy
        cfg = policy.config
        fields = ["chunk_size", "n_action_steps", "num_steps", "max_action_dim",
                  "max_state_dim", "resize_imgs_with_padding", "device", "use_amp",
                  "pretrained_path", "vlm_model_name", "tokenizer_max_length",
                  "pad_language_to", "add_image_special_tokens"]
        self.report["metadata"].update({
            "policy_class": type(policy).__name__,
            "policy_source": inspect.getfile(type(policy)),
            "config": {key: str(value) if isinstance(value, (Path, torch.device)) else value
                       for key in fields if (value := getattr(cfg, key, None)) is not None},
        })
        parameter = next(policy.parameters(), None)
        if parameter is not None:
            self.report["metadata"]["first_parameter"] = {
                "dtype": str(parameter.dtype), "device": str(parameter.device)}

        self._observe_output(policy, "prepare_images", "prepare_images.output")
        self._observe_output(policy, "prepare_state", "prepare_state.output")
        self._observe_output(model, "sample_noise", "initial_noise", save_as="initial_noise.pt")
        self._observe_output(model, "sample_actions", "sample_actions.output")
        self._observe_output(model, "embed_prefix", "embed_prefix.output")
        self._observe_output(model, "embed_suffix", "embed_suffix.first_output")
        vlm = getattr(model, "vlm_with_expert", None)
        if callable(getattr(vlm, "embed_image", None)):
            self._observe_output(vlm, "embed_image", "visual_tokens", per_call=True)

        original_chunk = policy._get_action_chunk

        @wraps(original_chunk)
        def chunk_wrapper(*args, **kwargs):
            self._chunk_calls += 1
            result = original_chunk(*args, **kwargs)
            if self.active:
                self.record("action_chunk", result)
                torch.save(result.detach().cpu().clone(), self.output_dir / "action_chunk.pt")
            return result

        self._replace(policy, "_get_action_chunk", chunk_wrapper)
        original_select = policy.select_action

        @wraps(original_select)
        def select_wrapper(*args, **kwargs):
            self._select_calls += 1
            before, chunks_before = self._queue_length(), self._chunk_calls
            capture = self._ready()
            self.active = capture
            try:
                if capture:
                    bound = inspect.signature(original_select).bind(*args, **kwargs)
                    batch = bound.arguments["batch"]
                    self.record("policy_input", batch)
                    snapshot = {key: value.detach().cpu().clone() for key, value in batch.items()
                                if isinstance(value, torch.Tensor)}
                    torch.save(snapshot, self.output_dir / "policy_input.pt")
                    self._save_input_images(snapshot)
                result = original_select(*args, **kwargs)
                if capture:
                    self.record("select_action.output", result)
                    torch.save(result.detach().cpu().clone(), self.output_dir / "selected_action.pt")
                    self.report["complete"] = "action_chunk" in self.report["tensors"]
                    if not self.report["complete"]:
                        raise RuntimeError("select_action did not generate the expected action chunk")
                if self._select_calls <= 3:
                    self.report["queue_steps"].append({
                        "select_call": self._select_calls, "before": before,
                        "after": self._queue_length(),
                        "generated_chunks": self._chunk_calls - chunks_before,
                    })
                if capture or self._select_calls <= 3:
                    self.flush()
                return result
            finally:
                self.active = False

        self._replace(policy, "select_action", select_wrapper)
        self.flush()

    def observe_processors(self, preprocessor, postprocessor):
        """使用官方 Pipeline 的 hook；返回原 Pipeline，不替换它的类型。"""
        def before_pre(index, transition):
            if index == 0 and self._ready() and not self.report.get("preprocessor_seen"):
                self.record("preprocessor.input", transition)
                self.report["preprocessor_seen"] = True

        def after_post(index, transition):
            if (index == len(postprocessor.steps)-1 and self.report["complete"]
                    and not self.report.get("postprocessor_seen")):
                self.record("postprocessor.output", transition)
                self.report["postprocessor_seen"] = True
                self.flush()

        for processor, attribute, hook in [
            (preprocessor, "before_step_hooks", before_pre),
            (postprocessor, "after_step_hooks", after_post),
        ]:
            hooks = getattr(processor, attribute, None)
            if hooks is not None:
                hooks.append(hook)
                self._undo.append(lambda hooks=hooks, hook=hook: hooks.remove(hook))

    def _save_input_images(self, snapshot):
        from PIL import Image

        for key, image in snapshot.items():
            if not key.startswith("observation.images."):
                continue
            if image.ndim == 5:
                image = image[:, -1]
            if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] == 0:
                continue
            image = image[0].float()
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                continue  # 不猜测其他数值空间，也不静默裁剪成看似正确的图片。
            pixels = image.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            filename = key.replace(".", "_").replace("/", "_") + "_batch0.png"
            Image.fromarray(pixels).save(self.output_dir / filename)

    def close(self):
        while self._undo:
            self._undo.pop()()
        self.flush()
