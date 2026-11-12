"""CPU-only regression tests; real image measurements require remote CUDA."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from corrector_multilayer import ReadOnlyVisionTap, mean_pool_tokens, quantize_per_token, dequantize
from corrector_multilayer_pilot import compare_arrays, evaluate_pair, summarize_pairs, atomic_json


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Module()
        self.encoder.layers=nn.ModuleList(nn.Linear(4,4) for _ in range(12))
        self.post_layernorm=nn.LayerNorm(4)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model=FakeVision()
        self.connector=nn.Linear(4,4)


class FakeWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=FakeModel()
    def get_vlm_model(self):
        return self.model
    def embed_image(self,x):
        for block in self.model.vision_model.encoder.layers:
            x=block(x)
        x=self.model.vision_model.post_layernorm(x)
        return self.model.connector(x)


class TestPilot(unittest.TestCase):
    def test_hook_exact_outputs_cleanup(self):
        torch.manual_seed(2)
        m=FakeWrapper().eval().requires_grad_(False)
        x=torch.randn(1,4,4)
        with torch.inference_mode():
            expected=m.embed_image(x)
            with ReadOnlyVisionTap(m, (2,5,8)) as tap:
                with tap.capture("cam1") as captured:
                    got=m.embed_image(x)
                self.assertTrue(torch.equal(expected,got))
                self.assertTrue(torch.equal(got,captured["connector"]))
                self.assertEqual(set(captured),{"block_02","block_05","block_08","vision_postnorm","connector"})
            self.assertEqual(len(m.model.connector._forward_hooks),0)
            for i in (2,5,8):
                self.assertEqual(len(m.model.vision_model.encoder.layers[i]._forward_hooks),0)

    def test_constant_delta_preservation(self):
        a=np.ones((2,1024,6),np.float32)*.5
        b=a+.125
        metrics=evaluate_pair({"block_05":a},{"block_05":b},["block_05"])
        for grid in ("4","8"):
            self.assertLess(metrics["block_05"][grid]["pooled_vs_full_delta"]["relative_l2"],1e-6)
            self.assertLess(metrics["block_05"][grid]["fp16_delta"]["relative_l2"],1e-6)
            self.assertLess(metrics["block_05"][grid]["int8_delta"]["relative_l2"],.01)

    def test_local_pixel_change_loss(self):
        a=np.zeros((2,1024,8),np.float32)
        b=a.copy()
        b[:,0,0]=1
        metrics=evaluate_pair({"block_02":a},{"block_02":b},["block_02"])
        err4=metrics["block_02"]["4"]["pooled_vs_full_delta"]["relative_l2"]
        err8=metrics["block_02"]["8"]["pooled_vs_full_delta"]["relative_l2"]
        self.assertGreater(err4,err8)
        self.assertGreater(err8,0)

    def test_dynamic_16_and_32_grids(self):
        a=np.zeros((2,1024,8),np.float32)
        b=a.copy()
        b[:,1,0]=1
        metrics=evaluate_pair({"block_08":a},{"block_08":b},["block_08"],grids=(8,16,32))
        self.assertEqual(set(metrics["block_08"]),{"8","16","32"})
        self.assertLess(metrics["block_08"]["32"]["pooled_vs_full_delta"]["relative_l2"],1e-7)
        self.assertGreater(metrics["block_08"]["8"]["pooled_vs_full_delta"]["relative_l2"],metrics["block_08"]["16"]["pooled_vs_full_delta"]["relative_l2"])
        s=summarize_pairs([{"metrics":metrics}])
        self.assertEqual(set(s["block_08"]),{"8","16","32"})

    def test_quantized_deep_cache_exactness_protocol(self):
        z=np.random.default_rng(1337).normal(size=(128,960)).astype(np.float32)
        cache_q,cache_scale=quantize_per_token(z)
        repeat_q,repeat_scale=quantize_per_token(z.copy())
        self.assertTrue(np.array_equal(cache_q,repeat_q))
        self.assertTrue(np.array_equal(cache_scale,repeat_scale))

    def test_token_quantization(self):
        x=np.random.default_rng(42).standard_normal((2,8,8,768)).astype(np.float32)
        q,sc=quantize_per_token(x)
        self.assertEqual(q.shape,x.shape)
        self.assertEqual(sc.shape,x.shape[:-1])
        self.assertLess(compare_arrays(x,dequantize(q,sc))["relative_l2"],.03)

    def test_no_overwrite_report(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"report.json"
            atomic_json(p,{"ok":True})
            self.assertTrue(json.loads(p.read_text())["ok"])
            with self.assertRaises(FileExistsError):
                atomic_json(p,{"ok":False})

    def test_pair_summary_counts(self):
        a=np.zeros((2,1024,3),np.float32)
        b=np.ones_like(a)
        m=evaluate_pair({"block_02":a},{"block_02":b},["block_02"])
        s=summarize_pairs([{"metrics":m},{"metrics":m}])
        self.assertEqual(s["block_02"]["8"]["fp16_delta"]["pairs"],2)


if __name__=="__main__":
    unittest.main()
