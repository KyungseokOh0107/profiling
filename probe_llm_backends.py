"""Numerical/backend diagnostics; never mix these probes with latency trials."""
import json
import argparse
import time
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2 as qwen
from benchmark_llm_native import MODEL, REVISION, input_cases, save_json

OUT = Path("results/overnight/backend_probe")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    if (output / "probe.json").exists():
        raise FileExistsError("Choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cases = input_cases(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
              torch_dtype=torch.float16, attn_implementation="eager").eval().cuda()
    ids = torch.tensor([cases["A"]], device="cuda")
    report = {"torch": torch.__version__, "flash_compiled": torch.backends.cuda.is_flash_attention_available(),
              "bf16_supported": torch.cuda.is_bf16_supported(), "model": MODEL, "revision": REVISION}
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
        report["unmodified_eager_fp16_finite"] = bool(logits.isfinite().all().item())
        original = qwen.eager_attention_forward
        diagnostic = []

        def inspect(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            keys = qwen.repeat_kv(key, module.num_key_value_groups)
            qk16 = torch.matmul(query, keys.transpose(2, 3))
            qk32 = torch.matmul(query.float(), keys.float().transpose(2, 3))
            finite32 = bool(qk32.isfinite().all().item())
            diagnostic.append({"layer": module.layer_idx,
                "query_finite": bool(query.isfinite().all().item()),
                "key_finite": bool(key.isfinite().all().item()),
                "qk_fp16_finite": bool(qk16.isfinite().all().item()),
                "qk_fp32_finite": finite32,
                "qk_fp32_max_abs": qk32.abs().max().item() if finite32 else None,
                "scaled_fp32_max_abs": (qk32*scaling).abs().max().item() if finite32 else None,
                "fp16_limit": torch.finfo(torch.float16).max})
            return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

        qwen.eager_attention_forward = inspect
        try:
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        finally:
            qwen.eager_attention_forward = original

        def prescale(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            keys = qwen.repeat_kv(key, module.num_key_value_groups)
            values = qwen.repeat_kv(value, module.num_key_value_groups)
            scores = (query * scaling) @ keys.transpose(2, 3)
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, :keys.shape[-2]]
            weights = scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
            return (weights @ values).transpose(1, 2).contiguous(), weights

        qwen.eager_attention_forward = prescale
        try:
            pre = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float()
            report["prescale_eager_fp16_finite"] = bool(pre.isfinite().all().item())
        finally:
            qwen.eager_attention_forward = original
        report["eager_layer_diagnostics"] = diagnostic
        # Controlled diagnostic: only QK accumulation/scaling/softmax use FP32.
        def safe(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            keys = qwen.repeat_kv(key, module.num_key_value_groups)
            values = qwen.repeat_kv(value, module.num_key_value_groups)
            scores = (query.float() @ keys.float().transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, :keys.shape[-2]].float()
            weights = scores.softmax(dim=-1).to(query.dtype)
            output = (weights @ values).transpose(1, 2).contiguous()
            return output, weights
        qwen.eager_attention_forward = safe
        try:
            repaired = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float()
            report["fp32_qk_eager_finite"] = bool(repaired.isfinite().all().item())
        finally:
            qwen.eager_attention_forward = original

        model.config._attn_implementation = "sdpa"
        backends = []
        for name, flags in [("math", (True, False, False)), ("efficient", (False, True, False)),
                             ("flash", (False, False, True))]:
            torch.backends.cuda.enable_math_sdp(flags[0])
            torch.backends.cuda.enable_mem_efficient_sdp(flags[1])
            torch.backends.cuda.enable_flash_sdp(flags[2])
            torch.backends.cuda.enable_cudnn_sdp(False)
            for length in (128, 2048):
                record = {"backend": name, "input_tokens": length}
                try:
                    x = torch.tensor([cases["long"][:length]], device="cuda")
                    with warnings.catch_warnings(record=True) as caught:
                        y = model(input_ids=x, use_cache=False, logits_to_keep=1).logits.float()
                    record["finite"] = bool(y.isfinite().all().item())
                    record["first_token"] = y.argmax().item()
                    record["warnings"] = [str(w.message) for w in caught]
                    if name == "math" and length == 128:
                        report["repaired_eager_vs_math_max_abs"] = (repaired-y).abs().max().item()
                        report["prescale_eager_vs_math_max_abs"] = (pre-y).abs().max().item()
                    durations = []
                    for _ in range(3):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        model(input_ids=x, use_cache=False, logits_to_keep=1)
                        torch.cuda.synchronize()
                        durations.append((time.perf_counter()-start)*1000)
                    record["warm_forward_ms"] = durations
                except Exception as exc:
                    record["error"] = repr(exc)
                backends.append(record)
        report["backends"] = backends
        # BF16 is a separate numerical probe, never pooled with FP16 trials.
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        model.to(torch.bfloat16)
        model.config._attn_implementation = "eager"
        bf = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
        report["bf16_eager_finite"] = bool(bf.isfinite().all().item())
    save_json(output / "probe.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
