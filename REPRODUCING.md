# 재현 절차와 초기 실험 구현 상세

전체 연구 질문과 최종 결과 요약은 [README](README.md)를 먼저 읽는다.
아래 preflight 항목은 설치 전의 역사적 기록이며, 현재 완료 상태와 구분한다.

## 다른 PC에서 환경 준비

실측 환경은 Windows, Python3.11.18, torch2.7.0+cu118, Transformers4.51.3이다.
원래 PC는 기존 Conda torch를 공유했지만 다른 PC에서는 독립 환경으로 준비할 수 있다.
아래 명령은 새 환경 준비 예시이며, 실제 재현 PC의 GPU·드라이버 호환성은 먼저 확인한다.

```powershell
py -3.11 -m venv .venv-llm
.\.venv-llm\Scripts\python.exe -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
.\.venv-llm\Scripts\python.exe -m pip install -r requirements-llm.txt huggingface-hub==0.36.2
.\.venv-llm\Scripts\huggingface-cli.exe download Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

모든 benchmark는 `local_files_only=True`로 읽으므로 지정 revision의 모델과 tokenizer를 캐시에 먼저 준비해야 한다.
가중치는 Git에 포함하지 않는다. 전체 원래 패키지 목록은 [환경 snapshot](results/llm_native_requirements.txt)에 있다.
다른 환경에서 얻은 결과는 이 저장소의 원래 측정과 별도 cohort로 보관한다.

## 원래 환경과 실험 기록

The extended real HTTP/SSE campaign is described in the
[Korean overnight report](LLM_OVERNIGHT_RESULTS.md) and its
[measurement protocol](LLM_OVERNIGHT_PROTOCOL.md). It uses a native sequential
GPU worker, **not vLLM and not continuous batching**. The separate static-batch
study calls the model directly and is not an HTTP serving benchmark.

Completed Windows native Qwen measurements are documented in the
[experiment report](LLM_NATIVE_RESULTS.md), with a
[comparison figure](results/llm_native_sdpa/interference.png).

## LLM experiment status

The target experiment is Qwen2.5-1.5B-Instruct FP16 on one RTX 3060:
A has 128 input / 256 output tokens; B arrives after A's 16th output token,
with either 128 or 2048 input tokens and 256 output tokens. The original target
included chunked prefill budgets 256, 512, 2048 and the installed vLLM default.
Windows preflight prevented that vLLM comparison; the native studies compare
per-request chunks 256, 512 and 2048.

### Measured preflight, 2026-11-03 KST

See [raw environment evidence](results/llm_preflight.json) and the reusable
[preflight script](llm_preflight.py).

- Windows build 18363; registered Ubuntu and Alpine distributions are WSL1.
- RTX 3060 12 GiB; Python 3.11.18; torch 2.7.0+cu118; runtime CUDA 11.8.
- A small FP16 CUDA matrix multiplication passed; one CUDA profiler event was captured.
- vLLM, Transformers and huggingface-hub are absent from the inspected interpreter.
- Docker client exists, but connection to its daemon failed.
- At that initial preflight stage, no LLM inference or serving latency measurement
  had run. The subsequent native experiment is documented below; vLLM remains untested.

Official GPU vLLM requires Linux and does not support native Windows:
https://docs.vllm.ai/en/stable/getting_started/installation/gpu/

The user selected the separately labelled Windows Transformers preliminary
experiment. A Transformers experiment must not be presented as evidence of
vLLM scheduler performance. No OS upgrade or reboot was attempted.

## Windows native Qwen experiment

[The benchmark](benchmark_llm_native.py) uses a single GPU worker and a client
thread connected by an in-process queue. Independent A/B KV caches are advanced
round-robin: one decode token or one prefill chunk per request per round. There
is **no HTTP server, continuous GPU batching, or vLLM scheduler**. Its
`chunk_tokens` parameter is not `max_num_batched_tokens`.

- Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- FP16 weights/KV; SDPA math only (FP32 intermediates); no CUDA graph or prefix reuse.
- A: 128 input tokens, 256 output tokens. B: 128 or 2048 input tokens, 256 output tokens.
- Inputs are fixed synthetic completion prompts with repeated English document
  text, not chat-template prompts or a semantic response-quality benchmark.
- Exact generated token IDs are streamed. B is submitted after receipt of A#16.
- Greedy decoding, EOS ignored, full output logits checked for finite values on
  every forward. All conditions pay this validation and GPU synchronization cost.
- Token tensors are already on the GPU before the request; model loading and
  tokenization are excluded. Throughput is total output tokens divided by time
  from client submission of A to the last worker completion event.
- Three full long-B warmups cover all chunk sizes. Five seeded shuffled blocks
  contain A alone, short B, and long B with chunk sizes 256, 512 and 2048.
- Profiler runs are separate. Their capture waits 12 forward calls, warms up 2,
  records 24, and exports a trace with A/B prefill/decode annotations.

The first eager-attention FP16 smoke test was invalid: it emitted token ID 0
repeatedly and a follow-up validation found NaN logits. Its results are retained
with an [explicit invalidation record](results/llm_native_smoke/INVALID.json).
The SDPA runs must be kept separate; the invalid smoke is never included in
summary tables or figures. The original checkpoint config has
`use_sliding_window=false`; the Transformers 4.51.3 warning checks the non-null
window-size field regardless, and the configured 32768 window exceeds this
experiment's maximum 2304-token context.

### Reproduce after the separate environment is prepared

The local `.venv-llm` shares existing torch 2.7.0+cu118 via `--system-site-packages`.
Its additional dependencies are pinned in [requirements-llm.txt](requirements-llm.txt),
with the complete installed package snapshot in
[llm_native_requirements.txt](results/llm_native_requirements.txt).

```powershell
$nativeReplay = 'results/native_replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv-llm\Scripts\python.exe benchmark_llm_native.py --repeats 5 --profile --output $nativeReplay
.\.venv-llm\Scripts\python.exe analyze_llm_native.py --input $nativeReplay
```

Use a new output directory for another experiment to preserve raw measurements.
[The analyzer](analyze_llm_native.py) excludes warmup/profile runs, verifies exact
token counts, actual input length and request overlap, and produces a CSV,
aggregate JSON and figure. Five repeats support a preliminary median/range
comparison, not p99 claims or a production configuration recommendation.

### Reproduce the small preflight

In PowerShell, using the verified existing Conda interpreter:

```powershell
$preflightReplay = 'results/preflight_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json'
& 'C:\Users\main\anaconda3\envs\profiling\python.exe' llm_preflight.py --gpu-smoke --output $preflightReplay
```

This records package versions, NVIDIA memory information and WSL registration,
then optionally executes a tiny CUDA operation and profiler capture. It does
not install dependencies or download a model. `--gpu-smoke` is an execution
check, not an LLM benchmark; omit it for metadata-only inspection.

## Reproduce the extended HTTP campaign

Use the existing cached model and verified Python 3.11 environment. Run GPU jobs
serially. These commands create a fresh timestamped output tree and explicitly
pass a future deadline: the scripts' historical default is November 3, 2026,
07:00 KST and must not be reused for future runs. Allow at least four hours;
incomplete trials must not be reported as planned repetitions.

```powershell
$studyPython = Join-Path $PWD '.venv-llm/Scripts/python.exe'
$studyRoot = 'results/replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$studyDeadline = (Get-Date).AddHours(4).ToString('o')
& $studyPython probe_llm_backends.py --output "$studyRoot/backend_probe_v2"
& $studyPython benchmark_llm_http.py --backend math --phase pairs --repeats 5 --deadline $studyDeadline --output "$studyRoot/http_pairs_math"
& $studyPython benchmark_llm_http.py --backend efficient --phase pairs --repeats 5 --deadline $studyDeadline --output "$studyRoot/http_pairs_efficient"
& $studyPython benchmark_llm_http.py --backend efficient --phase load --count 50 --repeats 2 --rates 0.08 0.24 --deadline $studyDeadline --output "$studyRoot/http_mixed_load"
& $studyPython benchmark_llm_crossover.py --repeats 8 --deadline $studyDeadline --output "$studyRoot/backend_crossover"
& $studyPython benchmark_llm_batch.py --repeats 5 --deadline $studyDeadline --output "$studyRoot/static_batch_fp16"
& $studyPython benchmark_llm_crossover.py --chat --repeats 5 --deadline $studyDeadline --output "$studyRoot/chat_pairs"
& $studyPython benchmark_llm_process.py --repeats 5 --deadline $studyDeadline --output "$studyRoot/client_process_crossover"
& $studyPython benchmark_llm_admission.py --repeats 2 --deadline $studyDeadline --output "$studyRoot/admission_crossover"
```

Rates and synthetic SLOs above replay this campaign's frozen design; a different
machine or workload needs a new pilot. The admission study can skip a block when
there is insufficient time. The original queue manifests preserve this run's
order, but point to historical output paths; do not rerun them unchanged.

After all GPU jobs finish, validate raw token indices/hashes, the A#16 trigger,
arrival replay, latency decomposition, SLO counts and drain denominators:

```powershell
& $studyPython summarize_llm_study.py $studyRoot
& $studyPython plot_llm_overnight.py $studyRoot
& $studyPython plot_llm_interview.py "$studyRoot/backend_crossover"
& $studyPython plot_llm_queue.py "$studyRoot/http_mixed_load/load_r0_rate0.24_chunk512.json"
& $studyPython -m unittest test_llm_stream_validation
```

Figures require completed relevant conditions. Warmups, pilots, numerical probes
and profiler captures are separate from final performance comparisons. The
earlier queue-transport cohort remains in [LLM_NATIVE_RESULTS.md](LLM_NATIVE_RESULTS.md).
