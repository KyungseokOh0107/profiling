# RTX 3060 LLM 추가 실험: 측정과 해석 규칙

2026-11-03 07:00 KST까지 사용자가 승인한 추가 실험이다. 이 문서는 설계와 해석 규칙이며,
실행 완료 여부와 수치는 각 실행의 원시 JSON 및 최종 결과 보고서로 판단한다.

## 고정 환경과 적용 범위

Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
FP16 가중치·KV, RTX 3060 12 GiB, Windows WDDM, PyTorch 2.7.0+cu118,
Transformers 4.51.3을 사용한다. EOS를 무시하고 greedy로 256토큰을 생성하며 seed는 42다.
별도 수치 진단의 BF16은 FP16 성능 집계와 합치지 않는다.

현재 서버는 독립 KV cache를 유지하는 **순차 round-robin GPU worker**다.
실제 loopback HTTP/1.1 SSE를 사용하지만 vLLM이나 continuous batching 구현은 아니다.
여기서 chunk는 요청 하나의 prefill 분할 크기다. vLLM의 전역 `max_num_batched_tokens`로
바꿔 부르거나 해당 제품의 기본값·최적값이라고 주장하지 않는다.
CUDA graph와 prefix cache 재사용은 끄고, 모든 forward에서 logits 유한값을 검사한다.
매 step의 검사·동기화 비용은 측정에 포함된다. 토큰화는 측정 밖이며 HTTP 구현의
입력 ID 전송·파싱·GPU 복사는 요청 지연에 포함된다.

## 실험별 가설과 반증

| 실험 | 가설 | 대조군 / 독립변수 | 통제변수 | 측정값 | 가설을 지지하지 않을 결과 |
|---|---|---|---|---|---|
| FP16 수치 진단 | eager attention의 unscaled QK overflow가 NaN을 전파한다 | 원래 eager / QK FP32 처리 / Q 사전 scaling | 동일 가중치·입력·나머지 연산 | 최초 비유한 layer, Q/K와 QK finite, FP32 최대 절댓값, 출력 finite | Q/K부터 비유한이거나 제한된 연산 변경으로 finite가 복구되지 않음 |
| HTTP 간섭 재현 | 긴 B prefill이 A의 다음 decode를 지연한다 | A 단독 / 짧은 B / 긴 B | A 128→256, B 출력256, A#16 수신 후 B | A 수신 간격, B TTFT, 실제 중첩, 서버 시간대 | B가 실제 중첩했는데 긴 B의 공백 증가가 재현되지 않음 |
| chunk sweep | 분할을 줄이면 A 공백이 줄고 B TTFT가 증가한다 | 256 / 512 / 2048 | 모델·backend·입출력·max_seqs8·동일 클라이언트 | A의 B 투입 후 최대 공백, B TTFT, output tokens/s | 공백 감소 또는 B TTFT 비용이 반복 실행에서 관찰되지 않음 |
| backend 교차 비교 | efficient SDPA가 math보다 prefill 간섭을 줄인다 | math / efficient × 세 chunk, 블록 내 무작위 순서 | 동일 FP16 모델과 persistent 서버 | 위 지연·처리량, 반복 블록별 차이, finite/cache 검증 | 동일 chunk에서 개선 방향이 재현되지 않거나 유효성 검사 실패 |
| 혼합 부하 | 과부하에서는 chunk만으로 입장 대기 문제를 해결할 수 없다 | .08 / .24 requests/s × 세 chunk | 80:20, 50요청, 같은 도착 기록, cap8, 2회 역순 반복 | 유형별 TTFT·간격, admission wait, goodput, TPS, 오류 | 과부하에서 실제 cap 대기가 없거나 chunk 변경만으로 지연 목표와 처리량 모두 회복 |
| static batch | 여러 요청을 한 forward에 묶으면 총 처리량이 증가할 수 있다 | batch1/2/4/8 | efficient FP16, 입력128, 출력256, CPUthreads4 | 총 TPS, request decode step, peak allocated, 실제 KV bytes | 배치 증가에도 처리량 증가가 없거나 latency·메모리 비용이 이점을 상쇄 |
| CPU threads | CPU dispatch 설정이 작은 batch 추론에 영향을 줄 수 있다 | threads1/2/4/8 | batch1, 입력128, 출력256, 동일 backend | TPS와 decode step | 반복 변동 범위 안에서 차이가 작음 |
| 문맥 길이 | 긴 cache가 decode 지연·KV 크기에 영향을 준다 | 입력128/512/1024/2048 | batch1, threads4, 출력256 | decode step, KV 실제/이론 bytes | KV 공식 불일치 또는 지연 증가가 재현되지 않음; 메모리 크기와 속도 가설은 분리 |
| chat 입력 | 간섭의 방향이 completion 입력에만 국한되지 않는다 | role delimiter가 있는 고정 chat 입력 × 세 chunk | exact128/2048, 동일 출력·backend | A 공백, B TTFT, 토큰 수·finite | 방향이 뒤집히거나 차이가 반복 변동에 묻힘 |
| 클라이언트 위치 | 동일 프로세스의 Python 경합이 측정에 영향을 줄 수 있다 | 서버 프로세스 내부 / 별도 프로세스 클라이언트 × 세 chunk | 같은 persistent GPU 서버·입력·backend, 순서 무작위 | A 공백, TTFT, 생성→수신 지연, TPS | 분리 전후 차이가 작으면 이 범위에서 영향이 제한적이라고 보고 |

## 시계·스트림·원시 데이터

SSE `accepted`, `token`, `done`, `error`는 별도 이벤트다. `token` 이벤트에 명시적인
token ID와 1-based index를 넣으므로 TCP 패킷·읽기 횟수·문자 수를 토큰 수로 세지 않는다.
한 이벤트의 빈 줄까지 읽은 시점을 수신 시각으로 기록한다. A#16 수신 직후 B 클라이언트를
제출하고, A#16 수신·B 예정·B 실제 전송 시각을 모두 남긴다.

`perf_counter_ns()`를 쓰며 실제 구현은 `QueryPerformanceCounter()`, 단조 증가,
adjustable=false, 보고된 해상도는 100ns다. 나노초 단위 저장이 1ns 측정 정밀도를 뜻하지 않는다.
Windows Python 3.10부터 이 시계는 프로세스 공통이므로 별도 클라이언트에서도 같은 축으로
비교한다. [Python 3.11 문서](https://docs.python.org/3.11/library/time.html#time.perf_counter),
[로컬 시계 확인](results/overnight/clock_info.json).

요청마다 request ID, 입력 토큰 수, scheduled/send/accepted/admitted 시각,
각 forward의 phase·입력 토큰 수·start/end·active/pending 수,
출력 token ID·index·generated/received 시각, done·status·error, 출력 hash를 보존한다.
실행별 모델 revision, backend, dtype, chunk, max_seqs, seed, 입력 ID/hash,
실행 순서·반복 번호·warmup 여부, 코드 snapshot/hash를 남긴다.

TTFT는 다음 합으로 검증한다. HTTP ingress + admission wait + 입장 후 첫 forward 대기
+ prefill forward 경과 시간 합 + prefill 사이 교대 대기 + 첫 토큰 전달 시간.
예정 시각 기준 TTFT에는 추가로 client send lag를 포함한다.
서버 forward 경과 시간은 CUDA 완료를 기다린 wall time이며 GPU kernel active time과 다르다.
클라이언트 최대 공백과 겹치는 서버 A/B prefill/decode 구간을 별도로 계산한다.

## 표본과 부하

기본 HTTP 비교는 조건별5회다. backend crossover는 블록별 무작위 순서로8회,
chat robustness는 조건별5회, static batch/CPU/context는 조건별5회 계획이다.
실제 완료 수가 적으면 원래 계획 수로 표기하지 않는다. 예비 실행과 warmup은 본 집계에서 제외한다.
프로파일러 실행의 latency·TPS는 제외하고, CUDA 이벤트와 실행 순서 확인에만 사용한다.

혼합 부하는 짧은 요청80%, 긴 요청20%, 출력은 모두256이다. 도착 간격은 seed로 생성한
지수분포 표본을 정규화해 유한 도착 창의 평균 rate를 맞춘다. 완전히 동일한 도착 기록을
각 chunk에 open-loop로 재생한다. 두 번째 반복에서는 조건 순서를 뒤집는다.
짧은50요청 이하 pilot에서 p99로 결론 내리지 않는다. 본 측정도 조건당 요청100개,
긴 요청20개·독립 실행2개이므로 p95와 반복 범위는 탐색적 결과로 해석한다.
토큰256개를 독립 요청256개처럼 취급하지 않는다.

예비10요청의 .24 rate에서는 cap 대기가 확인되지 않았다. 20요청으로 늘려 실제 입장 대기를
확인한 뒤 본 측정 전에 .08/.24를 고정했다.
[rate 고정 근거](results/overnight/LOAD_RATES_FROZEN.json).
.24는 제한된 길이의 과부하 burst이며, 안정적인 장기 서비스 처리 용량으로 해석하지 않는다.
.08도 순간적인 burst가 있으므로 평균 rate가 낮다는 이유만으로 대기열이 없다고 가정하지 않는다.

합성 서비스 목표는 **짧은 요청의 예정 시각 기준 TTFT≤2초, 긴 요청≤4초,
각 요청의 최대 수신 간격≤350ms**로 본 측정 전에 고정했다. 실제 업무 SLA가 아니다.
goodput은 세 조건 중 해당 유형의 TTFT 조건과 간격 조건을 만족해 완료한 요청 수를
첫 예정 도착부터 최종 완료/timeout까지의 시간으로 나눈다. drain을 분모에서 빼지 않는다.
오류 요청도 전체 요청 분모에 포함한다. 전체 output tokens/s와 goodput을 함께 제시한다.

## 메모리와 주장 범위

실제 모델 설정은 layer28, hidden1536, query heads12, KV heads2, head dimension128,
vocab151936, tied embeddings다. 고유 파라미터 수는 1,543,714,304이므로 FP16 가중치는
3,087,428,608 bytes다. KV는 토큰·시퀀스당 `2 × 28 × 2 × 128 × 2 = 28,672 bytes`다.
입력2048+출력256을 수용하는 논리 문맥2304는 시퀀스당63MiB의 KV 상한이다.
마지막 생성 토큰을 다음 forward에 넣기 전 종료하므로 실제 cache 길이는 입력+255다.
static batch 실험은 실제 K/V tensor bytes가 이 공식과 일치하는지 검증한다.

가중치와 KV 공식에는 활성값·allocator·CUDA context·graph pool·Windows 화면 사용량이 없다.
`torch.cuda.max_memory_allocated`도 전체 프로세스/장치 VRAM 사용량이 아니다.
이번 실행은 CUDA graph를 사용하지 않으므로 graph 메모리 비용을 실측했다고 주장하지 않는다.

GPU utilization이나 batch speedup만으로 DRAM bandwidth 병목을 입증하지 않는다.
Windows desktop, GPU clock·power 미고정, synthetic prompt, 자체 scheduler, 동기화·finite 검사 비용을
명시한다. 이번 결과는 장애 재현과 원인 축소 및 설정의 비용 비교 경험이며,
실제 vLLM의 최적값은 지원되는 Linux 환경에서 별도 검증해야 한다.
