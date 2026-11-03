# 결과 데이터 안내

이 폴더는 2026-11-03 RTX 3060 / Qwen2.5-1.5B-Instruct 측정의 원시 기록과 파생 집계를 보관한다.
최종 해석은 [전체 보고서](../LLM_OVERNIGHT_RESULTS.md), 재계산 명령은 [재현 절차](../REPRODUCING.md)에 있다.

## 어떤 데이터를 합쳐야 하는가

| 경로 | 용도 / 최종 집계 포함 여부 |
|---|---|
| `overnight/http_pairs_math`, `http_pairs_efficient` | HTTP 기본 대조군, 본 측정 |
| `overnight/backend_crossover` | 같은 서버의 backend×chunk 교차 비교, 본 측정 |
| `overnight/http_mixed_load` | 600요청 혼합 부하, 본 측정 |
| `overnight/static_batch_fp16` | HTTP와 별도인 실제 GPU batch·CPU·문맥 실험, 본 측정 |
| `overnight/chat_pairs`, `client_process_crossover`, `admission_crossover` | 입력 형식·프로세스·admission 상한 추가 검증, 본 측정 |
| `overnight/http_load_pilot*` | 도착률 선택용 예비 실행, 본 성능 집계 제외 |
| `overnight/backend_probe*` | 수치 오류·backend 사용 가능 여부 진단, 성능 집계 제외 |
| `overnight/http_smoke_math` | HTTP 구현 확인용 smoke, 본 측정 제외 |
| `llm_native_sdpa` | 이전 in-process queue 실험25회 및 별도 profiler 검증; HTTP209회 합계와 분리 |
| `llm_native_smoke` | NaN이 확인된 무효 eager FP16 실행; `INVALID.json`을 보존하며 수치는 사용하지 않음 |
| `llm_native_sdpa_smoke`, `llm_native_validation_attempt` | 구현 점검·검증 시도, 본 성능 집계 제외 |

각 cohort의 `warmup` 기록은 제외한다. Profiler를 켠 실행도 성능 집계에 합치지 않는다.
`llm_preflight.json`은 **설치 전** 환경 기록으로, 그 안의 미설치/미실행 상태는 이후 실험 완료 상태와 다르다.

## 파일을 읽는 방법

- `manifest.json`: 모델 revision, backend, 입력 hash, 실행 시점과 설정. 구체적인 기록 항목은 cohort마다 다르다.
- `inputs.json`, `arrivals_r*.json`: 실제 입력 token ID와 고정된 요청 도착 기록.
- `pair_*.json`, `load_*.json`: 실행별 summary, client requests, server 기록. 검증 가능한 원본이다.
- `summaries.json`: 완료 실행의 요약 목록. 원시 실행 파일과 개수가 일치하는지 검증한다.
- `aggregate.json`, `requests.csv`, `gap_attribution.json`, `validation.json`: 원시 기록에서 재생성 가능한 집계·검증 결과.
- `implementation_snapshot.py`: 실행 당시 코드. 현재 루트 스크립트와 다를 수 있으며, cohort별 실제 코드는 snapshot/hash로 확인한다.
- `overnight/RESULTS_DATA.json`: 완료 cohort 전체의 집계, 표본 수, 출력 hash, paired 비교, SLO 실패 유형.
- `overnight/EXPERIMENT_PLAN.json`, `LOAD_RATES_FROZEN.json`: 본 측정 전 계획과 도착률 선택 근거.
- `overnight/FINAL_DEADLINE_CHECK.json`: 실행 종료·잔존 프로세스·검증 상태를 확인한 당시 기록.

Static batch는 `r*_b*_t*_l*.json`에 batch·CPU threads·입력 길이, 각 step 시간, 실제/이론 KV bytes를 저장한다.

## 원시 HTTP 요청의 주요 필드

| 위치 | 주요 필드 | 의미 |
|---|---|---|
| client request | `request_id`, `prompt_tokens`, `status`, `error` | 요청 식별, 입력 길이, 성공/오류 |
| client request | `scheduled_ns`, `send_ns`, `done_ns` | 예정 도착, 실제 전송, 완료 수신 |
| `tokens[]` | `index`, `token_id`, `generated_ns`, `received_ns` | 생성 토큰과 서버 생성/클라이언트 수신 시각 |
| `aux_events[]` | `accepted`, `done` | 토큰으로 세지 않는 SSE 메타데이터 |
| server request | `accepted_ns`, `admitted_ns`, `completed_ns` | 접수, 실행 집합 입장, 생성 완료 |
| `steps[]` | `phase`, `input_tokens`, `start_ns`, `end_ns`, `active`, `pending` | prefill/decode 실행 구간과 요청 수 |

`*_ns`는 동일 PC의 단조 시계 값이며 UTC 시각이 아니다. ns 저장 단위를 실제1ns 정밀도로 해석하지 않는다.
TTFT 분해의 forward 시간은 동기화된 wall time으로, 순수 CUDA kernel active time과 구분한다.

## Git 포함 범위

측정 JSON/CSV, 설정, 코드 snapshot, 검증 요약과 PNG/SVG는 포함한다. 합성 입력만 사용했으며 실제 사용자 문서는 없다.
일부 환경 기록에는 원래 실행 PC의 절대 경로가 남아 있다. 이는 provenance이며 재현 PC에서 그대로 쓸 경로가 아니다.

가상환경, 모델 가중치, 임시 로그·ready 상태·tmp 파일, 대형 `*.pt.trace.json` profiler 원본은 제외한다.
Profiler 원본 두 개는 측정 PC에만 남아 있고, Git에는 [trace 검증 요약](llm_native_sdpa/trace_validation.json)을 포함한다.
원본 trace가 필요한 경우 [이전 실험 재현 절차](../REPRODUCING.md#reproduce-after-the-separate-environment-is-prepared)의 `--profile` 실행으로 새로 수집한다.
