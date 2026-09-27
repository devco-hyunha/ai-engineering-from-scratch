---
name: spec-decode-picker
description: 새로운 LLM 추론 워크로드에 적합한 추측적 디코딩(speculative decoding) 전략(vanilla / Medusa / EAGLE / lookahead) 및 튜닝 파라미터를 선택합니다.
version: 1.0.0
phase: 7
lesson: 16
tags: [inference, decoding, latency, speculative, optimization]
---

# Speculative Decoding Picker (추측적 디코딩 선택기)

엔지니어가 바닐라 추측 디코딩(vanilla speculative), Medusa, EAGLE 또는 lookahead 디코딩 중 하나를 선택하고, 특정 워크로드에 맞춰 `N`(초안 길이, draft length)을 조정할 수 있도록 도와줍니다.

## 수집해야 할 입력 사항 (Inputs to gather)

1. **검증 모델 (Verifier model)** — 최종 출력을 생성하는 LLM입니다. 모델의 크기가 중요합니다 (속도 향상을 위해 초안 생성 비용이 검증 비용보다 작아야 합니다).
2. **워크로드 유형 (Workload type)** — 코드, 채팅, 구조화된 출력(structured output), 요약 등입니다. 이는 수락률(acceptance rate)을 결정합니다.
3. **샘플링 전략 (Sampling strategy)** — greedy, low-T, high-T, beam 방식입니다. High-T 샘플링은 수락률을 저하시킵니다.
4. **하드웨어 타겟 (Hardware target)** — 메모리 예산에 따라 별도의 초안 모델(draft model)을 탑재할 수 있는지 여부가 결정됩니다.
5. **엔지니어링 예산 (Engineering budget)** — Medusa와 EAGLE은 미세 조정(fine-tuning)이 필요하지만, vanilla 방식과 lookahead 방식은 필요하지 않습니다.
6. **지연 시간 목표 (Latency target)** — 대화형 채팅(TTFT <500ms, 토큰당 <50ms)인지, 아니면 배치(batch, 처리량 우선) 방식인지 결정합니다.

## 결정 규칙 (Decision rules)

- **빠른 시작, 학습 없음 (Quick start, no training)**: 동일 계열의 1B–3B 모델을 사용하는 바닐라 초안(vanilla draft) 방식. 일반적인 경우보다 2배 빠름.
- **미세 조정 가능 (You can fine-tune)**: 검증기(verifier)의 은닉 상태(hidden states)를 사용하는 EAGLE-2 또는 EAGLE-3 방식. 일반적인 경우보다 3–4배 빠름.
- **미세 조정은 가능하지만 두 개의 모델을 실행할 수 없는 경우 (You can fine-tune but can't run two models)**: Medusa (검증기에 추가 헤드 부착). 2–3배 빠름.
- **학습 예산이 없고 초안 모델을 사용할 수 없는 경우 (No training budget, no draft model available)**: lookahead decoding 방식. 1.3–1.6배 빠름.
- **배치 중심 서빙 (Batch-heavy serving)**: 연속 배칭(continuous batching)이 더 중요합니다. 배치가 커질수록 검증기가 이미 포화 상태가 되므로 추측적 디코딩의 이득(speculative gains)은 감소합니다.
- **높은 온도 또는 확률적 샘플링 (High temperature or stochastic sampling)**: 수락률(acceptance)이 급격히 떨어집니다. $N$ 값을 낮추거나(2–3) 기능을 비활성화하는 것을 고려해 보세요.
- **구조화된 출력 (Structured output (JSON, code))**: 수락률이 높습니다. 최대 속도 향상을 위해 $N$ 값을 7 이상으로 높여 보세요.

## 튜닝 (Tuning)

- **N (초안 길이, draft length)**: 5에서 시작하세요. 수락률(acceptance)을 측정합니다. 만약 $\alpha > 0.9$라면 7로 늘리세요. 만약 $\alpha < 0.6$이라면 3으로 줄이세요.
- **초안 온도 (Draft temperature)**: 검증기(verifier)의 온도와 일치시키세요. 초안 샘플링이 일치하지 않으면 $\alpha$ 값이 손실됩니다.
- **트리 깊이 (Tree depth, EAGLE-2 / Medusa)**: 3~5개의 브랜치; 더 넓은 트리는 $\alpha > 0.8$일 때만 도움이 됩니다.
- **초안 모델 크기 (Draft model size)**: $\alpha > 0.7$을 달성하는 가장 작은 크기를 선택하세요. 70B 검증기에 1B 초안 모델을 사용하는 것이 일반적입니다. 검증기의 토크나이저(tokenizer) / 임베딩(embedding) 호환성보다 낮은 크기로 설정하지 마세요.

## 항상 주의할 점 (Always flag)

- 초안 모델(draft model)과 검증기(verifier)가 동일한 토크나이저를 사용하는지 확인하세요. 서로 다른 BPE 분할(split)은 추측적 보장(speculative guarantees)을 깨뜨립니다.
- 추측적 디코딩(Spec decoding)은 vLLM의 연속 배칭(continuous batching)과 상호작용합니다. 배치가 이미 포화 상태(saturated)라면 요청당 속도 향상 폭이 감소합니다.
- EAGLE의 은닉 상태(hidden-state) 입력은 검증기의 내부 구조를 필요로 합니다. 이는 HF API를 통해 항상 노출되는 것은 아닙니다. vLLM 또는 SGLang 런타임을 사용하는 것을 권장합니다.
- Medusa 헤드는 검증기 자체의 출력물에 대한 지도 미세 조정(supervised fine-tune)이 필요합니다. 데이터 수집 단계가 종종 가장 큰 비용을 차지합니다.

## 출력 형식 (Output format)

반환 사항:

1. **권장 사항 (Recommendation)** — 하나의 전략 명칭과 튜닝 파라미터 (예: "EAGLE-2, N=5, tree_depth=4").
2. **예상 속도 향상 (Expected speedup)** — 명시적인 $\alpha$ 가정을 포함할 것.
3. **호환성 점검 (Compatibility checks)** — 토크나이저 일치 여부, 런타임 지원 여부, KV 캐시 롤백(KV cache rollback) 지원 여부.
4. **대비책 (Fallback plan)** — 기본 전략의 성능이 미달할 경우, 다음에 시도할 방법.
5. **측정 계획 (Measurement plan)** — 대표 샘플을 통해 수락률(acceptance rate)과 속도 향상을 검증하는 방법.
