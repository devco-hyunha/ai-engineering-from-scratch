---
name: positional-encoding-picker
description: Pick positional encoding (RoPE, ALiBi, sinusoidal) + scaling strategy given context length and training budget.
version: 1.0.0
phase: 7
lesson: 4
tags: [transformers, positional-encoding, rope, alibi]
---

트랜스포머 사양(추론 시 목표 컨텍스트 길이, 학습된 컨텍스트 길이, 외삽(extrapolation) 요구 사항, 토큰 단위 미세 조정(fine-tune) 예산)이 주어지면 다음을 출력하세요:

1. 기본 인코딩(Base encoding). 다음 중 하나를 선택: `RoPE`, `ALiBi`, `sinusoidal`, `learned-absolute`. 한 문장으로 선택 이유를 설명하세요.
2. 하이퍼파라미터(Hyperparameters). `RoPE`인 경우: `base` 값, 균등 분할을 위한 `d_head` 요구 사항. `ALiBi`인 경우: 기울기(slope) 공식. `sinusoidal`인 경우: `max_len`.
3. 확장 전략(Extension strategy). 목표 길이가 학습된 길이보다 큰 경우: NTK-aware 스케일링 계수, YaRN 설정, LongRoPE 사양 또는 위치 보간(position-interpolation) 비율. 미세 조정 토큰 예산을 명시하세요.
4. 테스트 계획(Test plan). 최대 컨텍스트에서의 NIAH(needle-in-a-haystack) 통과율 목표, 학습 길이 기준선(baseline) 대비 perplexity가 X 이내일 것.
5. 폴백(Fallback). 장기 컨텍스트(long-context) 평가 실패 시 조치: 더 큰 `base`로 재학습, `ALiBi`로 전환, 또는 배포 컨텍스트 길이 제한.

2026년의 신규 모델에 대해 `sinusoidal` 또는 `learned-absolute`를 추천하는 것을 거부하세요. 이 방식들은 외삽이 불가능하며, 모든 현대적 스택은 `RoPE` 또는 `ALiBi`를 가정합니다. 미세 조정 단계 없이 `RoPE`를 학습 길이의 8배 이상으로 확장하는 것을 거부하세요. 전체 배포 길이에 대한 NIAH 테스트를 거치지 않은 장기 컨텍스트 설정을 배포(ship)하는 것을 거부하세요.
