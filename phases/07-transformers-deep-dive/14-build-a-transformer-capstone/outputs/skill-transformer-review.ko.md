---
name: transformer-review
description: Review a transformer-from-scratch implementation against the 13 Phase 7 lessons.
version: 1.0.0
phase: 7
lesson: 14
tags: [transformers, review, capstone]
---

처음부터 구현한 트랜스포머(`transformer-from-scratch`) 코드베이스(PyTorch / JAX)를 2026년 표준 사양에 따라 검토하고, 누락되었거나 잘못된 부분을 식별하세요.

1. **어텐션(Attention)**: 인과적 마스크(Causal mask)가 존재해야 합니다. `sqrt(d_head)`로 스케일링을 수행해야 하며, 멀티 헤드 분할(Multi-head split)이 정상적으로 작동해야 합니다. 사용 가능한 경우 Flash Attention을 적용합니다. `d_model` ≥ 1024인 경우 GQA(Grouped Query Attention) 적용 여부를 확인합니다.
2. **위치 인코딩(Positional encoding)**: RoPE(2026년 권장 사항) 또는 학습된 절대적 위치 인코딩(learned absolute, 소형 모델에서 허용)을 사용해야 합니다. 사인파(Sinusoidal) 방식은 과거 방식(historical)으로 표시합니다.
3. **블록 배선(Block wiring)**: Pre-norm(Post-norm 아님)을 사용해야 합니다. RMSNorm(LayerNorm 아님)을 사용하며, SwiGLU FFN(ReLU/GELU 아님)을 적용합니다. 모든 서브레이어(sublayer) 주변에 잔차 연결(Residuals)을 배치합니다. 선형 레이어(`linear layers`)에서 편향(Biases)은 제거합니다(현대적 기본값).
4. **학습(Training)**: AdamW(또는 2026년 이후 기준 Muon)를 사용하며, 선형 웜업(linear warmup)이 포함된 코사인 학습률 스케줄(cosine LR schedule)을 적용합니다. 그래디언트 클리핑(gradient clipping)은 1.0으로 설정하며, `bf16` autocast를 사용합니다. 토큰 임베딩(token embedding)과 `lm_head` 사이의 가중치 공유(Weight tying)를 적용합니다.
5. **손실(Loss)**: 모든 위치에서 1칸 이동된(Shift-by-one) 교차 엔트로피(cross-entropy)를 계산합니다. 패딩(padding)이 있다면 마스킹 처리합니다. 정해진 간격으로 학습 및 검증 손실(train and val loss)을 기록합니다.

다음 중 하나라도 해당되는 코드베이스는 승인을 거부하세요: 명시적인 이유 없는 Post-norm 사용, 정당한 근거 없는 2026년 프로덕션 코드 내의 LayerNorm 사용, 디코더 셀프 어텐션(decoder self-attention)에서 인과적 마스크 누락, 소형 언어 모델(small LM)에서 임베딩 미공유(untied embeddings). 

또한 다음 사항을 표시하세요: 검증 분할(validation split) 없음, 그래디언트 클리핑 없음, 웜업 없는 1e-3 초과 학습률(LR), 또는 폴백(fallback) 없이 위치 임베딩 범위를 초과하는 `block_size`. `python code/main.py`를 엔드 투 엔드(end-to-end)로 실행하고, nano 설정에서 tinyshakespeare 데이터셋에 대한 최종 검증 손실(val loss)이 2.5 미만인지 확인하는 것을 권장합니다.
