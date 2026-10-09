---
name: transformer-review
description: 7단계의 14강을 기준으로 트랜스포머-from-scratch 구현을 검토합니다.
version: 1.0.0
phase: 7
lesson: 14
tags: [transformers, review, capstone]
---

트랜스포머-from-scratch 코드베이스(PyTorch / JAX)가 주어지면, 2026년 기본 설정에 대해 검토하고 누락되거나 잘못된 부분을 플래그합니다:

1. 어텐션(Attention). 인과적 마스크(causal mask)가 존재합니다. `sqrt(d_head)`로 스케일링합니다. 멀티헤드 분할이 작동합니다. Flash Attention이 사용 가능한 경우 사용합니다. d_model ≥ 1024인 경우 GQA를 언급합니다.
2. 위치 인코딩(Positional encoding). RoPE (2026년 선호) 또는 학습된 절대 위치 인코딩(작은 모델에 허용됨). Sinusoidal은 역사적인 방식으로 플래그합니다.
3. 블록 연결. Pre-norm (post-norm이 아님). RMSNorm (LayerNorm이 아님). SwiGLU FFN (ReLU/GELU가 아님). 모든 서브레이어에 잔차 연결. 선형 레이어의 바이어스 제거 (현대적 기본 설정).
4. 학습. AdamW (또는 2026년 이후 Muon), 선형 워밍업이 포함된 코사인 학습률 스케줄, 1.0에서의 기울기 클리핑, bf16 autocast. 토큰 임베딩과 lm_head 간의 가중치 공유.
5. 손실. 모든 위치에서 shift-by-one 교차 엔트로피. 패딩이 있는 경우 마스킹. 고정 간격으로 학습 및 검증 손실을 로깅합니다.

다음 중 하나를 포함하는 코드베이스에 대해 서명하는 것을 거부합니다: 명시적인 이유 없는 post-norm, 근거 없는 2026년 프로덕션 코드의 LayerNorm, 디코더 셀프 어텐션의 인과적 마스크 누락, 작은 LM의 가중치 공유되지 않은 임베딩. 플래그: 검증 분할 없음, 기울기 클리핑 없음, 워밍업 없이 LR > 1e-3, 또는 폴백 없이 위치 인코딩 범위를 초과하는 block_size. `python code/main.py`를 엔드투엔드로 실행하고 tinyshakespeare의 nano 설정에서 최종 검증 손실이 2.5 미만으로 떨어지는지 확인하는 것을 권장합니다.
