---
name: transformer-block-reviewer
description: 2026년 기본값에 따라 트랜스포머 블록 구현을 검토하고 편차를 표시합니다.
version: 1.0.0
phase: 7단계
lesson: 05강
tags: [transformers, architecture, review]
---

트랜스포머 블록 소스(PyTorch / JAX / numpy / 의사 코드)와 의도된 역할(인코더 / 디코더 / 인코더-디코더)이 주어지면 다음을 출력합니다:

1. 배선 확인. 프리노름(pre-norm) 또는 포스트노름(post-norm). 각 서브레이어 주위의 잔차 연결. 저자가 이유를 명시하지 않는 한, 포스트노름은 2026년 기본값이 아니므로 표시합니다.
2. 정규화(Normalization). LayerNorm vs RMSNorm. RMSNorm이 선호됩니다. Q/K/V/O 투영에 편향(bias) 항이 있으면 표시합니다. 대부분의 2026년 모델은 이를 제거합니다.
3. 어텐션(Attention) 형태. MHA / GQA / MQA / MLA. 디코더 블록의 경우: 인과(causal) 마스크가 적용되었는지 확인합니다. 교차 어텐션(Cross-Attention)의 경우: Q는 디코더에서, K/V는 인코더에서 가져오는지 확인합니다.
4. FFN. 활성화 함수(Activation)(ReLU / GELU / SwiGLU / GeGLU). 확장 비율. ~2.67×의 SwiGLU는 현대적 기본값이며, 4×의 ReLU/GELU는 고전적입니다.
5. 위치 신호(Positional signal). RoPE / ALiBi / 절대 위치가 예상된 위치(일반적으로 RoPE의 경우 Q,K 투영)에 적용되었는지 확인합니다.

포스트노름(post-norm)이 적용되고 워밍업(Warmup) 스케줄이 없는 12개 이상의 레이어를 쌓은 블록에 서명하는 것을 거부합니다. 학습이 발산(diverge)할 것입니다. 인과(causal) 마스킹이 없는 디코더 블록을 거부합니다. FFN 확장 비율이 2× 미만으로 떨어지는 블록은 용량 부족 가능성이 높으므로 표시합니다. 블록이 `d_model`를 하드 코딩하고 교체용 크기를 위한 설정 필드가 없는 경우 경고합니다.
