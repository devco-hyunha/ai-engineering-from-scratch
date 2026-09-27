---
name: transformer-block-reviewer
description: Review a transformer block implementation against 2026 defaults and flag drift.
version: 1.0.0
phase: 7
lesson: 5
tags: [transformers, architecture, review]
---

트랜스포머 블록 소스(PyTorch / JAX / numpy / pseudocode)와 의도된 역할(encoder / decoder / encoder-decoder)이 주어지면, 다음 항목을 검토하여 출력하세요:

1. **연결 구조 확인(Wiring check).** Pre-norm 또는 post-norm 여부를 확인합니다. 각 서브레이어(sublayer) 주변의 잔차 연결(residual connections)을 확인하세요. 작성자가 별도의 이유를 명시하지 않는 한, post-norm은 2026년 기본값(default)이 아니므로 플래그를 표시합니다.
2. **정규화(Normalization).** LayerNorm 대신 RMSNorm을 사용했는지 확인합니다. RMSNorm 사용을 권장합니다. Q/K/V/O 프로젝션에 편향(bias) 항이 포함되어 있다면 플래그를 표시하세요. 대부분의 2026년 모델은 이를 제거합니다.
3. **어텐션 형태(Attention shape).** MHA / GQA / MQA / MLA 여부를 확인합니다. 디코더 블록의 경우 인과적 마스크(causal mask)가 적용되었는지 확인하세요. 크로스 어텐션(cross-attention)의 경우 Q는 디코더에서, K/V는 인코더에서 오는지 확인합니다.
4. **FFN.** 활성화 함수(ReLU / GELU / SwiGLU / GeGLU)와 확장 비율(Expansion ratio)을 확인합니다. 약 2.67×의 SwiGLU가 현대적 기본값이며, 4× ReLU/GELU는 클래식한 방식입니다.
5. **위치 신호(Positional signal).** RoPE / ALiBi / absolute 방식이 적절한 위치(일반적으로 RoPE의 경우 Q, K 프로젝션)에 적용되었는지 확인하세요.

**검토 기준 및 거부 조건:**
- post-norm을 사용하면서 웜업 스케줄(warmup schedule) 없이 12개 이상의 레이어를 쌓은 블록은 승인을 거부하세요. 학습이 발산할 가능성이 높습니다.
- 인과적 마스킹(causal masking)이 없는 디코더 블록은 승인을 거부하세요.
- FFN 확장 비율이 2× 미만인 블록은 용량 부족(under-capacity) 가능성이 높으므로 플래그를 표시하세요.
- `d_model`이 교체 가능한 크기 설정을 위한 `config` 필드 없이 하드코딩되어 있다면 경고를 표시하세요.
