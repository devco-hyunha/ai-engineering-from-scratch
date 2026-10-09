---
name: attention-shapes
description: 어텐션 구현의 모양 버그를 디버깅합니다.
phase: 5단계
lesson: 10강
---

고장난 어텐션 구현이 주어지면, 모양 불일치를 식별합니다. 출력:

1. 어떤 행렬의 모양이 잘못된지 식별합니다. 텐서를 명시하세요.
2. `(d_s, d_h, d_attn, T_enc, T_dec, batch_size)`에서 유도된, 모양이 어떻게 되어야 하는지 설명하세요.
3. 한 줄 수정안. 전치(Transpose), 재형성(Reshape), 또는 투영(Project)을 수행하세요.
4. 회귀를 잡기 위한 테스트. 일반적으로 `output.shape == (batch, T_dec, d_h)`, `weights.shape == (batch, T_dec, T_enc)`, `weights.sum(dim=-1)`가 1에 가깝다고 단언(Assert)합니다.

침묵하는 브로드캐스트를 숨기는 수정안을 추천하지 마세요. 브로드캐스트가 숨기는 버그는 나중에 침묵하는 정확도 저하로 표면화됩니다.

Bahdanau 혼란에 대해, 디코더 입력은 `s_{t-1}` (단계 전 상태)라고 주장하세요. Luong의 경우 `s_t` (단계 후 상태)입니다. 점곱 어텐션에서 가장 흔한 첫 번째 오류는 쿼리/키 차원 불일치입니다 — 이를 명시적으로 플래그하세요.
