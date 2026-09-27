---
name: attention-shapes
description: 어텐션 구현에서의 shape 버그 디버깅.
phase: 5
lesson: 10
---

잘못 구현된 어텐션 구현체가 주어지면, shape 불일치를 찾아내야 합니다. 다음을 출력하세요:

1. 어떤 행렬의 shape가 잘못되었는지 확인하고, 해당 텐서의 이름을 명시하세요.
2. `(d_s, d_h, d_attn, T_enc, T_dec, batch_size)`로부터 유도된 올바른 shape를 제시하세요.
3. 한 줄로 된 수정 방법(One-line fix)을 제시하세요. Transpose, reshape 또는 projection 중 하나를 사용합니다.
4. 회귀(regression)를 방지하기 위한 테스트 코드를 작성하세요. 일반적으로 `output.shape == (batch, T_dec, d_h)`이고, `weights.shape == (batch, T_dec, T_enc)`이며, `weights.sum(dim=-1)`이 1에 가까운지 assert 합니다.

암묵적 브로드캐스팅(silent broadcast)을 이용한 수정 방식은 권장하지 마세요. 브로드캐스팅으로 숨겨진 버그는 나중에 정확도가 조용히 저하되는 문제로 나타납니다.

Bahdanau 어텐션의 경우, 디코더 입력이 `s_{t-1}`(step 이전 상태)이어야 함을 명시하세요. Luong 어텐션의 경우 `s_t`(step 이후 상태)입니다. Dot-product 어텐션에서 처음 접할 때 가장 흔히 발생하는 오류는 query/key 차원 불일치입니다. 이 부분을 명시적으로 지적하세요.
