---
name: prompt-tensor-debugger
description: 딥러닝 코드에서 텐서 모양 오류를 단계별로 디버깅하는 프롬프트
phase: 1
lesson: 12
---

딥러닝 코드에서 텐서 모양(tensor shape) 오류가 발생했습니다. 이를 수정하는 데 도움을 주세요.

**오류 메시지:** [여기에 오류를 붙여넣으세요]

**내 텐서 모양:**
- [이름]: [모양]
- [이름]: [모양]

**수행하려는 연산:** [설명하세요]

---

디버깅 시 다음 과정을 정확히 따르세요:

**1단계: 연산 유형을 식별하세요.**
어떤 연산이 오류를 발생시켰나요? 다음 중 하나로 매핑하세요:
- 행렬 곱 / 선형 레이어 (내부 차원이 일치해야 함)
- 브로드캐스팅 (오른쪽부터 정렬, 각 차원이 같거나 1이어야 함)
- 연결 (연결 차원을 제외한 모든 차원이 일치해야 함)
- 합성곱 (특정 랭크와 채널 위치를 기대함)
- 리셰이프 (총 요소 수가 보존되어야 함)

**2단계: 모양 계약(shape contract)을 작성하세요.**
식별된 연산에 대해 예상되는 모양을 명시적으로 작성하세요:
```
matmul(A, B): A is (..., m, k), B is (..., k, n) -> (..., m, n)
broadcast(A, B): align right, each pair must be (equal) or (one is 1)
cat([A, B], dim=d): all dims match except dim d
Linear(in_f, out_f): input last dim must equal in_f
Conv2d(in_c, out_c, k): input must be (B, in_c, H, W)
```

**3단계: 불일치를 찾으세요.**
실제 모양을 계약과 비교하세요. 규칙을 위반하는 정확한 차원을 식별하세요.

**4단계: 최소한의 수정을 선택하세요.**
이 표에서 선택하세요:

| 증상 | 수정 |
|---|---|
| 배치 차원 누락 | `.unsqueeze(0)` |
| 채널 차원 누락 | `.unsqueeze(1)` |
| 추가된 크기 1 차원 | `.squeeze(dim)` |
| 행렬 곱의 내부 차원 오류 | `.transpose(-1, -2)` 또는 가중치 모양 확인 |
| NHWC에서 NCHW로 변환 필요 | `.permute(0, 3, 1, 2)` |
| NCHW에서 NHWC로 변환 필요 | `.permute(0, 2, 3, 1)` |
| 선형 레이어를 위해 공간 차원 평탄화 | `.flatten(1)` 또는 `.reshape(B, -1)` |
| 헤드 분리: (B,T,D)를 (B,H,T,D/H)로 | `.reshape(B, T, H, D//H).transpose(1, 2)` |
| 헤드 병합: (B,H,T,D/H)를 (B,T,D)로 | `.transpose(1, 2).reshape(B, T, H*(D//H))` |
| 비연속(non-contiguous) 텐서와 .view() 사용 | `.contiguous().view(...)` 또는 `.reshape(...)` 사용 |

**5단계: 수정을 검증하세요.**
각 단계에서 resulting shapes를 표시하세요. 모든 리셰이프에서 총 요소 수가 보존되는지 확인하세요. 연산의 모양 계약이 이제 충족되는지 확인하세요.

**6단계: 조용한 버그를 확인하세요.**
모양이 일치하더라도 다음을 검증해 보세요:
- 의도한 축을 따라 브로드캐스팅이 일어나고 있는지 (우연히 발생하지 않았는지)
- 감소 연산이 올바른 차원을 합산하고 있는지
- 배치 차원(dim 0)가 전체 순전파 과정에서 유지되는지
- 차원 순서가 중요할 때 전치 + 재구성이 사용되었는지 (재구성만 사용되지 않았는지)

응답을 다음 형식으로 작성하세요:
```
OPERATION: [what operation failed]
EXPECTED: [shape contract]
ACTUAL: [what shapes were provided]
MISMATCH: [which dimension, why]
FIX: [exact code]
RESULT: [shapes after fix]
```
