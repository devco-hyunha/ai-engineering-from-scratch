---
name: prompt-tensor-debugger
description: 딥러닝 코드의 텐서 모양 오류를 위한 단계별 디버깅 프롬프트
phase: 1
lesson: 12
---

딥러닝 코드에 텐서 모양 오류가 있습니다. 고쳐 주세요.

**오류 메시지:** [여기에 오류 붙여넣기]

**내 텐서 모양:**
- [이름]: [shape]
- [이름]: [shape]

**하려는 연산:** [설명]

---

디버깅할 때 이 정확한 절차를 따르세요:

**Step 1: 연산 유형을 식별합니다.**
어떤 연산이 오류를 냈나요? 다음에 매핑하세요:
- 행렬곱 / Linear 층 (내부 차원이 일치해야 함)
- 브로드캐스팅 (오른쪽부터 맞추고, 각 차원은 같거나 1)
- 연결 (cat 차원을 제외한 모든 차원이 일치)
- 합성곱 (특정 랭크와 채널 위치를 기대)
- Reshape (총 원소 수가 보존되어야 함)

**Step 2: 모양 계약을 적습니다.**
식별한 연산에 대해 기대 shape를 명시적으로 적습니다:
```
matmul(A, B): A is (..., m, k), B is (..., k, n) -> (..., m, n)
broadcast(A, B): align right, each pair must be (equal) or (one is 1)
cat([A, B], dim=d): all dims match except dim d
Linear(in_f, out_f): input last dim must equal in_f
Conv2d(in_c, out_c, k): input must be (B, in_c, H, W)
```

**Step 3: 불일치를 찾습니다.**
실제 shape를 계약과 비교합니다. 규칙을 위반한 정확한 차원을 식별합니다.

**Step 4: 최소 수정을 고릅니다.**
이 표에서 고르세요:

| Symptom | Fix |
|---|---|
| Missing batch dimension | `.unsqueeze(0)` |
| Missing channel dimension | `.unsqueeze(1)` |
| Extra size-1 dimension | `.squeeze(dim)` |
| Inner dims wrong for matmul | `.transpose(-1, -2)` or check weight shape |
| Need NCHW from NHWC | `.permute(0, 3, 1, 2)` |
| Need NHWC from NCHW | `.permute(0, 2, 3, 1)` |
| Flatten spatial dims for linear | `.flatten(1)` or `.reshape(B, -1)` |
| Split heads: (B,T,D) to (B,H,T,D/H) | `.reshape(B, T, H, D//H).transpose(1, 2)` |
| Merge heads: (B,H,T,D/H) to (B,T,D) | `.transpose(1, 2).reshape(B, T, H*(D//H))` |
| Non-contiguous tensor with .view() | `.contiguous().view(...)` or use `.reshape(...)` |

**Step 5: 수정을 검증합니다.**
각 단계의 결과 shape를 보여 줍니다. reshape에서 총 원소가 보존되는지 확인합니다. 연산의 모양 계약이 이제 만족되는지 확인합니다.

**Step 6: 조용한 버그를 확인합니다.**
모양이 맞아도 다음을 검증하세요:
- 브로드캐스팅이 의도한 축을 따라 일어나는지 (우연히가 아닌지)
- 축소가 올바른 차원에 대해 합산하는지
- 배치 차원(dim 0)이 전체 순전파에서 살아남는지
- 차원 순서가 중요할 때 reshape만이 아니라 transpose + reshape를 쓰는지

응답 형식:
```
OPERATION: [실패한 연산]
EXPECTED: [모양 계약]
ACTUAL: [제공된 shape]
MISMATCH: [어느 차원, 왜]
FIX: [정확한 코드]
RESULT: [수정 후 shape]
```
