---
name: skill-ctc-decoder
description: 길이 정규화를 포함하여 그리디 및 빔 검색 CTC 디코더를 처음부터 작성합니다
version: 1.0.0
phase: 4단계
lesson: 19강
tags: [ocr, ctc, decoding, sequence-models]
---

# CTC 디코더

CTC 출력에 대해 두 가지 디코딩 루틴을 생성합니다: 그리디 (빠름) 및 빔 (잡음이 있는 입력에서 더 좋음).

## 사용 시점

- 커스텀 CRNN 출력에 OCR 추론을 실행할 때.
- 사전 학습된 OCR 모델을 다양한 디코더와 비교 벤치마킹할 때.
- ctcdecode를 가져오지 않고 간단한 빔 검색을 구현할 때.

## 입력

- `log_probs`: 어휘에 대한 (T, N, C) 로그 소프트맥스 (관례에 따라 인덱스 0 = 공백).
- `vocab`: C개의 문자 목록.
- `beam_width` (빔 전용): 일반적으로 5-10.

## 그리디 디코더

```python
def greedy_ctc_decode(log_probs, vocab, blank=0):
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(vocab[idx])
            prev = idx
        out.append("".join(decoded))
    return out
```

## 빔 검색 디코더

```python
import heapq
import math

def beam_ctc_decode(log_probs, vocab, beam_width=5, blank=0):
    T, N, C = log_probs.shape
    lp = log_probs.cpu()
    results = []
    for n in range(N):
        beams = {("",): (0.0, -math.inf)}  # (prefix_tuple) -> (p_blank, p_nonblank)
        for t in range(T):
            logits_t = lp[t, n]
            new_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                for c in range(C):
                    p = logits_t[c].item()
                    if c == blank:
                        nb = p_b + p
                        nnb = p_nb + p
                        upd = new_beams.get(prefix, (-math.inf, -math.inf))
                        new_beams[prefix] = (
                            _logsumexp(upd[0], _logsumexp(nb, nnb)),
                            upd[1],
                        )
                    else:
                        last = prefix[-1] if prefix else ""
                        char = vocab[c]
                        if char == last:
                            # 케이스 1: 같은 접두어에 머무름 (p_nb에서 축소)
                            upd = new_beams.get(prefix, (-math.inf, -math.inf))
                            new_beams[prefix] = (upd[0], _logsumexp(upd[1], p_nb + p))
                            # 케이스 2: 공백으로 구분된 반복을 통해 접두어 확장 ("a_a" -> "aa")
                            new_prefix = prefix + (char,)
                            upd = new_beams.get(new_prefix, (-math.inf, -math.inf))
                            new_beams[new_prefix] = (upd[0], _logsumexp(upd[1], p_b + p))
                        else:
                            new_prefix = prefix + (char,)
                            upd = new_beams.get(new_prefix, (-math.inf, -math.inf))
                            nb = _logsumexp(p_b, p_nb) + p
                            new_beams[new_prefix] = (upd[0], _logsumexp(upd[1], nb))
            beams = dict(heapq.nlargest(
                beam_width,
                new_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
            ))
        best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
        results.append("".join(best))
    return results


def _logsumexp(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
```

## 규칙

- PyTorch의 `nn.CTCLoss`에서 CTC의 공백 인덱스는 관례적으로 0입니다.
- 빔 검색은 신뢰도가 낮은 입력에서 정확도를 향상시킵니다. 깨끗한 입력에서는 개선 효과가 CER 1% 미만입니다.
- 빔을 5 미만으로 가지 치기(prune)하지 마세요. 그 이하에서는 정확도-지연 시간 트레이드오프가 평탄해집니다.
- 엄격한 지연 시간 예산 내에서 빔 검색을 실행할 때는 그리디로 전환하세요. 대부분의 프로덕션 OCR 데이터에서는 품질 저하가 미미합니다.
- 대규모 어휘 (3000자 이상의 CJK)의 경우, 위의 순수 Python 버전 대신 `ctcdecode` (C++)로 전환하세요. Python 빔은 금방 병목 현상이 됩니다.
