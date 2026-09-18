---
name: skill-ctc-decoder
description: 길이 정규화를 포함해 그리디와 빔 서치 CTC 디코더를 처음부터 작성
version: 1.0.0
phase: 4
lesson: 19
tags: [ocr, ctc, decoding, sequence-models]
---

# CTC Decoder

CTC 출력을 위한 두 디코딩 루틴을 만듭니다. 그리디(빠름)와 빔(노이즈 입력에서 더 나음).

## When to use

- 커스텀 CRNN 출력으로 OCR 추론을 돌릴 때.
- 사전학습 OCR 모델을 다른 디코더와 벤치마크할 때.
- ctcdecode를 끌어오지 않고 단순 빔 서치를 구현할 때.

## Inputs

- `log_probs`: (T, N, C) 어휘에 대한 log-softmax (관례상 인덱스 0 = blank).
- `vocab`: C개 문자 리스트.
- `beam_width` (빔만): 보통 5–10.

## Greedy decoder

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

## Beam search decoder

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
                            # Case 1: stay on same prefix (collapse from p_nb)
                            upd = new_beams.get(prefix, (-math.inf, -math.inf))
                            new_beams[prefix] = (upd[0], _logsumexp(upd[1], p_nb + p))
                            # Case 2: extend prefix via blank-separated repeat ("a_a" -> "aa")
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

## Rules

- CTC에서 blank 인덱스는 PyTorch `nn.CTCLoss` 관례상 0입니다.
- 빔 서치는 저신뢰 입력에서 정확도를 올립니다. 깨끗한 입력에서 개선은 CER <1%입니다.
- 빔을 5 아래로 가지치기하지 마세요. 그 아래에서 정확도-지연 트레이드가 평평해집니다.
- 빡빡한 지연 예산 안에서 빔 서치를 돌릴 때는 그리디로 내리세요. 대부분 프로덕션 OCR 데이터에서 품질 타격은 작습니다.
- 큰 어휘(CJK 3000+ 문자)에서는 위 순수 Python 버전 대신 `ctcdecode`(C++)로 바꾸세요. Python 빔이 빠르게 병목이 됩니다.
