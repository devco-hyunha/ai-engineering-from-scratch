---
name: skill-recall-at-k-runner
description: train/val/gallery 분할과 적절한 데이터 계약으로 recall@K를 위한 깔끔한 평가 하네스를 작성합니다
version: 1.0.0
phase: 4
lesson: 20
tags: [retrieval, evaluation, recall, faiss]
---

# Recall@K Runner

쿼리 및 갤러리 이미지와 레이블이 포함된 폴더를 재현 가능한 recall@K 수치로 변환합니다.

## 사용 시점

- 새 백본(backbone)에 대한 첫 번째 검색 벤치마크.
- 미세 조정(Fine-tuning) 에포크(Epoch)에 걸쳐 임베딩(Embedding) 품질을 추적합니다.
- 동일한 데이터셋에서 두 검색 시스템을 비교합니다.

## 입력

- `query_images`: 경로 목록.
- `gallery_images`: 경로 목록 (쿼리가 겹칠 수도 있고 겹치지 않을 수도 있습니다).
- `query_labels`, `gallery_labels`: 클래스 또는 인스턴스 ID.
- `encoder_fn`: 호출 가능한 `image -> embedding` (사전 계산된 값 또는 실시간 값).
- `ks`: `[1, 5, 10]`와 같은 목록.

## 단계

1. 모든 갤러리 이미지를 한 번만 인코딩합니다. numpy 배열로 저장합니다.
2. 모든 쿼리 이미지를 인코딩합니다.
3. 두 임베딩(Embedding) 집합 모두를 L2 정규화(Normalization)합니다.
4. 각 쿼리에 대해 모든 갤러리 항목과의 유사도를 계산합니다.
5. 내림차순으로 정렬하고 top max(ks)를 가져옵니다.
6. 각 K에 대해, top-K 갤러리 항목 중 쿼리의 레이블과 공유하는 항목이 있는지 확인합니다.
7. `recall@K = fraction of queries that had at least one correct neighbour in top K`을 보고합니다.

## 출력 템플릿

```python
import numpy as np
from sklearn.preprocessing import normalize

def encode_all(images, encoder_fn, batch=32):
    out = []
    for i in range(0, len(images), batch):
        embs = encoder_fn(images[i:i + batch])
        out.append(embs)
    return np.concatenate(out)


def recall_at_k(query_emb, gallery_emb, q_labels, g_labels,
                ks=(1, 5, 10), query_ids=None, gallery_ids=None):
    if len(query_emb) == 0 or len(gallery_emb) == 0:
        return {f"recall@{k}": 0.0 for k in ks}

    g_label_set = set(g_labels.tolist())
    keep = np.array([lbl in g_label_set for lbl in q_labels])
    if not keep.any():
        return {f"recall@{k}": 0.0 for k in ks}

    q_emb_f = query_emb[keep]
    q_lab_f = q_labels[keep]
    q_id_f = query_ids[keep] if query_ids is not None else None

    q = normalize(q_emb_f)
    g = normalize(gallery_emb)
    sims = q @ g.T

    if q_id_f is not None and gallery_ids is not None:
        self_mask = q_id_f[:, None] == gallery_ids[None, :]
        sims = np.where(self_mask, -np.inf, sims)

    top_k_max = min(max(ks), g.shape[0])
    if top_k_max <= 0:
        return {f"recall@{k}": 0.0 for k in ks}

    top = np.argpartition(-sims, top_k_max - 1, axis=1)[:, :top_k_max]
    sorted_top = np.take_along_axis(
        top, np.argsort(-sims[np.arange(len(q))[:, None], top], axis=1), axis=1
    )
    out = {}
    for k in ks:
        k_eff = min(k, top_k_max)
        hits = np.any(g_labels[sorted_top[:, :k_eff]] == q_lab_f[:, None], axis=1)
        out[f"recall@{k}"] = float(hits.mean())
    return out


def evaluate(query_images, query_labels, gallery_images, gallery_labels, encoder_fn, ks=(1, 5, 10)):
    q_emb = encode_all(query_images, encoder_fn)
    g_emb = encode_all(gallery_images, encoder_fn)
    return recall_at_k(q_emb, g_emb, np.array(query_labels), np.array(gallery_labels), ks)
```

## 보고

```
[evaluation]
  num queries:   <int>
  num gallery:   <int>
  embedding_dim: <int>

[recall]
  recall@1:  <float>
  recall@5:  <float>
  recall@10: <float>
```

## 규칙

- 유사도를 계산하기 전에 임베딩(Embedding)을 정규화(Normalization)합니다. 정규화된 벡터에 대한 FAISS IndexFlatIP는 코사인 유사도(Cosine Similarity)와 같습니다.
- 쿼리의 ground-truth 레이블이 갤러리에 없는 경우, 이를 제외합니다. 그렇지 않으면 recall이 1 미만으로 자명하게 상한이 걸립니다.
- 쿼리와 갤러리가 겹치는 경우, 쿼리 자신을 자신의 top-K에서 제외합니다. 그렇지 않으면 검색이 아닌 자기 유사도를 측정하게 됩니다.
- `num_queries > 10,000`의 경우, OOM(Out Of Memory)를 피하기 위해 유사도 행렬 곱을 배치(batch) 처리합니다.
