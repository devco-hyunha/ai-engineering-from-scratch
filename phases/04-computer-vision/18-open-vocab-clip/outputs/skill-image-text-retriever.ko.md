---
name: skill-image-text-retriever
description: 임의의 CLIP 체크포인트로 이미지 임베딩 인덱스를 구축합니다. 텍스트 기반 및 이미지 기반 쿼리를 지원합니다.
version: 1.0.0
phase: 4단계
lesson: 18강
tags: [clip, retrieval, faiss, zero-shot]
---

# 이미지-텍스트 리트리버

CLIP 임베딩을 사용하여 이미지 폴더를 검색 가능한 인덱스로 변환합니다.

## 사용 시점

- 내부 카탈로그에 제로샷 이미지 검색을 구축할 때.
- 임베딩 거리를 통해 거의 동일한 이미지를 중복 제거할 때.
- 레이블이 지정된 데이터셋 없이 빠른 "유사 이미지 찾기" 컴포넌트를 구축할 때.

## 입력

- `image_folder`: 이미지 파일이 있는 디렉토리.
- `clip_model`: `openai/clip-vit-base-patch32` 또는 `google/siglip-base-patch16-224`와 같은 HuggingFace ID.
- `index_type`: flat | IVF | HNSW.
- `embedding_dim`: 모델에서 추론됩니다.

## 단계

1. CLIP 모델과 전처리기를 로드합니다.
2. 폴더 내의 모든 이미지를 배치로 인코딩합니다. 임베딩을 (N, D) float32 및 파일명 목록으로 저장합니다.
3. 임베딩 위에 FAISS 인덱스를 구축합니다. 코사인 유사도를 위해 L2 정규화된 벡터에 내적(inner-product)을 사용합니다.
4. 두 가지 쿼리 인터페이스를 노출합니다:
   - `search_by_text(text, k)` — 텍스트를 임베딩하고 검색합니다.
   - `search_by_image(image_path, k)` — 이미지를 임베딩하고 검색합니다.

## 출력 템플릿

```python
import os
import glob
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import faiss


class ImageTextRetriever:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim
        self.index = None
        self.filenames = []

    @torch.no_grad()
    def _encode_images(self, paths, batch=16):
        embs = []
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inputs = self.processor(images=imgs, return_tensors="pt")
            out = self.model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
            embs.append(out.cpu().numpy())
        return np.concatenate(embs).astype(np.float32)

    @torch.no_grad()
    def _encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        out = self.model.get_text_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
        return out.cpu().numpy().astype(np.float32)

    def build_index(self, folder, index_type="flat"):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        self.filenames = sorted(files)
        embs = self._encode_images(self.filenames)
        if index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = min(256, max(4, len(embs) // 32))
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.index.train(embs)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

    def search_by_text(self, text, k=5):
        q = self._encode_text([text])
        dist, idx = self.index.search(q, k)
        return [(self.filenames[i], float(d)) for d, i in zip(dist[0], idx[0])]

    def search_by_image(self, image_path, k=5):
        q = self._encode_images([image_path])
        dist, idx = self.index.search(q, k)
        return [(self.filenames[i], float(d)) for d, i in zip(dist[0], idx[0])]
```

## 보고서

```
[retriever]
  model:          <name>
  num_images:     <int>
  dim:            <int>
  index_type:     flat | IVF | HNSW
  index_size_mb:  <float>
```

## 규칙

- 인덱싱 전에 항상 임베딩을 L2 정규화하세요. 정규화된 벡터에 대한 FAISS의 내적은 코사인 유사도와 같습니다.
- 10만 장 미만의 이미지인 경우, `IndexFlatIP` (정확 검색)이 가장 단순하고 빠릅니다.
- 10만~1000만 장인 경우, `IndexIVFFlat`이 표준적인 트레이드오프입니다.
- 1000만 장 이상인 경우, HNSW 또는 제품 양자화(product-quantised) 변형을 사용하세요.
- 매 쿼리마다 인덱스를 재구축하지 마세요. 한 번 임베딩하고 여러 번 검색하세요.
