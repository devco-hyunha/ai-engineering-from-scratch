---
name: skill-vit-patch-and-pos-embed-inspector
description: ViT의 패치 임베딩과 위치 임베딩 shape가 모델의 예상 시퀀스 길이와 맞는지 검증합니다
version: 1.0.0
phase: 4
lesson: 14
tags: [vision-transformer, debugging, pytorch]
---

# ViT 패치·위치 임베딩 검사기 (ViT Patch and Positional Embedding Inspector)

가장 흔한 ViT 포팅 버그: 224x224에서 사전학습된 체크포인트를 384x384(또는 그 반대)로 설정된 모델에 로드하는 것. 위치 임베딩의 시퀀스 길이가 틀려 모델이 조용히 쓰레기를 만듭니다.

## 언제 쓰나요 (When to use)

- 기본이 아닌 해상도에서 사전학습 ViT를 파인튜닝할 때.
- ViT-B/16과 ViT-B/32 사이 가중치 포트가 실패하는 이유를 감사할 때; 검사기가 패치 크기 불일치를 플래그해 호출자가 강제 포트 대신 아키텍처를 바꾸도록 알립니다.
- 오류 없이 로드되지만 학습이 나쁜 ViT를 디버깅할 때.

## 입력 (Inputs)

- `model`: 인스턴스화된 ViT `nn.Module`.
- `expected_image_size`: 프로덕션에서 모델이 볼 H x W.
- `patch_size`: 예상 패치 크기.

## 단계 (Steps)

1. 모델 안의 패치 임베딩 conv를 찾으세요. `kernel_size`, `stride`, `in_channels`, `out_channels`를 보고하세요.
2. 예상 패치 수를 계산하세요. 정사각 이미지: `(image_size / patch_size)^2`. 직사각: `(H / patch_size) * (W / patch_size)`. `H % patch_size == 0`과 `W % patch_size == 0`을 요구하세요; 아니면 플래그하고 거부하세요.
3. 학습된 위치 임베딩을 찾으세요. shape `(1, N, dim)`을 보고하세요.
4. `N`을 `num_patches + 1`(CLS 있음) 또는 `num_patches`(CLS 없음)와 비교하세요. 불일치는 체크포인트가 다른 해상도나 패치 크기에서 사전학습되었음을 뜻합니다.
5. 패치 conv의 `out_channels`가 위치 임베딩의 `dim`과 같은지 확인하세요.
6. 모델이 새 해상도에 대해 위치 임베딩을 보간해야 한다면, 보간 유틸이 있는지 확인하세요(대부분의 `timm` ViT는 `resize_pos_embed`로 자동 처리).

## 보고 (Report)

```
[vit-inspector]
  image_size:         HxW
  patch_size:         <int>
  num_patches (computed): <int>
  patch_conv:         k=<int>  s=<int>  in=<int>  out=<int>
  pos_embed shape:    (1, N, dim)
  has CLS token:      yes | no
  pos_embed N:        <int>    expected: <int>
  verdict:            ok | mismatch

[if mismatch]
  action:  reinitialise pos_embed for new sequence length
  tool:    timm.models.vision_transformer.resize_pos_embed
```

## 규칙 (Rules)

- 경고 없이 조용히 보간하지 마세요; 사용자가 사전학습된 위치 구조가 바뀌었을 수 있음을 알도록 동작을 드러내세요.
- patch_size가 불일치하면 보간을 권하지 마세요 — 올바른 아키텍처로 바꾸세요.
- 모델을 제자리에서 고치려 하지 마세요; 보고하고 제안하세요.
