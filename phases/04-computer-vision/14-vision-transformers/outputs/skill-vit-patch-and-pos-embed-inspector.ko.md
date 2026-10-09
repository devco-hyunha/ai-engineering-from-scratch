---
name: skill-vit-patch-and-pos-embed-inspector
description: ViT의 패치 임베딩 및 위치 임베딩 형상이 모델의 예상 시퀀스 길이와 일치하는지 검증합니다.
version: 1.0.0
phase: 4
lesson: 14
tags: [vision-transformer, debugging, pytorch]
---

# ViT 패치 및 위치 임베딩 검사기

가장 흔한 ViT 이식 버그는 224x224로 사전 학습된 체크포인트를 384x384로 설정된 모델에 로드하는 경우(또는 그 반대)입니다. 위치 임베딩의 시퀀스 길이가 잘못되어 모델이 조용히 엉뚱한 결과를 생성합니다.

## 사용 시점

- 기본 해상도가 아닌 해상도로 사전 학습된 ViT를 미세 조정할 때.
- ViT-B/16강 ViT-B/32 사이의 가중치 이식이 실패하는 이유를 감사할 때; 검사기가 패치 크기 불일치를 표시하므로 호출자는 강제로 이식하는 대신 아키텍처를 교체해야 함을 알 수 있습니다.
- 오류 없이 로드되지만 학습이 잘되지 않는 ViT를 디버깅할 때.

## 입력

- `model`: 인스턴스화된 ViT `nn.Module`.
- `expected_image_size`: 모델이 프로덕션에서 볼 H x W.
- `patch_size`: 예상 패치 크기.

## 단계

1. 모델 내부의 패치 임베딩 컨브를 찾습니다. `kernel_size`, `stride`, `in_channels`, `out_channels`을 보고합니다.
2. 예상 패치 수를 계산합니다. 정사각형 이미지인 경우: `(image_size / patch_size)^2`. 직사각형인 경우: `(H / patch_size) * (W / patch_size)`. `H % patch_size == 0` 및 `W % patch_size == 0`을 요구합니다; 그렇지 않으면 표시하고 거부합니다.
3. 학습된 위치 임베딩을 찾습니다. 형상 `(1, N, dim)`을 보고합니다.
4. `N`을 `num_patches + 1`(CLS 포함) 또는 `num_patches`(CLS 제외)와 비교합니다. 불일치는 체크포인트가 다른 해상도나 패치 크기로 사전 학습되었음을 의미합니다.
5. 패치 컨브의 `out_channels`이 위치 임베딩의 `dim`과 동일한지 확인합니다.
6. 모델이 새로운 해상도를 위해 위치 임베딩을 보간해야 한다면, 보간 유틸리티가 존재하는지 확인합니다(대부분의 `timm` ViT는 `resize_pos_embed`을 통해 자동으로 수행합니다).

## 보고

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

## 규칙

- 경고 없이 조용히 보간하지 마세요; 사전 학습된 위치 구조가 이동했을 수 있으므로 사용자가 알 수 있도록 조치를 표시하세요.
- patch_size가 일치하지 않으면 보간을 권장하지 마세요. 올바른 아키텍처로 교체하세요.
- 모델을 제자리에서 수정하려 하지 마세요. 보고하고 제안하세요.
