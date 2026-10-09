---
name: skill-cmer-monitor
description: Cross-Modal Error Rate 모니터링, 대시보드 및 경보로 프로덕션 VLM 엔드포인트를 계측합니다
version: 1.0.0
phase: 4단계
lesson: 25강
tags: [vlm, production, monitoring, hallucination]
---

# CMER 모니터

교차 모달 정렬을 프로덕션 KPI의 일급 지표로 취급해 보세요.

## 사용 시점

- 이미지를 기반으로 텍스트를 생성하는 모든 VLM 엔드포인트를 배포할 때.
- 환각(Hallucination) 응답에 대한 보고를 조사할 때.
- 입력 분포 이동(Distribution Shift)이 모델 그라운딩(Grounding)을 저하시키는지 추적할 때.

## 입력

- `vlm_output`: 생성된 텍스트.
- `text_confidence`: 소프트맥스(Softmax) 이후의 토큰별 평균 확률, `[0, 1]` 범위. `exp(mean(log_probs))`로 계산합니다. 원시 로짓(Raw Logits)을 전달하지 마세요. 원시 로짓은 범위가 무한하며 `conf_threshold`는 확률을 가정합니다.
- `image_embedding`: 이미지의 CLIP 계열 임베딩(Embedding) (DINOv3, SigLIP, CLIP).
- `text_embedding`: 생성된 텍스트의 CLIP 계열 임베딩(Embedding).
- 선택적 `prompt_type`: 그룹화용 레이블 (vqa / ocr / captioning / agent).

## 요청별 계산

```python
import torch

def cmer_flag(image_emb, text_emb, text_conf, sim_thr=0.25, conf_thr=0.8):
    if image_emb.shape != text_emb.shape:
        raise ValueError(f"emb shape mismatch: {image_emb.shape} vs {text_emb.shape}")
    image_emb = image_emb / (image_emb.norm() + 1e-8)
    text_emb = text_emb / (text_emb.norm() + 1e-8)
    sim = float((image_emb * text_emb).sum())
    flagged = (text_conf > conf_thr) and (sim < sim_thr)
    return {"sim": sim, "flagged": flagged}
```

임베딩(Embedding)은 독립적인 CLIP 계열 인코더(Encoder)에서 나온 1-D PyTorch 텐서(Tensor) (`torch.float32`)입니다. NumPy 배열을 사용한다면 `.norm()`를 `np.linalg.norm(...)`로 교체하고 출력에 맞게 캐스팅하세요.

`sim`, `text_conf`, `flagged`, `prompt_type`, `timestamp`, `model_version`, `request_id`를 모니터링 파이프라인(Prometheus, DataDog, OpenTelemetry)에 저장하세요.

## 집계 지표

```
CMER = (flagged requests in window) / (total requests in window)
```

엔드포인트별, prompt_type별, 모델 버전별로 보고하세요.

## 경보 임계값

- 기준 CMER: 정상 트래픽으로 7일 동안 설정합니다.
- 경고: 1시간 동안 CMER가 기준의 1.5배 이상일 때.
- 치명적: 30분 동안 CMER가 기준의 2배 이상일 때, 또는 어떤 윈도우에서도 절대값이 15% 이상일 때.

## 대시보드 패널

1. 시간에 따른 CMER (5분 버킷, 7일 윈도우).
2. prompt_type별 CMER (누적 막대).
3. 시간별 `sim` 분포 (히스토그램).
4. 상위 환각(Hallucination) 출력 (하루에 플래그가 지정된 응답 20개를 샘플링하여 인간이 검토합니다).

## CMER 급증 시 조치

1. 플래그가 지정된 요청을 샘플링하세요.
2. 모델 버전이 의도치 않게 변경되지 않았는지 확인하세요.
3. 입력 분포를 점검하세요 (새로운 파일 형식? 새로운 이미지 소스? 압축 방식이 변경되었는지?).
4. 급증이 해소될 때까지 영향을 받은 트래픽을 인간 검토로 라우팅하세요.
5. 급증이 지속되면 모델을 미세 조정하거나 교체하세요. 경보를 억제하지 마세요.

## 규칙

- VLM 자체의 임베딩을 사용하여 CMER를 계산하지 마세요. 독립적인 인코더(DINOv3, SigLIP, 또는 CLIP-L/14)를 사용하세요. 그렇지 않으면 모델의 자기 일관성을 측정하는 것이지, 정렬을 측정하는 것이 아닙니다.
- `flagged` 비트뿐만 아니라 원시 `sim` 값을 항상 로깅하세요. 분포 이동은 플래그 비율이 변경되기 전에 하위 사분위에서 나타납니다.
- CMER 모니터링 없이 VLM 엔드포인트를 출시하지 마세요. 환각은 생산 환경에서의 주요 실패 모드이며, 이 지표 없이는 조용히 진행됩니다.
- 민감한 영역(의료, 법률, 금융)에서는 `sim_threshold`을 0.35 이상으로 높이세요. 플래그 조건은 `sim < sim_threshold`이므로, 더 높은 임계값은 더 많은 출력을 잠재적으로 그라운딩되지 않은 것으로 포착합니다. 이는 고위험 사용 사례에 대한 올바른 기본값입니다.
