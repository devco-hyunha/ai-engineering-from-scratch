---
name: skill-cmer-monitor
description: Cross-Modal Error Rate 모니터링·대시보드·알림으로 프로덕션 VLM 엔드포인트를 계측합니다
version: 1.0.0
phase: 4
lesson: 25
tags: [vlm, production, monitoring, hallucination]
---

# CMER 모니터 (CMER Monitor)

교차 모달 정렬을 일급 프로덕션 KPI로 취급합니다.

## 언제 쓰나요 (When to use)

- 이미지에 그라운딩된 텍스트를 내는 모든 VLM 엔드포인트를 배포할 때.
- 환각 응답 보고를 조사할 때.
- 입력 분포 이동이 모델 그라운딩을 악화시키는지 추적할 때.

## 입력 (Inputs)

- `vlm_output`: 생성된 텍스트.
- `text_confidence`: softmax 후 토큰당 확률의 평균, `[0, 1]`. `exp(mean(log_probs))`로 계산하세요. 원시 로짓을 넘기지 마세요; 원시 로짓은 비한정이고 `conf_threshold`는 확률을 가정합니다.
- `image_embedding`: 이미지의 CLIP 계열 임베딩(DINOv3, SigLIP, CLIP).
- `text_embedding`: 생성 텍스트의 CLIP 계열 임베딩.
- 선택적 `prompt_type`: 그룹화용 라벨(vqa / ocr / captioning / agent).

## 요청당 계산 (Per-request computation)

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

임베딩은 독립 CLIP 계열 인코더의 1-D PyTorch 텐서(`torch.float32`)입니다. NumPy 배열을 쓰면 `.norm()`을 `np.linalg.norm(...)`로 바꾸고 출력을 그에 맞게 캐스트하세요.

`sim`, `text_conf`, `flagged`, `prompt_type`, `timestamp`, `model_version`, `request_id`를 모니터링 파이프라인(Prometheus, DataDog, OpenTelemetry)에 저장하세요.

## 집계 지표 (Aggregate metric)

```
CMER = (flagged requests in window) / (total requests in window)
```

엔드포인트별, prompt_type별, 모델 버전별로 보고하세요.

## 알림 임계값 (Alert thresholds)

- 기준 CMER: 정상 트래픽 7일로 확립.
- 경고: 1시간 동안 CMER >= 기준의 1.5배.
- 심각: 30분 동안 CMER >= 기준의 2배, 또는 어느 윈도우든 절대값 > 15%.

## 대시보드 패널 (Dashboard panels)

1. 시간에 따른 CMER(5분 버킷, 7일 윈도우).
2. prompt_type별 CMER(스택 바).
3. 시간당 `sim` 분포(히스토그램).
4. 상위 환각 출력(사람 검토용으로 하루 플래그된 응답 20개 샘플).

## CMER가 급등할 때 행동 (Actions when CMER spikes)

1. 플래그된 요청을 샘플링합니다.
2. 모델 버전이 의도치 않게 바뀌지 않았는지 확인합니다.
3. 입력 분포를 확인합니다(새 파일 형식? 새 이미지 소스? 다르게 압축?).
4. 급등이 해소될 때까지 영향 트래픽을 사람 검토로 라우팅합니다.
5. 급등이 지속되면 모델을 파인튜닝하거나 교체하세요; 알림을 억누르지 마세요.

## 규칙 (Rules)

- VLM 자체의 임베딩으로 CMER를 절대 계산하지 마세요; 독립 인코더(DINOv3, SigLIP, 또는 CLIP-L/14)를 쓰세요. 그렇지 않으면 정렬이 아니라 모델의 자기 일관성을 측정합니다.
- `flagged` 비트만이 아니라 원시 `sim` 값을 항상 로깅하세요; 분포 이동은 플래그율이 바뀌기 전에 하위 사분위수에 나타납니다.
- CMER 모니터링 없이 VLM 엔드포인트를 배포하지 마세요; 환각이 지배적 프로덕션 실패 모드이며 이 지표 없이는 조용합니다.
- 민감 도메인(의료, 법률, 금융)에서는 `sim_threshold`를 0.35 이상으로 올리세요; 플래그 조건은 `sim < sim_threshold`이므로 더 높은 임계값이 잠재적으로 그라운딩되지 않은 출력을 더 많이 잡습니다 — 고위험 용도의 올바른 기본값입니다.
