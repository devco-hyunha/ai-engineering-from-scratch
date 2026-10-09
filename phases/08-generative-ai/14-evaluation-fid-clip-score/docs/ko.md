# 평가 — FID, CLIP 점수, 인간 선호도

> 모든 생성형 모델 리더보드는 FID, CLIP 점수, 그리고 인간 선호도 아레나에서의 승률을 인용합니다. 각 수치에는 집요한 연구자가 조작할 수 있는 실패 모드(failure mode)가 존재합니다. 실패 모드를 알지 못하면, 진정한 개선과 조작된 결과를 구분할 수 없습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 8단계 · 01강 (분류 체계), 2단계 · 04강 (평가 지표)
**시간:** 약 45분

## 문제점

생성형 모델은 *샘플 품질*과 *조건 준수(conditioning adherence)*로 평가됩니다. 둘 다 폐쇄형(closed-form) 측정 지표가 없습니다. 모델이 10,000개의 이미지를 렌더링해야 하고, 누군가는 그것에 숫자를 부여해야 하며, 모델 계열, 해상도, 아키텍처를 넘나들며 그 숫자를 신뢰해야 합니다. 2014-2026년의 검증 과정을 통과한 세 가지 지표가 있습니다:

- **FID (Fréchet Inception Distance).** Inception 네트워크의 특징(feature) 공간에서 두 분포 — 실제와 생성된 것 — 사이의 거리입니다. 낮을수록 좋습니다.
- **CLIP 점수.** 생성된 이미지의 CLIP 이미지 임베딩과 프롬프트의 CLIP 텍스트 임베딩 간의 코사인 유사도(Cosine Similarity)입니다. 높을수록 좋습니다. 프롬프트 준수를 측정합니다.
- **인간 선호도.** 동일한 프롬프트로 두 모델을 맞붙이고, 인간(또는 GPT-4급 모델)이 더 나은 것을 선택하게 한 후, Elo 점수로 집계합니다.

또한 IS (Inception Score, 대부분 폐기됨), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k도 볼 수 있습니다. 각각이 이전 지표의 한 가지 실패를 보완합니다.

## 개념

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — 샘플 품질

Heusel et al. (2017). 단계:

1. N개의 실제 이미지와 N개의 생성된 이미지에 대해 Inception-v3 특징(2048-D)을 추출합니다.
2. 각 풀(pool)에 가우시안을 적합(fit)합니다: 평균 `μ_r, μ_g` 및 공분산 `Σ_r, Σ_g`을 계산합니다.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

해석: 특징 공간에서 두 다변량 가우시안 간의 Fréchet 거리입니다. 낮을수록 분포가 더 유사합니다.

실패 모드:
- **작은 N에 대해 편향됨.** FID는 특징 분포에 대한 평균 제곱 오차입니다 — 작은 N은 공분산을 과소평가하여 FID를 허위하게 낮게 만듭니다. 항상 N ≥ 10,000을 사용하세요.
- **Inception 의존적.** Inception-v3는 ImageNet으로 학습되었습니다. ImageNet과 거리가 먼 도메인(얼굴, 예술, 텍스트 이미지)은 의미 없는 FID를 생성합니다. 도메인 전용 특징 추출기를 사용하세요.
- **게임을 이용하기.** Inception 사전에 과적합하면 시각적 품질 개선 없이 낮은 FID를 얻을 수 있습니다. CMMD(아래 참조)로 이를 극복하세요.

### CLIP 점수 — 프롬프트 준수

Radford et al. (2021). 생성된 이미지 + 프롬프트에 대해:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

30,000개의 생성된 이미지에 대해 평균을 내면 모델 간에 비교 가능한 스칼라 값이 됩니다.

실패 모드:
- **CLIP 자체의 맹점.** CLIP은 구성적 추론이 약합니다("파란 구슬 위에 빨간 큐브"는 종종 실패합니다). 모델은 복잡한 프롬프트를 실제로 따르지 않더라도 CLIP 점수에서 잘 순위를 매길 수 있습니다.
- **짧은 프롬프트 편향.** 짧은 프롬프트는 야생(wild)에서 CLIP-이미지 매치가 더 많습니다. 긴 프롬프트는 기계적으로 CLIP 점수가 낮습니다.
- **프롬프트 조작.** 프롬프트에 "high quality, 4k, masterpiece"를 포함하면 이미지-텍스트 바인딩을 개선하지 않고도 CLIP 점수를 부풀릴 수 있습니다.

CMMD (Jayasumana et al., 2024)는 이러한 문제 중 일부는 해결합니다: Inception 대신 CLIP 특징을 사용하며, Fréchet 대신 최대 평균 불일치(maximum-mean discrepancy)를 사용합니다. 미묘한 품질 차이를 감지하는 데 더 능숙합니다.

### 인간 선호도 — 기준 진실(ground truth)

프롬프트 풀을 선택하세요. 모델 A와 모델 B로 생성하세요. 쌍을 인간(또는 강력한 LLM 심사위원)에게 보여주세요. 승리를 Elo 또는 Bradley-Terry 점수로 집계하세요. 벤치마크:

- **PartiPrompts (Google)**: 1,600개의 다양한 프롬프트, 12개 카테고리.
- **HPSv2**: 107,000개의 인간 주석, 자동화된 대리 지표로 널리 사용됨.
- **ImageReward**: 137,000개의 프롬프트-이미지 선호 쌍, MIT 라이선스.
- **PickScore**: Pick-a-Pic의 260만 선호도 데이터로 학습됨.
- **Chatbot-Arena 스타일의 이미지 아레나**: https://imagearena.ai/ 및 기타.

실패 모드:
- **심사위원의 편차.** 비전문가는 전문가와 다른 선호도를 가집니다. 둘 다 사용하세요.
- **프롬프트 분포.** 선별된(cherry-picked) 프롬프트는 특정 계열에 유리합니다. 항상 문서화하세요.
- **LLM 심사위원의 보상 조작(reward hacking).** GPT-4 심사위원은 예쁘지만 잘못된 출력에 속습니다. 인간과 삼각측량(triangulate)하세요.

## 함께 사용하세요

프로덕션 평가 보고서에는 다음이 포함되어야 합니다:

1. 보유한 실제 분포에 대한 10-30k 샘플의 FID (샘플 품질).
2. 동일한 샘플에 대해 프롬프트와 비교한 CLIP 점수 / CMMD (순응도).
3. 이전 모델과의 블라인드 아레나에서의 승률 (전체 선호도).
4. 실패 모드 분석: 알려진 문제(손 해부학, 텍스트 렌더링, 일관된 객체 수)에 대해 플래그가 지정된 50개의 랜덤 샘플링된 출력.

단일 지표는 거짓입니다. 세 개의 보조 지표와 정성적 검토가 주장입니다.

```figure
gx-fid-distributions
```

## 구현하기

`code/main.py`는 합성 "특징 벡터"(Inception 특징의 대체로 4-D 벡터를 사용)에 대해 FID, CLIP 점수 유사 및 Elo 집계 기능을 구현합니다. 다음을 볼 수 있습니다:

- 작은 N과 큰 N에서의 FID 계산 — 편향.
- 특징 풀 간 코사인 유사도로서의 "CLIP 점수".
- 합성 선호 스트림에서의 Elo 업데이트 규칙.

### 1단계: 네 줄의 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 2단계: CLIP 스타일 코사인 유사도

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 3단계: Elo 집계

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 함정

- **N=1000에서의 FID.** N=10k 미만에서는 휴리스틱이 신뢰할 수 없습니다. 낮은 N의 FID를 보고하는 논문은 조작입니다.
- **해상도 간 FID 비교.** Inception의 299×299 리사이즈는 특징 분포를 변경합니다. 일치하는 해상도에서만 비교하세요.
- **하나의 시드 보고.** 최소 3개의 시드를 실행하세요. 표준 편차를 보고하세요.
- **음수 프롬프트를 통한 CLIP 점수 팽창.** 일부 파이프라인은 프롬프트에 과적합하여 CLIP을 부풀립니다. 시각적 포화를 확인하세요.
- **프롬프트 중복으로 인한 Elo 편향.** 두 모델 모두 벤치마크 프롬프트를 학습 중에 봤다면 Elo는 무의미합니다. 보유 프롬프트 세트를 사용하세요.
- **유료 크라우드 인간 평가의 편향.** Prolific, MTurk annotators는 더 젊은 / 기술 친화적인 경향이 있습니다. 채용된 예술/디자인 전문가와 혼합하세요.

## 사용하기

2026년 프로덕션 평가 프로토콜:

| 기둥 | 최소 | 권장 |
|--------|---------|-------------|
| 샘플 품질 | 보유 실제에 대한 10k FID | + 5k CMMD + 카테고리별 하위 집합 FID |
| 프롬프트 순응도 | 30k CLIP 점수 | + HPSv2 + ImageReward + VQA 스타일 질문 답변 |
| 선호도 | 200 블라인드 쌍 vs 기준선 | + 2000 페어링된 인간 + LLM 심사 + Chatbot Arena |
| 실패 분석 | 50 수동 플래그 | 500 수동 플래그 + 자동화된 안전 분류기 |

네 가지 기둥 모두를 하나의 보고서에 담으면 주장이 됩니다. 하나만 담으면 마케팅이 됩니다.

## 출시하기

`outputs/skill-eval-report.md`을 저장하세요. 스킬은 새로운 모델 체크포인트와 기준선을 받아 전체 평가 계획(샘플 크기, 지표, 실패 모드 탐지, 승인 기준)을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행하세요. 동일한 합성 분포에서 N=100강 N=1000의 FID를 비교하세요. 편향의 크기를 보고하세요.
2. **중간.** 합성 CLIP 스타일 기능에서 CMMD를 구현하세요(공식 참고: Jayasumana et al., 2024). FID와 비교하여 품질 차이에 대한 민감도를 비교하세요.
3. **어려움.** HPSv2 설정을 복제하세요. Pick-a-Pic의 하위 집합에서 1000개의 이미지-프롬프트 쌍을 가져오고, 선호도에 대해 작은 CLIP 기반 스코어를 미세 조정하며, 보존된 세트와의 일치도를 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | 실제 이미지와 생성된 이미지의 Inception 기능에 대한 가우시안 적합의 Fréchet 거리. |
| CLIP 점수 | "텍스트-이미지 유사성" | CLIP 이미지 임베딩과 텍스트 임베딩 간의 코사인 유사도(Cosine Similarity). |
| CMMD | "FID의 대체제" | CLIP 기능의 MMD; 편향이 적고 가우시안 가정이 없음. |
| IS | "Inception score" | Exp KL(p(y|x) || p(y)); 최신 모델에서는 상관도가 낮아 폐기됨. |
| HPSv2 / ImageReward / PickScore | "학습된 선호도 대리 지표" | 인간 선호도로 학습된 작은 모델; 자동 심사자로 사용됨. |
| Elo | "체스 레이팅" | 쌍별 승리의 Bradley-Terry 집계. |
| PartiPrompts | "벤치마크 프롬프트 세트" | 12개 카테고리에 걸친 Google이 큐레이션한 1,600개 프롬프트. |
| FD-DINO | "자기 지도 대체제" | DINOv2 기능을 사용하는 FD; ImageNet 밖의 도메인에 더 적합. |

## 프로덕션 노트: 평가는 추론 워크로드이기도 합니다

10,000개 샘플에 대해 FID를 실행하려면 10,000개의 이미지를 생성해야 합니다. 단일 L4 GPU에서 50단계 SDXL base 모델로 1024² 해상도의 이미지를 생성하는 경우, 단일 요청 추론에 약 11시간이 소요됩니다. 평가 예산은 실재하는 제약이며, 이 상황은 오프라인 추론 시나리오(처리량 극대화, TTFT 무시)에 정확히 해당합니다:

- **배치를 크게 하고 지연은 무시하세요.** 오프라인 평가 = 메모리에 Fits하는 가장 큰 크기로 정적 배치를 수행합니다. 80GB H100에서 `pipe(...).images`와 `num_images_per_prompt=8`을 사용하면 단일 요청 대비 벽시계 시간이 4-6배 빨라집니다.
- **실제 특징을 캐싱하세요.** 실제 참조 세트에 대한 Inception (FID) 또는 CLIP (CLIP-score, CMMD) 특징 추출은 *한 번만* 수행하고 `.npz`으로 저장하세요. 매 평가마다 재계산하지 마세요.

CI / 회귀 게이트의 경우: PR마다 500개 샘플 하위 집합에 대해 FID + CLIP 점수를 실행하세요 (~30분); 전체 10,000개 샘플에 대해 FID + HPSv2 + Elo를 야간마다 실행하세요.

## 추가 읽기

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 논문.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 실패 모드 조사.
