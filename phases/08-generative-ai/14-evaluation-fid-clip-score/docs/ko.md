# 평가(Evaluation) — FID, CLIP Score, 인간 선호도(Human Preference)

> 모든 생성 모델 리더보드는 FID, CLIP score, 그리고 인간 선호도 아레나(human-preference arena)의 승률을 인용합니다. 각 지표에는 연구자가 악용(gaming)할 수 있는 실패 모드(failure mode)가 존재합니다. 이러한 실패 모드를 알지 못한다면, 실제적인 성능 개선과 지표 조작을 구분할 수 없습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time:** ~45 minutes

## 문제점 (The Problem)

생성 모델은 *샘플 품질(sample quality)*과 *조건 준수(conditioning adherence)*를 기준으로 평가됩니다. 이 두 가지 모두 닫힌 형태(closed-form)의 측정법이 존재하지 않습니다. 모델이 10,000장의 이미지를 생성한다면, 무언가는 이 이미지들에 숫자를 부여해야 합니다. 여러분은 모델 계열, 해상도, 아키텍처 전반에 걸쳐 그 숫자를 신뢰해야만 합니다. 2014년부터 현재까지 혹독한 검증 과정을 견뎌내고 살아남은 세 가지 핵심 지표는 다음과 같습니다:

- **FID (Fréchet Inception Distance).** Inception 네트워크의 특징 공간(feature space) 내에서 실제 분포와 생성된 분포 사이의 거리를 측정합니다. 값이 낮을수록 좋습니다.
- **CLIP score.** 생성된 이미지의 CLIP-image 임베딩과 프롬프트의 CLIP-text 임베딩 사이의 코사인 유사도(cosine similarity)를 측정합니다. 값이 높을수록 좋습니다. 프롬프트 준수 여부를 측정합니다.
- **인간 선호도 (Human preference).** 동일한 프롬프트에 대해 두 모델을 일대일로 맞붙인 뒤, 인간(또는 GPT-4급 모델)이 더 나은 모델을 선택하게 하여 이를 Elo 점수로 집계합니다.

또한 다음과 같은 지표들도 접하게 될 것입니다: IS (Inception Score, 현재는 거의 사용되지 않음), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. 각 지표는 이전 지표의 결함을 보완하기 위해 등장했습니다.

## 개념 (The Concept)

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — 샘플 품질 (sample quality)

Heusel et al. (2017). 단계:

1. $N$개의 실제 이미지와 $N$개의 생성된 이미지에 대해 Inception-v3 특징(2048-D)을 추출합니다.
2. 각 풀(pool)에 가우시안 분포를 맞춥니다: 평균 `μ_r, μ_g`와 공분산 `Σ_r, Σ_g`를 계산합니다.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`를 계산합니다.

해석: 특징 공간(feature space) 내 두 다변량 가우시안(multivariate Gaussians) 사이의 프레셰 거리(Fréchet distance)입니다. 값이 낮을수록 두 분포가 더 유사함을 의미합니다.

실패 사례(Failure modes):
- **작은 $N$에서의 편향(Biased on small N).** FID는 특징 분포에 대한 평균 제곱값입니다. $N$이 작으면 공분산을 과소평가하여 FID가 실제보다 낮게 측정될 수 있습니다. 항상 $N \ge 10,000$을 사용하세요.
- **Inception 의존성(Inception-dependent).** Inception-v3는 ImageNet으로 학습되었습니다. ImageNet과 거리가 먼 도메인(얼굴, 예술, 텍스트 이미지)은 의미 없는 FID를 생성합니다. 도메인 특화 특징 추출기(domain-specific feature extractor)를 사용하세요.
- **게이밍(Gaming).** Inception 사전 분포(prior)에 과적합(overfitting)하면 시각적 품질 개선 없이 FID만 낮출 수 있습니다. 아래의 CMMD를 사용하여 이를 극복해 보세요.

### CLIP 점수(CLIP score) — 프롬프트 준수(prompt adherence)

Radford et al. (2021). 생성된 이미지 + 프롬프트에 대하여:

```python
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

3만 개의 생성된 이미지에 대해 평균을 내면 → 모델 간 비교 가능한 스칼라(scalar) 값이 됩니다.

실패 사례(Failure modes):
- **CLIP 자체의 사각지대.** CLIP은 결합적 추론(compositional reasoning) 능력이 약합니다(예: "파란 구체 위의 빨간 큐브"와 같은 경우 자주 실패함). 모델이 복잡한 프롬프트를 실제로 따르지 않고도 CLIP 점수에서 높은 순위를 기록할 수 있습니다.
- **짧은 프롬프트 편향(Short prompt bias).** 짧은 프롬프트는 실제 데이터에서 CLIP-이미지 매칭이 더 많이 발생합니다. 긴 프롬프트는 기계적으로 CLIP 점수가 낮게 측정됩니다.
- **프롬프트 게임(Prompt gaming).** 프롬프트에 "high quality, 4k, masterpiece"를 포함하면 이미지-텍스트 결합(image-text binding)을 개선하지 않고도 CLIP 점수를 부풀릴 수 있습니다.

CMMD (Jayasumana et al., 2024)는 이러한 문제 중 일부를 해결합니다: Inception 대신 CLIP 특징(features)을 사용하며, Fréchet 대신 최대 평균 불일치(maximum-mean discrepancy)를 사용합니다. 미세한 품질 차이를 감지하는 데 더 효과적입니다.

### 인간 선호도(Human preference) — 정답(Ground truth)

프롬프트 풀을 선정합니다. 모델 A와 모델 B로 각각 생성합니다. 생성된 쌍(pairs)을 인간(또는 강력한 LLM 판사)에게 보여줍니다. 승리 횟수를 집계하여 Elo 점수 또는 Bradley-Terry 점수를 산출합니다. 주요 벤치마크는 다음과 같습니다:

- **PartiPrompts (Google)**: 12개 카테고리의 1,600개 다양한 프롬프트.
- **HPSv2**: 107k개의 인간 주석(human annotations)을 포함하며, 자동화된 대리 지표(automated proxy)로 널리 사용됩니다.
- **ImageReward**: 137k개의 프롬프트-이미지 선호도 쌍을 포함하며, MIT 라이선스입니다.
- **PickScore**: Pick-a-Pic의 2.6M 선호도 데이터를 기반으로 학습되었습니다.
- **Chatbot-Arena 스타일의 이미지 아레나(image arenas)**: https://imagearena.ai/ 및 기타 서비스.

실패 모드(Failure modes):
- **판사 변동성(Judge variance).** 비전문가는 전문가와 다른 선호도를 가질 수 있습니다. 두 집단 모두를 활용하세요.
- **프롬프트 분포(Prompt distribution).** 체리피킹(Cherry-picked)된 프롬프트는 특정 모델 계열에 유리할 수 있습니다. 항상 이를 문서화하세요.
- **LLM 판사의 보상 해킹(LLM-judge reward hacking).** GPT-4-judge는 시각적으로 예쁘지만 내용이 틀린 출력물에 속을 수 있습니다. 인간의 평가와 함께 삼각 측량(Triangulate)하여 검증하세요.

## 함께 사용하기 (Use together)

프로덕션 평가 보고서(production eval report)에는 다음 내용이 포함되어야 합니다:

1. 별도로 분리된 실제 데이터 분포(held-out real distribution)와 비교한 10k-30k 샘플에 대한 FID (샘플 품질).
2. 동일한 샘플과 해당 프롬프트 간의 CLIP score / CMMD (지시 이행도).
3. 이전 모델과 비교한 블라인드 아레나(blinded arena)에서의 승률 (전반적인 선호도).
4. 실패 모드 분석(Failure mode analysis): 무작위로 추출된 50개의 출력물에 대해 알려진 문제점(손의 해부학적 구조, 텍스트 렌더링, 일관된 객체 수 등)을 표시.

단일 지표는 거짓일 수 있습니다. 세 가지의 상호 보완적인 지표와 정성적 검토(qualitative review)가 결합되어야 비로소 하나의 주장(claim)이 됩니다.

```figure
gx-fid-distributions
```

## 구현하기 (Build It)

`code/main.py`는 합성 "특징 벡터(feature vectors)"(Inception 특징을 대신하기 위해 4차원 벡터를 사용함)에 대해 FID, CLIP-score 방식, 그리고 Elo 집계(aggregation)를 구현합니다. 다음 내용을 확인할 수 있습니다:

- 작은 $N$과 큰 $N$에서의 FID 계산 — 편향(bias) 확인.
- 특징 풀(feature pools) 간의 코사인 유사도로 계산되는 "CLIP 점수(CLIP score)".
- 합성 선호도 스트림(synthetic preference stream)을 이용한 Elo 업데이트 규칙.

### 1단계: 네 줄로 구현하는 FID (FID in four lines)

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 2단계: CLIP 스타일 코사인 유사도 (CLIP-style cosine-similarity)

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 3단계: Elo 집계 (Elo aggregation)

```python
def elo_update(r_a, r_b, winner, k=32):
    # 승리 확률(expected score) 계산
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    # 새로운 레이팅 계산
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 주의 사항 (Pitfalls)

- **N=1000에서의 FID.** $N=10k$ 미만에서는 휴리스틱(Heuristic)을 신뢰할 수 없습니다. 낮은 $N$에서의 FID를 보고하는 논문들은 수치를 조작(gaming)하고 있을 가능성이 높습니다.
- **해상도 간의 FID 비교.** Inception의 299×299 리사이징은 특징 분포(feature distribution)를 변화시킵니다. 반드시 동일한 해상도에서만 비교하세요.
- **단일 시드(seed) 보고.** 최소 3개의 시드를 실행하세요. 표준 편차(std)를 함께 보고해야 합니다.
- **부정 프롬프트(negative prompts)를 통한 CLIP 점수 부풀리기.** 일부 파이프라인은 프롬프트에 과적합(over-fitting)되어 CLIP 점수를 높이기도 합니다. 시각적 포화(visual saturation) 현상이 있는지 확인하세요.
- **프롬프트 중복으로 인한 Elo 편향.** 만약 두 모델 모두 학습 과정에서 벤치마크 프롬프트를 보았다면, Elo 점수는 의미가 없습니다. 학습에 사용되지 않은(held-out) 프롬프트 세트를 사용하세요.
- **유료 크라우드소싱을 통한 인간 평가(Human eval)의 왜곡.** Prolific이나 MTurk 작업자들은 연령대가 낮거나 기술 친화적인 경향이 있습니다. 모집된 예술/디자인 전문가들과 혼합하여 평가하세요.

## 활용하기 (Use It)

2026년의 프로덕션 평가 프로토콜(Production eval protocol):

| 기둥 (Pillar) | 최소 기준 (Minimum) | 권장 기준 (Recommended) |
|--------|---------|-------------|
| 샘플 품질 (Sample quality) | 홀드아웃(held-out) 실제 데이터 10k 대비 FID | + 5k 데이터에 대한 CMMD + 카테고리별 서브셋 FID |
| 프롬프트 준수 (Prompt adherence) | 30k 데이터에 대한 CLIP score | + HPSv2 + ImageReward + VQA 방식의 질의응답 |
| 선호도 (Preference) | 베이스라인 대비 200개의 블라인드 쌍(blinded pairs) | + 2,000개의 인간/LLM-judge 쌍 + Chatbot Arena |
| 실패 분석 (Failure analysis) | 50개의 수동 플래그(hand-flagged) | 500개의 수동 플래그 + 자동화된 안전 분류기(safety classifier) |

네 가지 기둥이 모두 포함된 하나의 보고서가 기술적 주장(claim)이 됩니다. 이 중 하나만 있는 경우라면 마케팅(marketing)에 불과합니다.

## Ship It (실행하기)

`outputs/skill-eval-report.md`를 저장하세요. `skill`은 새로운 모델 체크포인트와 베이스라인(baseline)을 입력받아 샘플 크기, 메트릭, 실패 모드 조사(failure-mode probes), 승인 기준(sign-off criteria)을 포함한 전체 평가 계획을 출력합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 동일한 합성 분포(synthetic distributions)에서 $N=100$일 때와 $N=1000$일 때의 FID를 비교해 보세요. 편향 크기(bias magnitude)를 보고하세요.
2. **중간 (Medium).** 합성 CLIP 스타일 특징(synthetic CLIP-style features)으로부터 CMMD를 구현해 보세요 (공식은 Jayasumana et al., 2024를 참조하세요). 품질 차이에 대한 민감도를 FID와 비교해 보세요.
3. **어려움 (Hard).** HPSv2 설정을 재현해 보세요: Pick-a-Pic의 하위 집합에서 1000개의 이미지-프롬프트 쌍을 가져와서, 선호도(preferences)를 바탕으로 작은 CLIP 기반 스코어러(scorer)를 미세 조정(fine-tune)하고, 이를 홀드아웃 세트(held-out set)와 비교하여 일치도를 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | 실제 데이터와 생성 데이터의 Inception 특징값에 대한 가우시안 적합(Gaussian fits)의 Fréchet 거리. |
| CLIP score | "텍스트-이미지 유사도" | CLIP 이미지 임베딩과 텍스트 임베딩 사이의 코사인 유사도. |
| CMMD | "FID의 대체제" | CLIP 특징 기반 MMD; 편향이 적고 가우시안 가정을 하지 않음. |
| IS | "Inception score" | $\text{Exp KL}(p(y|x) \parallel p(y))$; 최신 모델에서는 상관관계가 낮아 더 이상 사용되지 않음. |
| HPSv2 / ImageReward / PickScore | "학습된 선호도 프록시" | 인간의 선호도를 학습한 소형 모델; 자동 평가자(automatic judges)로 사용됨. |
| Elo | "체스 레이팅" | 쌍체 비교 승률을 Bradley-Terry 모델로 집계한 점수. |
| PartiPrompts | "벤치마크 프롬프트 세트" | 12개 카테고리에 걸쳐 Google이 큐레이션한 1,600개의 프롬프트. |
| FD-DINO | "자기지도 학습 대체제" | DINOv2 특징을 사용하는 FD; ImageNet 범위를 벗어난 도메인에서 더 효과적임. |

## 프로덕션 노트: 평가는 추론 워크로드이기도 합니다 (Production note: evaluation is an inference workload too)

10k개의 샘플에 대해 FID를 실행한다는 것은 10k개의 이미지를 생성한다는 것을 의미합니다. 단일 L4 GPU에서 1024² 해상도의 50-step SDXL 베이스 모델을 사용할 경우, 이는 약 11시간의 단일 요청(single-request) 추론이 소요됩니다. 평가 예산은 실제적인 제약 사항이며, 그 프레임워크는 정확히 오프라인 추론(offline-inference) 시나리오(처리량 극대화, TTFT 무시)와 동일합니다:

- **지연 시간(latency)은 잊고 배치(Batch)를 최적화하세요.** 오프라인 평가는 메모리에 들어갈 수 있는 가장 큰 크기로 정적 배칭(static batching)을 수행하는 것과 같습니다. 80GB H100에서 `num_images_per_prompt=8`로 `pipe(...).images`를 실행하면 단일 요청 방식보다 실제 소요 시간(wall-clock time) 기준으로 4~6배 더 빠릅니다.
- **실제 특징(real features)을 캐싱하세요.** 실제 참조 세트에 대한 Inception (FID) 또는 CLIP (CLIP-score, CMMD) 특징 추출은 *단 한 번*만 실행하여 `.npz` 파일로 저장합니다. 평가할 때마다 다시 계산하지 마세요.

CI / 회귀 테스트 게이트(regression gates)의 경우: PR당 500개 샘플의 서브셋에 대해 FID + CLIP score를 실행하세요(약 30분 소요). 전체 10k FID + HPSv2 + Elo는 매일 밤(nightly) 실행하세요.

## 추가 읽을거리 (Further Reading)

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 논문.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 실패 모드(failure-mode) 조사 논문.
