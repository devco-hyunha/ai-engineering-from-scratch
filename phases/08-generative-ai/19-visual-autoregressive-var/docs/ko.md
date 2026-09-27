# 시각적 자기회귀 모델링 (Visual Autoregressive Modeling, VAR): 차기 스케일 예측 (Next-Scale Prediction)

> 확산 모델(Diffusion models)은 시간 축(denoising steps)에 따라 반복적으로 샘플링합니다. 반면 VAR은 스케일(scale)에 따라 반복적으로 샘플링합니다. 즉, 1x1 토큰을 예측한 다음 2x2, 4x4 순으로 최종 해상도까지 예측하며, 각 스케일은 이전 스케일을 조건(conditioning)으로 삼습니다. 2024년 발표된 논문에 따르면, VAR은 이미지 생성에 있어 GPT 스타일의 스케일링 법칙(scaling laws)을 따르며, 동일한 연산 예산(compute budget) 내에서 DiT를 능가하는 성능을 보여주었습니다. 이 레슨에서는 그 핵심 메커니즘을 구축해 봅니다.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## 문제점 (The Problem)

자기회귀(Autoregressive) 생성은 예측 가능한 확장성(Scalability) 덕분에 언어 모델링을 지배해 왔습니다. 즉, 더 많은 연산량과 더 많은 파라미터를 투입할수록 당혹도(Perplexity)는 낮아지고 출력 품질은 향상됩니다. 이미지 생성 분야에서는 2024년 이전까지 두 가지 주요 AR 시도가 있었습니다: PixelRNN/PixelCNN(픽셀 단위)과 DALL-E 1 / Parti / MuseGAN(VQ-VAE 코드를 이용한 토큰 단위)입니다.

두 방식 모두 생성 순서 문제(generation-order problem)를 겪었습니다. 픽셀과 토큰은 2D 그리드에 배치되어 있지만, AR 모델은 이를 1D 래스터 순서(raster order)로 방문해야 합니다. 초기의 모서리 픽셀은 이미지가 최종적으로 어떤 모습이 될지 전혀 알 수 없습니다. 이로 인해 생성 품질의 확장성은 텍스트 기반의 GPT보다 떨어졌으며, 동일한 연산량 대비 확산 모델(diffusion model)의 품질에 도달하지 못했습니다.

VAR은 생성되는 대상 자체를 변경함으로써 생성 순서 문제를 해결합니다. 공간상에서 이미지 토큰을 하나씩 예측하는 대신, VAR은 점진적으로 해상도가 높아지는 전체 이미지를 예측합니다. 1단계: 1x1 토큰(전체 이미지의 "요약")을 예측합니다. 2단계: 2x2 토큰 그리드(더 거친 특징들)를 예측합니다. 3단계: 4x4 그리드를 예측합니다. K단계: 최종 (H/8)x(W/8) 그리드를 예측합니다.

각 스케일(scale)은 이전의 모든 스케일을 참조하며("스케일 순서"에 따른 인과적 관계), 해당 스케일 내에서는 병렬로 처리됩니다. 이로 인해 순서 문제가 사라집니다. 스케일 $k$에서의 전체 이미지는 단 한 번의 트랜스포머 패스(transformer pass)로 생성됩니다.

## 개념 (The Concept)

### VQ-VAE 멀티스케일 토크나이저 (Multi-Scale Tokenizer)

VAR에는 **멀티스케일 이산 토크나이저(multi-scale discrete tokenizer)**가 필요합니다. 이미지 `x`에 대해, 이 토크나이저는 점진적으로 해상도가 높아지는 토큰 그리드 시퀀스를 생성합니다:

```
x -> encoder -> latent f
f -> 1x1에서 토큰화: shape (1, 1)인 토큰 그리드 z_1
f -> 2x2에서 토큰화: shape (2, 2)인 토큰 그리드 z_2
...
f -> (H/p)x(W/p)에서 토큰화: shape (H/p, W/p)인 토큰 그리드 z_K
```

각 `z_k`는 동일한 코드북(typical size 4096-16384)을 사용합니다. 각 스케일에서의 토큰화는 독립적이지 않습니다. 각 스케일의 잔차(residuals)를 합산하여 `f`를 재구성하도록 학습됩니다:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

이는 **잔차 VQ(residual VQ)**의 변형입니다. 스케일 `k`는 스케일 `1..k-1`이 놓친 부분을 포착합니다. 디코더는 모든 스케일 임베딩의 합을 받아 이미지를 생성합니다.

멀티스케일 VQ 토크나이저는 (VQGAN과 같이) 한 번 학습된 후 동결(frozen)됩니다. 모든 생성 작업은 그 위에 구축된 자기회귀(autoregressive) 모델이 수행합니다.

### 차기 스케일 예측 (Next-Scale Prediction)

생성 모델은 이전의 모든 스케일로부터 토큰을 보고 다음 스케일의 토큰을 예측하는 트랜스포머(Transformer)입니다.

입력 시퀀스 구조:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

위치 임베딩(Position embeddings)은 스케일 인덱스와 해당 스케일 내의 공간적 위치를 모두 인코딩합니다. 어텐션(Attention)은 스케일 순서에 따라 인과적(Causal)으로 작동합니다. 즉, 스케일 `k`의 위치 `(i, j)`에 있는 토큰은 스케일 `1..k`에 있는 모든 토큰과, 스케일 `k` 내에서 사용된 내부 순서상 이전에 위치한 스케일 `k`의 토큰들을 참조할 수 있습니다 (VAR는 스케일 내 인과성이 없는 고정된 위치 어텐션을 사용하며, 한 스케일 내의 모든 위치는 병렬로 예측됩니다).

학습 손실(Training loss): 각 스케일 `k`에서, 이전 스케일의 모든 토큰이 주어졌을 때 토큰 `z_k`를 예측합니다. 이산적 VQ 코드(discrete VQ codes)에 대한 교차 엔트로피 손실(Cross-entropy loss)을 사용합니다. "시퀀스"가 이제 스케일 구조를 갖는다는 점을 제외하면 GPT와 동일한 구조입니다.

### 생성 (Generation)

추론 시:
```
generate z_1 = sample from p(z_1)                    # 1개 토큰
generate z_2 = sample from p(z_2 | z_1)              # 4개 토큰 병렬 생성
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16개 토큰 병렬 생성
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

K = 10개의 스케일(scales)인 경우, 생성에는 10번의 트랜스포머 순전파(forward pass)가 필요합니다. 각 패스는 해당 스케일 전체를 병렬로 생성하며, 스케일 내부에서 토큰별 자기회귀(autoregression)를 수행하지 않습니다. 256x256 이미지의 경우, 이는 DiT의 28~50회 패스에 비해 대략 10회의 패스만 필요함을 의미합니다.

### 왜 Next-Scale이 Next-Token보다 우수한가 (Why Next-Scale Wins Over Next-Token)

세 가지 구조적 이점:
1. **거친 단계에서 세밀한 단계로의 정렬 (Coarse-to-fine aligns with natural image statistics):** 인간의 시각적 인지와 이미지 데이터셋 모두 스케일에 따른 규칙성을 보입니다. 저주파(low-frequency) 구조는 안정적이고 예측 가능하며, 고주파(high-frequency) 디테일은 저주파 콘텐츠에 종속적입니다. Next-scale 예측은 이 점을 활용합니다.
2. **스케일 내 병렬 생성 (Parallel generation within scale):** GPT 방식의 토큰 자기회귀(AR)와 달리, VAR는 하나의 스케일에 있는 모든 토큰을 한 번에 생성합니다. 유효 생성 길이는 선형(linear)이 아닌 로그 스케일(log-scale)로 작동합니다.
3. **생성 순서 편향 없음 (No generation order bias):** 스케일 `k`의 토큰들은 스케일 `k-1`의 모든 정보를 참조합니다. 따라서 후속 컨텍스트가 제공되기 전에 초기 토큰이 먼저 결정되어야 하는 "왼쪽" 또는 "위쪽" 편향(bias)이 존재하지 않습니다.

### 스케일링 법칙 (Scaling Law)

Tian et al.은 VAR가 ImageNet에서의 FID에 대해 GPT가 perplexity에서 보여주는 것과 마찬가지로 거듭제곱 법칙(power-law) 스케일링 곡선을 따른다는 것을 입증했습니다. 파라미터 수나 연산량을 두 배로 늘리면 오차는 신뢰할 수 있는 수준으로 절반이 됩니다. 이는 언어 모델만큼이나 깔끔하게 이러한 스케일링 동작을 보여준 최초의 이미지 생성 모델이었습니다. 그 결과, VAR 규모의 예측은 아키텍처별 경험적 추측이 아닌, 연산량으로부터 예측 가능한 것이 되었습니다.

### 확산 모델과의 관계 (Relationship to Diffusion)

VAR와 확산 모델(Diffusion)은 동일한 데이터 압축(data-compression) 서사를 공유합니다. 즉, 두 방식 모두 생성 문제를 일련의 더 쉬운 하위 문제들로 분해합니다.

- 확산 모델(Diffusion): 점진적으로 노이즈를 추가하며, 그 한 단계를 되돌리는 법을 학습합니다.
- VAR: 점진적으로 해상도를 추가하며, 다음 스케일(scale)을 예측하는 법을 학습합니다.

두 모델은 문제를 바라보는 서로 다른 축을 가집니다. 하지만 둘 다 다루기 쉬운 조건부 분포(conditional distributions)를 생성해 냅니다. 경험적으로 VAR는 추론(inference) 속도가 더 빠르며(더 적은 패스, 스케일 내 모든 과정의 병렬 처리), 클래스 조건부 ImageNet 작업에서 DiT와 대등하거나 더 나은 성능을 보여줍니다. 텍스트 조건부 VAR(VARclip, HART)는 현재 활발히 연구되고 있는 방향입니다.

```figure
gx-var-next-scale
```

## 구현하기 (Build It)

`code/main.py`에서 다음 과정을 수행합니다:
1. 합성 "이미지" 데이터(2D 가우시안 링)를 사용하여 아주 작은 **멀티 스케일 VQ 토크나이저(multi-scale VQ tokenizer)**를 구축합니다.
2. 토큰을 다음 스케일로 예측(next-scale-predict)하도록 **VAR 스타일 트랜스포머(VAR-style transformer)**를 학습시킵니다.
3. 트랜스포머를 4번 호출(4개 스케일)하여 샘플링하고 디코딩합니다.
4. 스케일 순서로 정렬된 학습이 스케일 내에서 병렬 생성을 가능하게 하는지 확인합니다.

이것은 토이 구현(toy implementation)입니다. 핵심은 스케일 구조화된 어텐션 마스크(scale-structured attention mask)와 스케일 내 병렬 생성(parallel-within-scale generation)이 실제로 작동하는 것을 확인하는 것입니다.

## Ship It (실행하기)

이 레슨은 `outputs/skill-var-tokenizer-designer.md`를 생성합니다 — 이는 멀티 스케일 토크나이저(multi-scale tokenizer)를 설계하기 위한 스킬입니다: 스케일 수, 스케일 비율, 코드북 크기, 잔차 공유(residual sharing), 디코더 아키텍처 등을 설계할 수 있습니다.

## 연습 문제 (Exercises)

1. **스케일 수 절제 실험 (Scale count ablation).** 4, 6, 8, 10개의 스케일로 VAR를 학습시켜 보세요. 자기회귀 패스(autoregressive passes) 횟수에 따른 재구성 품질(reconstruction quality)을 측정합니다. 스케일이 많을수록 잔차(residuals)가 더 정교해져 품질은 좋아지지만, 패스 횟수는 늘어납니다.

2. **코드북 크기 (Codebook size).** 코드북 크기를 512, 4096, 16384로 설정하여 토크나이저를 학습시켜 보세요. 코드북이 클수록 재구성은 잘 되지만 예측은 더 어려워집니다. 성능의 변곡점(knee)을 찾아보세요.

3. **스케일 내 병렬성 확인 (Parallel-within-scale check).** 학습된 VAR를 대상으로 어텐션 패턴(attention pattern)을 명시적으로 측정해 보세요. 스케일 `k` 내에서 모델이 스케일 간(cross-scale) 위치에는 어텐션을 주지만, 스케일 내(intra-scale) 위치에는 주지 않는지 확인합니다. 마스크(mask) 구현이 올바른지 검증해 보세요.

4. **VAR vs DiT 스케일링 (VAR vs DiT scaling).** 동일한 ImageNet 클래스 조건부(class-conditional) 태스크에 대해, VAR와 DiT를 동일한 파라미터 예산(예: 33M, 130M, 458M)으로 학습시켜 보세요. 연산량(compute) 대비 FID를 그래프로 그려보세요. VAR는 각 크기에서 DiT보다 앞서 나가야 합니다. 작은 규모에서도 논문의 결과를 재현해 보세요.

5. **텍스트 조건화 (Text conditioning).** adaLN을 통해 CLIP 풀링된 텍스트 임베딩을 추가적인 조건부 입력으로 받도록 VAR를 확장해 보세요. 이것이 HART 레시피입니다. 텍스트 정렬 샘플링(text-aligned sampling) 시 FID가 얼마나 개선되는지 확인해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 사람들이 말하는 방식 | 실제 의미 |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | VQ 토큰 그리드 피라미드 상에서 다음 스케일(next-scale)을 예측하여 이미지를 생성하는 방식 |
| Next-scale prediction | "거친 것부터, 그다음 정교한 것 순으로 예측" | 모델이 이전의 모든 스케일을 조건(conditioning)으로 하여, 점진적으로 증가하는 해상도 스케일의 토큰을 예측함 |
| Multi-scale VQ tokenizer | "Residual VQ" | 해상도가 증가하는 K개의 토큰 그리드를 생성하며, 디코더가 모든 스케일을 합산하는 VQ-VAE |
| Scale k | "피라미드 레벨 k" | K개의 해상도 레벨 중 하나로, k=1일 때 1x1부터 k=K일 때 (H/p)x(W/p)까지 구성됨 |
| Parallel-within-scale | "스케일당 한 번의 순전파" | 스케일 k의 모든 토큰은 자기회귀(autoregressively) 방식이 아닌, 단 한 번의 트랜스포머 패스(pass)로 예측됨 |
| Causal-across-scales | "스케일 순서 기반 어텐션" | 스케일 k의 토큰은 1..k 스케일의 모든 토큰을 참조(attend)할 수 있지만, k+1..K 스케일은 참조할 수 없음 |
| Residual VQ | "가산적 토큰화(Additive tokenization)" | 각 스케일의 토큰은 하위 스케일이 남긴 잔차(residual)를 인코딩하며, 디코더는 모든 스케일의 임베딩을 합산함 |
| VAR scaling law | "Image GPT 스케일링" | FID가 언어 모델의 퍼플렉서티(perplexity)처럼 연산량에 따라 예측 가능한 멱법칙(power law)을 따름 |
| HART | "Hybrid VAR + text" | MaskGIT 방식의 반복적 디코딩과 VAR의 스케일 구조를 결합한 텍스트 조건부 VAR 변형 모델 |
| Scale position embedding | "(scale, row, col) 트리플" | 위치 인코딩(Positional encoding)에 스케일 인덱스와 해당 스케일 내의 공간 좌표를 모두 포함함 |

## 추가 읽을거리 (Further Reading)

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 논문, 표준 참조 문헌
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, 확산 모델(diffusion) 비교 베이스라인
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, VAR의 멀티 스케일 토크나이저가 확장한 토크나이저 계열
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, 이산 이미지 토큰화(discrete image tokenization)의 기초
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 텍스트 조건부(text-conditional) VAR
