# ControlNet, LoRA 및 컨디셔닝 (Conditioning)

> 텍스트만으로는 제어 신호가 투박할 수밖에 없습니다. ControlNet을 사용하면 사전 학습된 확산 모델(diffusion model)을 복제하여 깊이 지도(depth map), 포즈 스켈레톤(pose skeleton), 낙서(scribble) 또는 에지 이미지(edge image)로 모델을 조종할 수 있습니다. LoRA를 사용하면 1,000만 개의 파라미터만 학습하여 20억(2B) 개의 파라미터를 가진 모델을 미세 조정할 수 있습니다. 이 두 기술이 결합되어 Stable Diffusion은 단순한 장난감 수준을 넘어, 2026년 모든 에이전시에서 사용하는 이미지 파이프라인으로 진화했습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)
**Time:** ~75 minutes

## 문제점 (The Problem)

"번화한 거리에서 빨간 드레스를 입은 여성이 개를 산책시키고 있다"와 같은 프롬프트는 모델에게 개의 *위치*, 여성의 *자세*, 또는 거리의 *원근감*에 대한 정보를 전혀 제공하지 않습니다. 텍스트는 이미지를 지정하는 데 필요한 요소 중 약 10%만을 확정할 수 있습니다. 나머지는 시각적인 요소이며 말로 효율적으로 설명하기 어렵습니다.

모든 신호(자세, 깊이, Canny, 세그멘테이션)에 대해 매번 처음부터 새로운 조건부 모델(conditional model)을 학습시키는 것은 비용이 너무 많이 듭니다. 여러분은 2.6B 파라미터 규모의 SDXL 백본(backbone)을 동결(frozen) 상태로 유지하면서, 조건을 읽어들이는 작은 사이드 네트워크(side-network)를 부착하여 백본의 중간 특징(intermediate features)을 미세하게 조정(nudge)하기를 원합니다. 이것이 바로 ControlNet입니다.

또한 전체 모델을 재학습하지 않고도 모델에게 새로운 개념(여러분의 얼굴, 제품, 스타일)을 가르치고 싶을 것입니다. 전체 모델보다 100배 더 작은 변화량(delta)을 원할 것입니다. 이것이 바로 LoRA(Low-Rank Adapters)입니다. 기존의 어텐션 가중치(attention weights)에 끼워 넣는 저차원 어댑터 방식입니다.

ControlNet + LoRA + 텍스트 = 2026년 실무자의 도구 모음(toolkit)입니다. 대부분의 상용 이미지 파이프라인은 SDXL / SD3 / Flux 베이스 모델 위에 2~5개의 LoRA, 1~3개의 ControlNet, 그리고 IP-Adapter를 계층적으로 쌓아서 사용합니다.

## 개념 (The Concept)

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

사전 학습된 SD(Stable Diffusion)를 준비합니다. U-Net의 인코더(encoder) 절반을 *복제(Clone)*하세요. 원본은 동결(freeze)합니다. 복제된 모델이 추가적인 조건부 입력(edges, depth, pose 등)을 받아들일 수 있도록 학습시킵니다. 이 복제본을 *제로 컨볼루션(zero-convolution)* 스킵 연결(1×1 conv이며 0으로 초기화됨 — 처음에는 아무 동작도 하지 않는 no-op 상태로 시작하여, 차이값인 delta를 학습함)을 통해 원본의 디코더(decoder) 절반에 다시 연결합니다.

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

제로 컨볼루션 초기화는 ControlNet이 항등 함수(identity) 상태로 시작함을 의미합니다. 즉, 학습 전에는 원본 모델에 아무런 영향을 주지 않습니다. 표준 확산 손실(diffusion loss)을 사용하여 100만 개의 (프롬프트, 조건, 이미지) 트리플 세트로 학습합니다.

모달리티별 ControlNet은 작은 사이드 모델 형태로 제공됩니다 (SDXL의 경우 약 360M, SD 1.5의 경우 약 70M). 추론 시 다음과 같이 모델들을 조합할 수 있습니다:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

모델 내의 모든 선형 레이어(linear layer) `W ∈ R^{d×d}`에 대해, `W`를 동결(freeze)하고 다음과 같은 저차원 델타(low-rank delta)를 추가합니다:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

이때 `r << d`를 만족해야 합니다. 어텐션(attention)의 경우 랭크(rank) 4-16이 표준이며, 집중적인 미세 조정(heavy fine-tunes)에는 64-128 랭크를 사용합니다. 새로운 파라미터의 수는 `d²` 대신 `2 · d · r`이 됩니다. `d=640`, `r=16`인 SDXL 어텐션의 경우, 어댑터당 410k개의 파라미터 대신 20k개의 파라미터만 필요하며, 이는 20배의 감소를 의미합니다. 모델 전체를 기준으로 볼 때, 기본 모델이 5GB라면 LoRA는 보통 20-200MB 수준입니다.

추론(inference) 시에는 LoRA의 스케일을 조정할 수 있습니다: `W' = W + α · B @ A`. 보통 `α = 0.5-1.5` 범위를 사용합니다. 여러 개의 LoRA는 가산적으로(additively) 쌓을 수 있습니다 (단, 이들이 비선형적인 방식으로 상호작용할 수 있다는 점에 주의해야 합니다).

### IP-Adapter (Ye et al., 2023)

텍스트와 함께 *이미지*를 조건(conditioning)으로 받아들이는 아주 작은 어댑터(adapter)입니다. CLIP 이미지 인코더를 사용하여 이미지 토큰을 생성하고, 이를 텍스트 토큰과 함께 크로스 어텐션(cross-attention)에 주입합니다. 베이스 모델당 크기는 약 20MB입니다. LoRA 없이도 "이 참조 이미지의 스타일로 이미지를 생성해 줘"와 같은 작업이 가능합니다.

### 구성 가능성 매트릭스 (Composability matrix)

| 도구 (Tool) | 제어 대상 (What it controls) | 크기 (Size) | 사용 시점 (When to use) |
|------|------------------|------|-------------|
| ControlNet | 공간 구조 (포즈, 깊이, 에지) | 70-360MB | 정확한 레이아웃 및 구도 제어 시 |
| LoRA | 스타일, 피사체, 개념 | 20-200MB | 개인화 또는 특정 스타일 적용 시 |
| IP-Adapter | 참조 이미지로부터의 스타일 또는 피사체 | 20MB | 텍스트로 설명할 수 없는 외형을 구현할 때 |
| Textual Inversion | 새로운 토큰으로서의 단일 개념 | 10KB | 레거시 방식, 주로 LoRA로 대체됨 |
| DreamBooth | 피사체에 대한 전체 미세 조정 (Full fine-tune) | 2-5GB | 강력한 정체성 구현 필요 시, 높은 연산 자원 필요 |
| T2I-Adapter | 가벼운 ControlNet 대안 | 70MB | 에지 디바이스 또는 추론 예산이 제한적일 때 |

ControlNet ≈ 공간적(spatial). LoRA ≈ 의미적(semantic). 두 가지를 모두 사용해 보세요.

```figure
v4-controlnet-zero
```

## 직접 구현해 보기 (Build It)

`code/main.py`는 다음 두 가지 메커니즘을 1차원(1-D) 상에서 시뮬레이션합니다:

1. **LoRA.** 사전 학습된 선형 레이어 `W`를 준비합니다. 이를 동결(Freeze)합니다. `W + BA`가 목표 선형 레이어와 일치하도록 저차원 행렬 `B @ A`를 학습시킵니다. `r = 1`만으로도 랭크-1 보정(rank-1 correction)을 완벽하게 학습할 수 있음을 보여줍니다.

2. **ControlNet-lite.** "동결된 베이스(frozen base)" 예측기와 추가 신호를 읽어들이는 "사이드 네트워크(side network)"로 구성됩니다. 사이드 네트워크의 출력은 0으로 초기화된 학습 가능한 스칼라 값에 의해 게이팅(gated)됩니다(우리가 구현한 zero-conv 방식). 학습을 진행하며 게이트 값이 점차 증가하는 것을 관찰해 보세요.

### 1단계: LoRA 수학 (LoRA math)

```python
def lora(W, A, B, x, alpha=1.0):
    # W는 동결(frozen) 상태이며, A와 B는 학습 가능한 저차원 행렬(low-rank factors)입니다.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 2단계: 사이드 네트워크의 제로 초기화 (zero-init side network)

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate는 0으로 초기화됨
h = base(x) + gated
```

0단계에서는 출력이 `base`와 동일합니다. 초기 학습 단계에서 `gate`는 천천히 업데이트되므로, 파괴적인 드리프트(catastrophic drift)가 발생하지 않습니다.

## 주의 사항 (Pitfalls)

- **LoRA 과도한 스케일링 (Over-scaling LoRAs).** `α = 2` 또는 `α = 3`은 "더 강하게 만들기" 위한 흔한 편법이지만, 결과물이 과도하게 스타일화되거나 깨질 수 있습니다. `α ≤ 1.5`를 유지하세요.
- **ControlNet 가중치 충돌 (ControlNet weight conflict).** Pose ControlNet 가중치를 1.0으로, Depth ControlNet 가중치를 1.0으로 설정하면 대개 결과물이 과도하게 나옵니다. 가중치의 합이 ≈ 1.0이 되도록 하는 것이 안전한 기본값입니다.
- **잘못된 베이스 모델에 LoRA 적용 (LoRA on the wrong base).** SDXL LoRA는 어텐션 차원(attention dimensions)이 일치하지 않기 때문에 SD 1.5에서 아무런 동작을 하지 않습니다(no-op). Diffusers 0.30+ 버전부터는 경고가 표시됩니다.
- **Textual Inversion 드리프트 (Textual Inversion drift).** 특정 체크포인트에서 학습된 토큰은 다른 체크포인트에서 심하게 드리프트(drift)됩니다. LoRA가 더 높은 이식성을 가집니다.
- **LoRA 가중치 병합 및 저장 (LoRA weight-merging and storage).** 더 빠른 추론(런타임 추가 연산 없음)을 위해 LoRA를 베이스 모델 가중치에 구워 넣을(bake) 수 있지만, 이 경우 런타임에 `α`를 조절하는 기능을 잃게 됩니다. 두 버전 모두 보관하세요.

## 활용하기 (Use It)

| 목표 (Goal) | 2026년 파이프라인 (2026 pipeline) |
|------|---------------|
| 브랜드 아트 스타일 재현 | rank 32로 약 30장의 선별된 이미지로 학습된 LoRA |
| 생성된 이미지에 내 얼굴 넣기 | DreamBooth 또는 LoRA + IP-Adapter-FaceID |
| 특정 포즈 + 프롬프트 | ControlNet-Openpose + SDXL + 텍스트 |
| 깊이(Depth)를 인지하는 구도 | ControlNet-Depth + SD3 |
| 참조 이미지 + 프롬프트 | IP-Adapter + 텍스트 |
| 정확한 레이아웃 | ControlNet-Scribble 또는 ControlNet-Canny |
| 배경 교체 | ControlNet-Seg + Inpainting (Lesson 09) |
| 빠른 1단계 스타일 적용 | SDXL-Turbo 상의 LCM-LoRA |

## Ship It (실행하기)

`outputs/skill-sd-toolkit-composer.md`를 저장하세요. 이 스킬은 작업(입력 에셋: 프롬프트, 선택 사항인 참조 이미지, 선택 사항인 포즈, 선택 사항인 깊이, 선택 사항인 스크리블)을 입력받아 도구 스택(tool stack), 가중치(weights), 그리고 재현 가능한 시드 프로토콜(reproducible seed protocol)을 출력합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`에서 LoRA 랭크 `r`을 1에서 4까지 변화시켜 보세요. 어떤 랭크에서 LoRA가 랭크-2 타겟 델타(target delta)와 정확히 일치하나요?
2. **중간 (Medium).** 두 개의 서로 다른 타겟 변환(target transforms)에 대해 각각 별도의 LoRA를 학습시키세요. 이들을 함께 로드하여 가산적 상호작용(additive interaction)을 보여주세요. 언제 상호작용의 선형성(linearity)이 깨지나요?
3. **어려움 (Hard).** `diffusers`를 사용하여 다음을 스택(stack)하세요: SDXL-base + Canny-ControlNet (가중치 0.8) + 스타일 LoRA ($\alpha$ 0.8) + IP-Adapter (가중치 0.6). 스택 가중치가 변함에 따라 FID와 프롬프트 준수도(prompt-adherence) 간의 트레이드오프(trade-off)를 측정해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| ControlNet | "공간 제어 (Spatial control)" | 복제된 인코더 + zero-conv 스킵 연결; 컨디셔닝 이미지를 읽어들임. |
| Zero convolution | "항등 함수로 시작 (Starts as identity)" | 0으로 초기화된 1×1 conv; ControlNet은 아무런 동작을 하지 않는 상태(no-op)로 시작함. |
| LoRA | "저차원 어댑터 (Low-rank adapter)" | `W + B @ A`, `r << d`; 전체 미세 조정(full fine-tune)보다 파라미터 수가 100배 적음. |
| rank r | "조절 노브 (The knob)" | LoRA 압축 정도; 일반적으로 4-16을 사용하며, 강력한 개인화를 위해서는 64 이상을 사용함. |
| α | "LoRA 강도 (LoRA strength)" | 런타임 시 LoRA 델타(delta)의 스케일링 값. |
| IP-Adapter | "참조 이미지 (Reference image)" | CLIP-image 토큰을 통한 소규모 이미지 컨디셔닝 어댑터. |
| DreamBooth | "전체 피사체 미세 조정 (Full subject fine-tune)" | 특정 피사체의 이미지 약 30장을 사용하여 전체 모델을 학습함. |
| Textual Inversion | "새로운 토큰 (New token)" | 새로운 단어 임베딩(word embedding)만을 학습함; 레거시 방식이며 대부분 대체됨. |

## 프로덕션 노트: LoRA 스왑, ControlNet 레인, 멀티테넌트 서빙 (Production note: LoRA swaps, ControlNet lanes, multi-tenant serving)

실제 텍스트-이미지(text-to-image) SaaS는 동일한 베이스 체크포인트 위에서 수백 개의 LoRA와 수십 개의 ControlNet을 서비스합니다. 이러한 서빙 문제는 LLM의 멀티테넌시(multi-tenancy) 문제와 매우 유사합니다(프로덕션 문헌에서는 LLM 사례를 컨티뉴어스 배칭(continuous batching) 및 LoRAX / S-LoRA 관점에서 다룹니다):

- **LoRA를 병합하지 말고 핫스왑(Hot-swap)하세요.** `W' = W + α·B·A`를 베이스에 병합하면 스텝당 추론 속도가 약 3~5% 빨라지지만, `α`와 베이스 모델이 고정됩니다. LoRA를 rank-r 델타(delta) 형태로 VRAM에 핫 상태로 유지하세요. `diffusers`는 요청별 활성화를 위해 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`를 제공합니다. 스왑 비용은 `2 · d · r · num_layers` 가중치만큼이며, 이는 MB 단위의 규모로 1초 미만이 소요됩니다.
- **두 번째 어텐션 레인(attention lane)으로서의 ControlNet.** 복제된 인코더는 베이스 모델과 병렬로 실행됩니다. 가중치 1.0인 두 개의 ControlNet을 사용하면, 하나의 병합된 패스가 아니라 스텝당 두 번의 추가 순전파(forward pass)가 발생합니다. 이로 인해 배치 사이즈(batch-size) 여유 공간은 이차 함수적으로 감소합니다. 활성화된 ControlNet당 약 1.5배의 스텝 비용을 예산으로 잡으세요.
- **양자화된 LoRA(Quantized LoRAs)도 사용 가능합니다.** 베이스 모델을 양자화했다면(Lesson 07, 8GB에서의 Flux 참조), LoRA 델타 또한 8비트 또는 4비트로 깔끔하게 양자화됩니다. QLoRA 스타일의 로딩을 사용하면 메모리 폭발 없이 4비트 Flux 베이스 위에 5~10개의 LoRA를 쌓을 수 있습니다.

Flux 관련 특이사항: Niels의 Flux-on-8GB 노트북은 베이스를 4비트로 양자화합니다. 해당 양자화된 베이스 위에 `weight_name="pytorch_lora_weights.safetensors"`를 사용하여 스타일 LoRA(`pipe.load_lora_weights("user/style-lora")`)를 쌓아도 여전히 잘 작동합니다. 이것이 2026년에 대부분의 SaaS 에이전시가 배포하는 방식입니다.

## 추가 학습 자료 (Further Reading)

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (원래 LLM용으로 개발되었으나 확산 모델로 이식됨).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet의 가벼운 대안.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 참조용 파이프라인 문서.
