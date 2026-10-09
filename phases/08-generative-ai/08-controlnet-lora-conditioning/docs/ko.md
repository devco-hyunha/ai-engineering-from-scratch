# ControlNet, LoRA 및 조건부 생성

> 텍스트만으로는 제어 신호가 거칠습니다. ControlNet은 사전 학습된 확산 모델을 복제하여 깊이 맵, 자세 골격, 낙서, 또는 가장자리 이미지로 모델을 제어할 수 있게 해줍니다. LoRA는 20억 파라미터 모델을 1,000만 파라미터만 학습하여 미세 조정할 수 있게 해줍니다. 이 두 기술은 Stable Diffusion을 장난감에서 모든 에이전시에서 출시되는 2026년 이미지 파이프라인으로 변화시켰습니다.

**유형:** Build
**언어:** Python
**선수 요건:** 8단계 · 07강 (잠재 확산), 10단계 (LLM从零부터 시작하기 — LoRA 기초)
**시간:** 약 75분

## 문제점

"바쁜 거리에서 개를 산책시키는 빨간 드레스를 입은 여성"과 같은 프롬프트는 모델에게 개가 *어디*에 있는지, 여성이 *어떤 자세*를 취하고 있는지, 거리의 *관점*이 무엇인지에 대한 정보를 제공하지 않습니다. 텍스트는 이미지를 지정하는 데 필요한 정보의 약 10%만 고정합니다. 나머지 부분은 시각적이며 단어로 효율적으로 설명할 수 없습니다.

각 신호(자세, 깊이, canny, 분할)마다 새로운 조건부 모델을 처음부터 학습하는 것은 불가능합니다. 26억 파라미터 SDXL 백본을 동결한 상태로 유지하고, 조건부 입력을 읽는 작은 보조 네트워크를 부착하여 백본의 중간 특징을 살짝 조정(nudge)하길 원합니다. 이것이 ControlNet입니다.

또한 전체 모델을 재학습하지 않고 모델에 새로운 개념(당신의 얼굴, 제품, 스타일)을 가르치길 원합니다. 100배 더 작은 델타를 원합니다. 이것이 LoRA입니다. 기존 어텐션 가중치에 연결되는 저랭크 어댑터입니다.

ControlNet + LoRA + 텍스트 = 2026년 실무자의 도구 키트입니다. 대부분의 생산용 이미지 파이프라인은 SDXL / SD3 / Flux 기반 위에 2-5개의 LoRA, 1-3개의 ControlNet, 그리고 IP-Adapter를 겹쳐서 사용합니다.

## 개념

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

사전 학습된 SD를 가져오세요. U-Net의 인코더 부분을 *복제*하세요. 원본은 동결하세요. 복제본이 추가 조건부 입력(가장자리, 깊이, 자세)을 받도록 학습하세요. 복제본을 원본의 디코더 부분과 *제로 컨볼루션* 스킵 연결(0으로 초기화된 1×1 컨볼루션 — 시작은 no-op, 델타를 학습)로 연결하세요.

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

제로 컨브 초기화(zero-conv init)는 ControlNet이 항등 변환(identity)으로 시작한다는 의미입니다. 즉, 학습 전에도 해를 끼치지 않습니다. 표준 확산 손실(standard diffusion loss)로 100만 개의 (프롬프트, 조건, 이미지) 삼중(triple)을 학습합니다.

모달리티별 ControlNet은 작은 보조 모델로 제공됩니다 (SDXL의 경우 약 360M, SD 1.5의 경우 약 70M). 추론 시 이를 조합할 수 있습니다:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

모델 내의 모든 선형 레이어 `W ∈ R^{d×d}`에 대해, `W`을 고정(freeze)하고 저랭크 델타(low-rank delta)를 추가합니다:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

`r << d`를 사용합니다. 어텐션의 경우 랭크 4-16이 표준이며, 무거운 미세 조정(fine-tune)의 경우 랭크 64-128을 사용합니다. 새로운 매개변수 수: `2 · d · r`는 `d²` 대신 사용됩니다. `d=640`, `r=16`를 사용하는 SDXL 어텐션의 경우, 어댑터당 410k가 아닌 20k 매개변수입니다. 이는 20배 감소입니다. 전체 모델에 걸쳐: LoRA는 보통 20-200MB인 반면, 기본 모델은 5GB입니다.

추론 시 LoRA를 스케일링(scale)할 수 있습니다: `W' = W + α · B @ A`. `α = 0.5-1.5`는 정상입니다. 여러 LoRA는 가산적으로 스택(stack)됩니다 (비선형적으로 상호작용한다는 일반적인 주의사항이 있습니다).

### IP-Adapter (Ye et al., 2023)

이미지를 조건(conditioning)으로 수용하는 (텍스트와 함께) 작은 어댑터입니다. CLIP 이미지 인코더를 사용하여 이미지 토큰을 생성하고, 텍스트 토큰과 함께 교차 어텐션(cross-attention)에 주입합니다. 기본 모델당 약 20MB입니다. LoRA 없이 "이 참조 이미지 스타일로 이미지를 생성"할 수 있습니다.

## 조합성 매트릭스

| 도구 | 제어 대상 | 크기 | 사용 시점 |
|------|------------------|------|-------------|
| ControlNet | 공간 구조 (포즈, 깊이, 엣지) | 70-360MB | 정확한 레이아웃, 구성 |
| LoRA | 스타일, 대상, 개념 | 20-200MB | 개인화, 스타일 |
| IP-Adapter | 참조 이미지로부터의 스타일 또는 대상 | 20MB | 텍스트로 외형을 설명할 수 없는 경우 |
| Textual Inversion | 새로운 토큰으로서의 단일 개념 | 10KB | 레거시, 대부분 LoRA로 대체됨 |
| DreamBooth | 대상에 대한 완전한 미세 조정 | 2-5GB | 강한 정체성, 높은 컴퓨팅 |
| T2I-Adapter | 더 가벼운 ControlNet 대안 | 70MB | 엣지 디바이스, 추론 예산 |

ControlNet ≈ 공간적. LoRA ≈ 의미론적. 둘 다 사용하세요.

```figure
v4-controlnet-zero
```

## 구현하기

`code/main.py`는 1-D에서 두 메커니즘을 시뮬레이션합니다:

1. **LoRA.** 사전 학습된 선형 레이어 `W`. 이를 동결하세요. `W + BA`가 목표 선형 레이어와 일치하도록 낮은 랭크의 `B @ A`를 학습하세요. `r = 1`가 랭크 1 보정을 완벽하게 학습하는 데 충분함을 보이세요.

2. **ControlNet-lite.** "동결된 기본(base)" 예측기와 추가 신호를 읽는 "사이드 네트워크(side network)"가 있습니다. 사이드 네트워크의 출력은 0으로 초기화된 학습 가능한 스칼라 게이트로 제어됩니다(우리의 zero-conv 버전). 학습하여 게이트가 상승하는 것을 관찰하세요.

### 1단계: LoRA 수학

```python
def lora(W, A, B, x, alpha=1.0):
    # W는 동결되어 있으며, A와 B는 학습 가능한 낮은 랭크의 인자(factor)입니다.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 2단계: zero-init 사이드 네트워크

```python
side_out = control_net(x, condition)
gated = gate * side_out  # 게이트가 0으로 초기화됨
h = base(x) + gated
```

0단계에서 출력은 기본(base) 모델과 동일합니다. 초기 학습 동안 `gate`는 천천히 업데이트됩니다 — 치명적인 드리프트(drift)가 발생하지 않습니다.

## 함정

- **LoRA 과잉 스케일링.** `α = 2` 또는 `α = 3`는 "더 강하게 만들기" 위한 흔한 해킹 기법으로, 과하게 스타일화되거나 깨진 출력을 생성합니다. `α ≤ 1.5`를 유지하세요.
- **ControlNet 가중치 충돌.** Pose ControlNet을 가중치 1.0으로, Depth ControlNet을 가중치 1.0으로 사용하는 것은 보통 과잉 보정(overshoot)을 일으킵니다. 가중치의 합이 ≈ 1.0인 것이 안전한 기본값입니다.
- **잘못된 기본 모델에 LoRA 적용.** SDXL LoRA는 어텐션 차원이 일치하지 않아 SD 1.5에서 조용히 아무 동작도 하지 않습니다(no-op). Diffusers는 0.30+ 버전에서 경고합니다.
- **Textual Inversion 드리프트.** 한 체크포인트에서 학습된 토큰은 다른 체크포인트에서 심하게 드리프트됩니다. LoRA는 더 이식성이 좋습니다.
- **LoRA 가중치 병합 및 저장.** LoRA를 기본 모델 가중치에 구워(bake) 넣어 추론 속도를 높일 수 있습니다(런타임 추가 없음). 하지만 런타임에 `α`를 스케일링하는 능력을 잃게 됩니다. 두 버전을 모두 보관하세요.

## 사용하기

| 목표 | 2026 파이프라인 |
|------|---------------|
| 브랜드의 아트 스타일 재현 | 랭크 32로 약 30장의 큐레이션된 이미지로 학습된 LoRA |
| 생성된 이미지에 내 얼굴 넣기 | DreamBooth 또는 LoRA + IP-Adapter-FaceID |
| 특정 포즈 + 프롬프트 | ControlNet-Openpose + SDXL + 텍스트 |
| 깊이(depth) 인식 구성 | ControlNet-Depth + SD3 |
| 참조(reference) + 프롬프트 | IP-Adapter + 텍스트 |
| 정확한 레이아웃 | ControlNet-Scribble 또는 ControlNet-Canny |
| 배경 교체 | ControlNet-Seg + Inpainting (09강) |
| 빠른 1단계 스타일 | SDXL-Turbo에 LCM-LoRA 적용 |

## 출시하기

`outputs/skill-sd-toolkit-composer.md`을 저장하세요. 스킬은 작업(입력 자산: 프롬프트, 선택적 참조 이미지, 선택적 포즈, 선택적 깊이, 선택적 낙서)을 받아 도구 스택, 가중치, 재현 가능한 시드 프로토콜을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`에서 LoRA 랭크 `r`을 1부터 4까지 변경해 보세요. 어떤 랭크에서 LoRA가 랭크-2 타겔 델타와 정확히 일치합니까?
2. **중간.** 두 개의 타겔 변환에 각각 별도의 LoRA를 학습하세요. 이를 함께 로드하여 가산 상호작용을 보여주세요. 상호작용이 선형성을 깨뜨리는 시점은 언제입니까?
3. **어려움.** diffusers를 사용하여 스택을 구성하세요: SDXL-base + Canny-ControlNet (가중치 0.8) + 스타일 LoRA (α 0.8) + IP-Adapter (가중치 0.6). 스택 가중치가 변함에 따라 FID 대 프롬프트 준수(trade-off)를 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| ControlNet | "공간 제어" | 복제된 인코더 + zero-conv 스킵; 조건부 이미지를 읽습니다. |
| Zero convolution | "항등 변환으로 시작" | 1×1 conv가 0으로 초기화됨; ControlNet은 no-op으로 시작합니다. |
| LoRA | "저랭크 적응" | `W + B @ A`, `r << d`; 전체 미세 조정보다 매개변수가 100배 적습니다. |
| 랭크 r | "조절 노브" | LoRA 압축; 일반적으로 4-16, 무거운 개인화에는 64+를 사용합니다. |
| α | "LoRA 강도" | LoRA 델타의 런타임 스케일링입니다. |
| IP-Adapter | "참조 이미지" | CLIP-image 토큰을 통한 소형 이미지 조건부 어댑터입니다. |
| DreamBooth | "전체 대상 미세 조정" | 대상의 약 30개 이미지로 전체 모델을 학습합니다. |
| Textual Inversion | "새 토큰" | 새로운 단어 임베딩만 학습합니다; 레거시이며 대부분 대체되었습니다. |

## 프로덕션 노트: LoRA 교체, ControlNet 레인, 다중 테넌트 서빙

실제 텍스트-이미지 SaaS는 동일한 기본 체크포인트 위에서 수백 개의 LoRA와 수십 개의 ControlNet을 서빙합니다. 서빙 문제는 LLM 다중 테넌트와 매우 유사합니다(프로덕션 문헌은 LLM의 경우를 연속 배치 및 LoRAX / S-LoRA 아래에서 다루고 있습니다):

- **LoRA를 핫 스왑하고, 병합하지 마세요.** `W' = W + α·B·A`을 기본 모델에 병합하면 단계별 추론이 약 3-5% 빨라지지만 `α`과 기본 모델이 고정됩니다. LoRA를 VRAM에 랭크-r 델타로 핫 상태로 유지하세요. diffusers는 요청별 활성화를 위해 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`을 제공합니다. 스왑 비용은 `2 · d · r · num_layers` 가중치이며, MB 단위이고 1초 미만입니다.
- **ControlNet을 두 번째 어텐션 레인으로 사용하세요.** 클론된 인코더는 기본 모델과 병렬로 실행됩니다. 가중치가 각각 1.0인 ControlNet 두 개는 단계당 두 번의 추가 순방향 패스를 의미하며, 하나의 병합된 패스가 아닙니다. 배치 크기 여유가 2차적으로 감소합니다. 활성 ControlNet당 단계 비용이 약 1.5배 증가하는 것을 고려하세요.
- **양자화된 LoRA도 가능합니다.** 기본 모델을 양자화한 경우 (07강 참고, 8GB에서의 Flux), LoRA 델타도 8비트 또는 4비트로 깔끔하게 양자화됩니다. QLoRA 스타일 로딩을 사용하면 4비트 Flux 기본 모델 위에 5-10개의 LoRA를 메모리를 초과하지 않고 쌓을 수 있습니다.

Flux 전용: Niels의 Flux-on-8GB 노트북은 기본 모델을 4비트로 양자화합니다. 양자화된 기본 모델에 `weight_name="pytorch_lora_weights.safetensors"`에서 스타일 LoRA (`pipe.load_lora_weights("user/style-lora")`)를 쌓아도 여전히 작동합니다. 이것은 2026년 SaaS 에이전시가 가장 많이 출시하는 레시피입니다.

## 추가 읽기

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (원래 LLM용; 확산 모델로 이식됨).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet의 더 가벼운 대안.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 참조 파이프라인.
