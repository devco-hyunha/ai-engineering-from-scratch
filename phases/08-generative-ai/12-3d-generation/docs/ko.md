# 3D 생성

> 3D는 2D-to-3D 활용도가 가장 높은 모달리티(Modality)입니다. 2023년의 돌파구는 3D 가우시안 스플래팅(3D Gaussian Splatting)이었습니다. 2024-2026년의 생성형 연구는 다중 뷰(multi-view) 확산 모델과 3D 재구성(reconstruction)을 결합하여, 단일 프롬프트나 사진으로 객체와 장면을 생성합니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 4단계 (비전), 8단계 · 07강 (잠재 확산)
**시간:** 약 45분

## 문제점

3D 콘텐츠는 다루기 어렵습니다:

- **표현.** 메시(mesh), 점 구름(point cloud), 복셀 그리드(voxel grid), 부호 거리 필드(SDF), 신경 방사 필드(NeRF), 3D 가우시안. 각각에 트레이드오프가 있습니다.
- **데이터 희소성.** ImageNet에는 1,400만 장의 이미지가 있습니다. 가장 큰 깨끗한 3D 데이터셋(Objaverse-XL, 2023)은 약 1,000만 개의 객체를 포함하지만, 대부분 품질이 낮습니다.
- **메모리.** 512³ 복셀 그리드는 1억 2,800만 개의 복셀이며, 유용한 장면 NeRF는 광선당 100만 개의 샘플이 필요합니다. 생성은 재구성보다 더 어렵습니다.
- **지도.** 2D 이미지의 경우 픽셀이 있습니다. 3D의 경우 일반적으로 몇 개의 2D 뷰만 있으며, 이를 3D로 리프트(lift)해야 합니다.

2026년 스택은 두 문제를 분리합니다. 먼저, 확산 모델로 *2D 다중 뷰 이미지*를 생성합니다. 그 다음, *3D 표현*(일반적으로 가우시안 스플래팅)을 해당 이미지에 맞춥니다.

## 개념

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### 표현: 3D 가우시안 스플래팅 (Kerbl et al., 2023)

장면을 약 100만 개의 3D 가우시안 구름으로 표현합니다. 각 가우시안은 59개의 매개변수를 가집니다: 위치 (3), 공분산 (6, 또는 쿼터니언 4 + 스케일 3), 불투명도 (1), 구면 조화 색 (3차에서 48, 0차에서 3).

렌더링 = 투영 + 알파 합성. 빠릅니다 (4090에서 1080p 기준 약 100 fps). 미분 가능합니다. 정답 사진에 대해 경사 하강법으로 맞춥니다. 소비자용 GPU에서 장면을 맞추는 데 5-30분이 걸립니다.

2023-2024년의 두 가지 혁신이 그 위에 추가되었습니다:
- **생성형 가우시안 스플랫.** LGM, LRM, InstantMesh 같은 모델은 하나 또는 몇 장의 이미지에서 가우시안 구름을 직접 예측합니다.
- **4D 가우시안 스플래팅.** 동적 장면을 위해 프레임별 오프셋을 가진 가우시안.

### 다중 뷰 확산

사전 학습된 이미지 확산 모델을 미세 조정하여 텍스트 프롬프트나 단일 이미지로부터 동일한 객체의 일관된 여러 뷰를 생성합니다. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). 일반적으로 객체 주변에 4-16개의 뷰를 출력하며, 가우시안 스플래팅(Gaussian splatting)이나 NeRF를 통해 3D로 리프트합니다.

### 텍스트-투-3D 파이프라인

| 모델 | 입력 | 출력 | 시간 |
|-------|-------|--------|------|
| DreamFusion (2022) | 텍스트 | SDS를 통한 NeRF | 자산당 약 1시간 |
| Magic3D | 텍스트 | 메시 + 텍스처 | 약 40분 |
| Shap-E (OpenAI, 2023) | 텍스트 | 암시적 3D | 약 1분 |
| SJC / ProlificDreamer | 텍스트 | NeRF / 메시 | 약 30분 |
| LRM (Meta, 2023) | 이미지 | 트라이플레인(triplane) | 약 5초 |
| InstantMesh (2024) | 이미지 | 메시 | 약 10초 |
| SV3D (Stability, 2024) | 이미지 | 신규 뷰 | 약 2분 |
| CAT3D (Google, 2024) | 1-64개 이미지 | 3D NeRF | 약 1분 |
| TripoSR (2024) | 이미지 | 메시 | 약 1초 |
| Meshy 4 (2025) | 텍스트 + 이미지 | PBR 메시 | 약 30초 |
| Rodin Gen-1.5 (2025) | 텍스트 + 이미지 | PBR 메시 | 약 60초 |
| Tencent Hunyuan3D 2.0 (2025) | 이미지 | 메시 | 약 30초 |

2025-2026 방향: 게임 엔진에 적합한 PBR 재질을 가진 직접 텍스트-투-메시 모델. 다중 뷰 확산 중간 단계는 여전히 범용 객체에 대해 가장 잘 수행되는 레시피입니다.

### NeRF (맥락)

신경 방사선 필드(Neural Radiance Field, Mildenhall et al., 2020). 작은 MLP가 `(x, y, z, view direction)`를 입력받아 `(color, density)`를 출력합니다. 광선을 따라 적분하여 렌더링합니다. 메시 기반 신규 뷰 합성보다 품질이 뛰어나지만 렌더링 속도는 100-1000배 느립니다. 대부분의 실시간 사용에서는 가우시안 스플래팅(Gaussian splatting)에 의해 대체되었지만, 연구 분야에서는 여전히 지배적입니다.

```figure
v4-3d-multiview
```

## 구현하기

`code/main.py`는 장난감 2D "가우시안 스플래팅" 적합을 구현합니다: 합성 타겟 이미지(부드러운 그라디언트)를 2D 가우시안 스플랫의 합으로 표현합니다. 타겟과 일치하도록 위치, 색상, 공분산을 경사 하강법으로 최적화합니다. 두 가지 핵심 연산인 순방향 렌더링(스플랫 + 알파 합성)과 경사 하강법을 통한 적합을 볼 수 있습니다.

### 1단계: 2D 가우시안 스플랫

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 2단계: 스플랫 합으로 렌더링

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

실제 3D 가우시안 스플래팅은 가우시안을 깊이 순으로 정렬하고 알파 합성을 순서대로 수행합니다. 우리의 2D 장난감은 단순히 합산합니다.

### 3단계: 경사 하강법으로 피팅

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 함정

- **시점 불일치.** 4개의 시점을 독립적으로 생성하여 객체 구조에 대한 의견이 불일치하면 3D 피팅이 흐려집니다. 해결책: 공유 어텐션을 사용하는 다중 시점 확산.
- **후면 환각.** 단일 이미지 → 3D는 보이지 않는 면을 만들어야 합니다. 품질이 극도로 변동합니다.
- **가우시안 스플래팅 폭발.** 제약 없는 훈련은 1000만 개의 스플래팅으로 성장하여 과적합됩니다. 밀도화 + 가지치기 휴리스틱(3D-GS 원 논문)이 필수입니다.
- **위상 문제.** 암시적 필드(SDF)에서 생성된 메시는 종종 구멍이 있거나 자기 교차가 발생합니다. 출시 전에 리메셔(blender의 voxel remesh 등)를 실행하세요.
- **훈련 데이터의 라이선스.** Objaverse는 혼합 라이선스를 가지고 있으며, 상업적 사용은 모델마다 다릅니다.

## 사용하기

| 작업 | 2026년 선택 |
|------|-----------|
| 사진에서 장면 재구성 | 가우시안 스플래팅 (3DGS, Gsplat, Scaniverse) |
| 게임을 위한 텍스트-3D 객체 | Meshy 4 또는 Rodin Gen-1.5 (PBR 출력) |
| 이미지-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 몇 장의 이미지에서 새로운 시점 합성 | CAT3D, SV3D |
| 동적 장면 재구성 | 4D 가우시안 스플래팅 |
| 아바타 / 의상 입은 인간 | Gaussian Avatar, HUGS |
| 연구 / SOTA | 지난 주에 발표된 것 |

게임이나 전자 상거래 파이프라인에서 생산용 3D를 출시하려면: Meshy 4 또는 Rodin Gen-1.5는 Unity / Unreal로 바로 들어가는 PBR 메시를 출력합니다.

## 출시하기

`outputs/skill-3d-pipeline.md`을 저장하세요. 스킬은 3D 브리프(입력: 텍스트 / 한 장의 이미지 / 몇 장의 이미지; 출력: 메시 / 스플래팅 / NeRF; 사용: 렌더링 / 게임 / VR)를 받아 파이프라인(다중 시점 확산 + 피팅, 또는 직접 메시 모델), 기본 모델, 반복 예산, 위상 후처리, 필요한 재료 채널을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`을 4, 16, 64개의 가우시안으로 실행하세요. 목표에 대한 최종 MSE를 보고하세요.
2. **중간.** 색상 가우시안(RGB)으로 확장하세요. 재구성이 목표 색상 패턴과 일치하는지 확인하세요.
3. **난이도: 상.** gsplat 또는 Nerfstudio를 사용하여 50장의 사진 촬영으로 실제 물체를 복원해 보세요. 적합 시간(fit time)과 홀드아웃 뷰(held-out views)에서의 최종 SSIM을 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | 장면을 3D 가우시안 구름으로 표현하며, 미분 가능한 알파 합성 렌더링을 수행합니다. |
| NeRF | "Neural radiance field" | 3D 지점에서 색상과 밀도를 출력하는 MLP이며, 레이 적분으로 렌더링합니다. |
| Triplane | "Three 2-D planes" | 3D를 세 개의 2D 축 정렬 특징 그리드로 분해하며, 볼륨 방식보다 저렴합니다. |
| SDS | "Score distillation sampling" | 2D 확산 점수(score)를 유사 기울기로 사용하여 3D 모델을 학습합니다. |
| Multi-view diffusion | "Many views at once" | 일관된 카메라 뷰의 배치를 출력하는 확산 모델입니다. |
| PBR | "Physically-based rendering" | 알베도, 거칠기, 금속성, 법선 채널을 포함하는 재질입니다. |
| Densification | "Grow splats" | 3DGS 학습 휴리스틱으로, 고경사 영역에서 스플릿을 분할/복제합니다. |

## 프로덕션 노트: 3D에는 아직 공유된 기반(substrate)이 없습니다

이미지(잠재 확산 + DiT)와 비디오(시공간 DiT)와 달리, 2026년 현재 3D에는 단일 지배적 런타임이 없습니다. 프로덕션 결정 트리는 표현 방식에 따라 갈라집니다:

- **NeRF / triplane.** 추론은 레이 마칭(ray-marching)과 샘플별 MLP 순전파로 이루어집니다. 512² 렌더링에는 수백만 번의 MLP 순전파가 필요합니다. 레이 샘플을 aggressively 배치(batch)하세요; SDPA/xformers가 적용됩니다.
- **Multi-view diffusion + LRM reconstruction.** 2단계 파이프라인입니다. 1단계(multi-view DiT)는 07강과 같은 확산 서버입니다. 2단계(LRM transformer)는 뷰에 대한 원샷 순전파입니다. 전체 지연 프로파일은 "확산 + 원샷"이므로, 각 단계의 서빙 프리미티브를 적절히 선택하세요.
- **SDS / DreamFusion.** 추론이 아닌 자산별 최적화입니다. 요청 핸들러가 아닌 빌드 잡(build jobs)을 구축하세요.

대부분의 2026년 제품에서는 "요청 시 multi-view diffusion 모델을 실행하고, 비동기적으로 3DGS로 복원하며, 실시간 뷰잉을 위해 3DGS를 서빙하는" 것이 정답입니다. 이는 GPU 추론 서버(빠름)와 오프라인 옵티마이저(느림) 간에 워크로드를 깔끔하게 분리합니다.

## 추가 읽기

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 다중 뷰 확산.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d-novel-multi-view-synthesis-and-3d-generation-from-a-single-image-using-latent-video-diffusion) — SV3D.
