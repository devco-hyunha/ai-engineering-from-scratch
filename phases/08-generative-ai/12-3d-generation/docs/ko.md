# 3D 생성 (3D Generation)

> 3D는 2D-to-3D 활용도가 가장 강력한 모달리티(modality)입니다. 2023년의 돌파구는 3D Gaussian Splatting이었습니다. 2024-2026년의 생성형 추진력은 단일 프롬프트나 사진으로부터 객체와 장면을 생성하기 위해 다중 뷰 확산(multi-view diffusion)과 3D 재구성(3D reconstruction)을 상단에 쌓아 올리는 구조를 가집니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 minutes

## 문제점 (The Problem)

3D 콘텐츠 제작은 매우 까다롭습니다:

- **표현 방식 (Representation).** 메쉬(Meshes), 포인트 클라우드(point clouds), 복셀 그리드(voxel grids), 부호 거리 함수(SDFs), 신경 복사장(NeRFs), 3D 가우시안(3D Gaussians) 등이 있습니다. 각 방식은 저마다의 트레이드오프(trade-offs)가 존재합니다.
- **데이터 부족 (Data scarcity).** ImageNet에는 1,400만 장의 이미지가 있습니다. 가장 큰 규모의 정제된 3D 데이터셋(Objaverse-XL, 2023)은 약 1,000만 개의 객체를 포함하고 있지만, 대부분 품질이 낮습니다.
- **메모리 (Memory).** $512^3$ 복셀 그리드는 1억 2,800만 개의 복셀을 가집니다. 유용한 장면을 구현하는 NeRF는 광선(ray)당 100만 개의 샘플이 필요합니다. 생성(Generation)은 재구성(Reconstruction)보다 훨씬 어렵습니다.
- **지도 학습 (Supervision).** 2D 이미지의 경우 픽셀 데이터가 존재합니다. 하지만 3D의 경우 대개 소수의 2D 뷰(view)만을 가지고 있으며, 이를 3D로 끌어올려야(lift) 합니다.

2026년의 기술 스택은 이 두 문제를 분리합니다. 첫째, 확산 모델(diffusion model)을 사용하여 *2D 다중 뷰 이미지(2D multi-view images)*를 생성합니다. 둘째, 해당 이미지들에 *3D 표현 방식(3D representation)*(주로 가우시안 스플래팅)을 맞춥니다(fit).

## 개념 (The Concept)

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### 표현 방식: 3D 가우시안 스플래팅 (3D Gaussian Splatting; Kerbl et al., 2023)

장면을 약 100만 개의 3D 가우시안(3D Gaussians) 구름으로 표현합니다. 각 가우시안은 59개의 파라미터를 가집니다: 위치(position, 3), 공분산(covariance, 6 또는 쿼터니언 4 + 스케일 3), 불투명도(opacity, 1), 구면 조화 함수 색상(spherical-harmonics color, 차수 3일 때 48, 차수 0일 때 3).

렌더링(Rendering) = 투영(projection) + 알파 합성(alpha-compositing)입니다. 매우 빠릅니다(RTX 4090 기준 1080p에서 약 100 fps). 미분 가능(Differentiable)하며, 실제 사진(ground-truth photos)을 대상으로 경사 하강법(gradient descent)을 통해 최적화합니다. 소비자용 GPU에서 장면 하나를 맞추는 데 5~30분이 소요됩니다.

그 위에 더해진 2023-2024년의 두 가지 혁신 기술은 다음과 같습니다:
- **생성형 가우시안 스플랫(Generative Gaussian splats).** LGM, LRM, InstantMesh와 같은 모델은 한 장 또는 몇 장의 이미지로부터 가우시안 구름을 직접 예측합니다.
- **4D 가우시안 스플래팅(4D Gaussian Splatting).** 동적 장면(dynamic scenes)을 위해 프레임별 오프셋(per-frame offsets)을 가진 가우시안을 사용합니다.

### 멀티뷰 확산 모델 (Multi-view diffusion)

텍스트 프롬프트나 단일 이미지를 기반으로 동일한 객체의 일관된 여러 뷰(view)를 생성하도록 사전 학습된 이미지 확산 모델을 미세 조정(Fine-tune)합니다. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024) 등이 이에 해당합니다. 일반적으로 객체 주변의 4~16개 뷰를 출력하며, 이를 가우시안 스플래팅(Gaussian splatting) 또는 NeRF를 통해 3D로 변환합니다.

### Text-to-3D 파이프라인 (Text-to-3D pipelines)

| 모델 (Model) | 입력 (Input) | 출력 (Output) | 소요 시간 (Time) |
|-------|-------|--------|------|
| DreamFusion (2022) | text | SDS를 통한 NeRF | 에셋당 ~1시간 |
| Magic3D | text | mesh + texture | ~40분 |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1분 |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30분 |
| LRM (Meta, 2023) | image | triplane | ~5초 |
| InstantMesh (2024) | image | mesh | ~10초 |
| SV3D (Stability, 2024) | image | novel views | ~2분 |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1분 |
| TripoSR (2024) | image | mesh | ~1초 |
| Meshy 4 (2025) | text + image | PBR mesh | ~30초 |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60초 |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30초 |

2025-2026년 방향성: 게임 엔진에 적합한 PBR 재질을 포함한 직접적인 text-to-mesh 모델. 일반적인 객체에 대해서는 다중 뷰 확산(Multi-view diffusion)을 중간 단계로 사용하는 방식이 여전히 가장 성능이 좋은 레시피입니다.

### NeRF (문맥 참조용)

Neural Radiance Field (Mildenhall et al., 2020). 아주 작은 MLP가 `(x, y, z, view direction)`을 입력받아 `(color, density)`를 출력합니다. 광선(rays)을 따라 적분하여 렌더링합니다. 품질 면에서는 메쉬(mesh) 기반의 새로운 시점 합성(novel-view synthesis)보다 뛰어나지만, 렌더링 속도는 100~1000배 더 느립니다. 대부분의 실시간 용도로는 Gaussian splatting에 의해 대체되었으나, 연구 분야에서는 여전히 지배적입니다.

```figure
v4-3d-multiview
```

## 구현하기 (Build It)

`code/main.py`는 간단한 2D "가우시안 스플래팅(Gaussian splatting)" 피팅을 구현합니다. 합성 타겟 이미지(매끄러운 그라데이션)를 2D 가우시안 스플랫(Gaussian splats)의 합으로 표현합니다. 타겟과 일치하도록 경사 하강법(gradient descent)을 사용하여 위치, 색상 및 공분산(covariances)을 최적화합니다. 여기에서 두 가지 핵심 연산인 순방향 렌더링(forward render: 스플랫 + 알파 합성)과 경사 하강법을 통한 피팅(fit by gradient descent)을 확인할 수 있습니다.

### 1단계: 2D 가우시안 스플랫 (2D Gaussian splat)

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 2단계: 스플랫(splats) 합산으로 렌더링하기

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

실제 3D 가우시안 스플래팅(3D Gaussian splatting)은 가우시안을 깊이(depth) 순으로 정렬한 뒤 순서대로 알파 합성(alpha-compositing)을 수행합니다. 우리의 2D 장난감 모델은 단순히 합산만 수행합니다.

### 3단계: 경사 하강법(Gradient Descent)을 통한 학습(fit)

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 주의 사항 (Pitfalls)

- **뷰 불일치 (View inconsistency).** 4개의 뷰를 독립적으로 생성했을 때 객체의 구조에 대해 서로 다른 정보를 제공하면, 3D 피팅(fit) 결과가 흐릿해집니다. 해결책: 공유 어텐션(shared attention)을 사용하는 멀티뷰 디퓨전(multi-view diffusion)을 활용해 보세요.
- **뒷면 환각 (Back-side hallucination).** 단일 이미지에서 3D를 생성할 때는 보이지 않는 측면을 새로 만들어내야 합니다. 이 과정에서 품질이 크게 달라질 수 있습니다.
- **가우시안 스플랫 폭발 (Gaussian splat explosion).** 제약 없는 학습은 스플랫(splat)의 수를 1,000만 개까지 늘려 과적합(overfitting)을 유발합니다. 밀도화(Densification) 및 가지치기(pruning) 휴리스틱(3D-GS 원본 논문 참조)이 필수적입니다.
- **위상 문제 (Topology issues).** 암시적 필드(implicit fields, SDFs)로부터 생성된 메쉬(mesh)는 구멍이 있거나 자기 교차(self-intersections)가 발생하는 경우가 많습니다. 결과물을 배포하기 전에 리메셔(remesher, 예: Blender의 voxel remesh)를 실행해 보세요.
- **학습 데이터 라이선스 (License of training data).** Objaverse는 라이선스가 혼합되어 있습니다. 상업적 이용 가능 여부는 모델마다 다를 수 있습니다.

## 활용하기 (Use It)

| 작업 (Task) | 2026년 추천 (2026 pick) |
|------|-----------|
| 사진 기반 장면 재구성 (Scene reconstruction from photos) | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| 게임용 텍스트-3D 객체 생성 (Text-to-3D object for games) | Meshy 4 또는 Rodin Gen-1.5 (PBR 출력) |
| 이미지-3D 변환 (Image-to-3D) | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 소수 이미지를 활용한 새로운 시점 합성 (Novel-view synthesis from few images) | CAT3D, SV3D |
| 동적 장면 재구성 (Dynamic scene reconstruction) | 4D Gaussian Splatting |
| 아바타 / 의상을 입은 사람 (Avatar / clothed human) | Gaussian Avatar, HUGS |
| 연구 / 최신 기술 (Research / SOTA) | 지난주에 새로 나온 것 (Whatever dropped last week) |

게임이나 이커머스 파이프라인에 프로덕션급 3D를 적용하려면 Meshy 4 또는 Rodin Gen-1.5를 사용해 보세요. 이들은 Unity / Unreal로 바로 가져갈 수 있는 PBR 메쉬를 출력합니다.

## Ship It

`outputs/skill-3d-pipeline.md`를 저장하세요. 이 기술(Skill)은 3D 브리프(입력: 텍스트 / 이미지 한 장 / 여러 장의 이미지; 출력: mesh / splat / NeRF; 용도: 렌더링 / 게임 / VR)를 입력받아 다음을 출력합니다: 파이프라인(multi-view diffusion + fit, 또는 직접적인 mesh 모델), 베이스 모델, 반복 예산(iteration budget), 토폴로지 후처리, 필요한 재질 채널(material channels).

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** 4개, 16개, 64개의 가우시안(Gaussian)을 사용하여 `code/main.py`를 실행해 보세요. 최종 MSE와 타겟(target) 값을 비교하여 보고하세요.
2. **중간 (Medium).** 컬러 가우시안(RGB)으로 확장해 보세요. 재구성된 결과가 타겟 색상 패턴과 일치하는지 확인하세요.
3. **어려움 (Hard).** `gsplat` 또는 `Nerfstudio`를 사용하여 50장의 사진 촬영본으로부터 실제 객체를 재구성해 보세요. 학습 시간(fit time)과 테스트 뷰(held-out views)에서의 최종 SSIM을 보고하세요.

## 주요 용어 (Key Terms)

| 용어 | 흔히 말하는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | 장면을 3D 가우시안(Gaussian) 구름으로 표현; 미분 가능한 알파 합성(alpha-composite) 렌더링. |
| NeRF | "Neural radiance field" | 3D 지점의 색상(color)과 밀도(density)를 출력하는 MLP; 광선 적분(ray integration)을 통한 렌더링. |
| Triplane | "세 개의 2D 평면" | 3D 데이터를 축에 정렬된 세 개의 2D 특징 그리드(feature grids)로 분해; 볼륨 방식보다 비용이 저렴함. |
| SDS | "Score distillation sampling" | 2D 확산(diffusion) 점수를 의사 그래디언트(pseudo-gradient)로 사용하여 3D 모델을 학습. |
| Multi-view diffusion | "한 번에 여러 뷰" | 일관된 카메라 뷰의 배치를 출력하는 확산 모델. |
| PBR | "Physically-based rendering" | 알베도(albedo), 거칠기(roughness), 금속성(metallic), 노멀(normal) 채널을 가진 재질. |
| Densification | "스플랫(splat) 키우기" | 3DGS 학습 휴리스틱: 그래디언트가 높은 영역에서 스플랫을 분할(split) 또는 복제(clone)함. |

## 프로덕션 노트: 3D는 아직 공유된 기반 기술(shared substrate)이 없습니다

이미지(latent diffusion + DiT) 및 비디오(spatiotemporal DiT)와 달리, 2026년 현재 3D 분야에는 지배적인 단일 런타임이 존재하지 않습니다. 프로덕션 결정 트리(decision tree)는 표현 방식(representation)에 따라 갈라집니다:

- **NeRF / triplane.** 추론은 레이 마칭(ray-marching)과 샘플당 MLP 순전파(forward)로 이루어집니다. 512² 렌더링에는 수백만 번의 MLP 순전파가 필요합니다. 레이 샘플을 공격적으로 배치(batch) 처리해야 하며, SDPA/xformers를 적용할 수 있습니다.
- **Multi-view diffusion + LRM reconstruction.** 2단계 파이프라인입니다. 1단계(multi-view DiT)는 Lesson 07과 동일한 디퓨전 서버입니다. 2단계(LRM transformer)는 뷰(views)들에 대한 원샷(one-shot) 순전파 패스입니다. 전체 지연 시간(latency) 프로필은 "diffusion + one-shot" 형태이므로, 각 단계에 적합한 서빙 프리미티브(serving primitives)를 선택해야 합니다.
- **SDS / DreamFusion.** 추론이 아닌 자산별 최적화(per-asset optimization) 방식입니다. 요청 처리기(request handler)가 아닌 빌드 작업(build job)으로 구축해야 합니다.

2026년의 대부분의 제품에 대한 정답은 "요청 시 멀티뷰 디퓨전 모델을 실행하고, 비동기적으로 3DGS로 재구성한 뒤, 실시간 시청을 위해 3DGS를 서빙한다"입니다. 이는 작업 부하를 GPU 추론 서버(빠름)와 오프라인 최적화기(느림)로 깔끔하게 분리합니다.

## 추가 읽을거리 (Further Reading)

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 다중 뷰 확산(multi-view diffusion).
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.
