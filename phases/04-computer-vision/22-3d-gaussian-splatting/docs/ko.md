# 처음부터 만드는 3D Gaussian Splatting (3D Gaussian Splatting from Scratch)

> 장면은 수백만 개의 3D Gaussian 구름입니다. 각각은 위치, 방향, 스케일, 불투명도, 그리고 시야 방향에 따라 달라지는 색을 가집니다. 래스터화하고, 래스터화를 통해 역전파하면 끝입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 13 (3D Vision & NeRF), Phase 1 Lesson 12 (Tensor Operations), Phase 4 Lesson 10 (Diffusion basics optional)
**Time:** ~90 minutes

## 학습 목표 (Learning Objectives)

- 2026년에 사실적 3D 재구성의 프로덕션 기본값이 NeRF에서 3D Gaussian Splatting으로 바뀐 이유를 설명합니다
- Gaussian당 여섯 파라미터(위치, 회전 쿼터니언, 스케일, 불투명도, 구면 조화 색, 선택적 feature)와 각각이 기여하는 float 개수를 말합니다
- `alpha` 합성으로 2D Gaussian splatting 래스터라이저를 처음부터 구현한 뒤, 3D 경우가 같은 루프로 투영되는 방식을 보입니다
- `nerfstudio`, `gsplat`, 또는 `SuperSplat`으로 20–50장 사진에서 장면을 재구성하고 `KHR_gaussian_splatting` glTF 확장 또는 OpenUSD 26.03 `UsdVolParticleField3DGaussianSplat` 스키마로보냅니다

## 문제 상황 (The Problem)

NeRF는 장면을 MLP 가중치로 저장합니다. 렌더링된 픽셀마다 광선을 따라 수백 번의 MLP 쿼리가 필요합니다. 학습은 수 시간, 렌더링은 수 초가 걸리고, 가중치는 편집할 수 없습니다 — 장면 안에서 의자를 옮기려면 다시 학습해야 합니다.

3D Gaussian Splatting(Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023)이 그 전부를 바꿨습니다. 장면은 명시적인 3D Gaussian 집합입니다. 렌더링은 100+ fps GPU 래스터화입니다. 학습은 수 분입니다. 편집은 직접적입니다: Gaussian 일부를 이동하면 의자가 움직입니다. 2026년까지 Khronos Group이 Gaussian splat용 glTF 확장을 비준했고, OpenUSD 26.03이 Gaussian splat 스키마를 실으며, Zillow와 Apartments.com이 부동산 렌더링에 쓰고, 3D 재구성의 새 연구 대부분은 핵심 3DGS 아이디어의 변형입니다.

멘탈 모델은 단순하지만, 수학에는 움직이는 부분이 많아 대부분의 소개가 래스터화부터 시작하고 투영과 구면 조화를 건너뜁니다. 이 레슨은 전체를 만듭니다 — 먼저 2D 버전, 그다음 3D 확장.

## 핵심 개념 (The Concept)

### Gaussian이 담는 것 (What a Gaussian carries)

하나의 3D Gaussian은 다음 속성을 가진 공간상의 파라미터 blob입니다:

```
position         mu         (3,)    centre in world coordinates
rotation         q          (4,)    unit quaternion encoding orientation
scale            s          (3,)    log-scales per axis (exponentiated at render time)
opacity          alpha      (1,)    post-sigmoid opacity [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   view-dependent colour
```

회전 + 스케일이 3x3 공분산을 만듭니다: `Sigma = R S S^T R^T`. 그것이 3D에서 Gaussian의 형태입니다. 구면 조화는 시야 방향에 따라 색이 바뀌게 합니다 — 스펙큘러 하이라이트, 미세한 광택, 시야 의존 glow — 뷰별 텍스처를 저장하지 않고. SH 차수 3이면 색 채널당 16계수, Gaussian당 색만으로 48 float입니다.

장면은 보통 1–5백만 Gaussian을 가집니다. 각각 대략 60 float(3 + 4 + 3 + 1 + 48 + misc)를 저장합니다. 5백만 Gaussian 장면이면 240 MB — 점별 텍스처가 있는 동등한 포인트 클라우드보다 훨씬 작고, 고해상도로 다시 렌더한 NeRF MLP 가중치보다 한 자릿수 작습니다.

### 레이 마칭이 아니라 래스터화 (Rasterisation, not ray marching)

```mermaid
flowchart LR
    SCENE["수백만 개의 3D Gaussian<br/>(위치, 회전, 스케일,<br/>불투명도, SH 색)"] --> PROJ["2D로 투영<br/>(카메라 extrinsics + intrinsics)"]
    PROJ --> TILES["타일에 할당<br/>(16x16 스크린 공간)"]
    TILES --> SORT["타일별<br/>깊이 정렬"]
    SORT --> ALPHA["앞에서 뒤로<br/>Alpha 합성"]
    ALPHA --> PIX["픽셀 색"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

다섯 단계, 모두 GPU 친화적입니다. 픽셀당 MLP 쿼리가 없습니다. RTX 3080 Ti 하나로 6백만 splat을 147 fps로 렌더합니다.

### 투영 단계 (The projection step)

월드 위치 `mu`와 3D 공분산 `Sigma`를 가진 3D Gaussian은 스크린 위치 `mu'`와 2D 공분산 `Sigma'`의 2D Gaussian으로 투영됩니다:

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'
```

2D Gaussian의 footprint는 `Sigma'`의 고유벡터를 축으로 하는 타원입니다. 그 타원 안의 모든 픽셀이 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`로 가중된 Gaussian 기여를 받습니다.

### Alpha 합성 규칙 (The alpha-compositing rule)

한 픽셀에 대해, 그 픽셀을 덮는 Gaussian을 뒤에서 앞으로(또는 동등하게 앞에서 뒤로 역공식으로) 정렬합니다. 색은 1980년대 이후 모든 반투명 래스터라이저와 같은 식으로 합성됩니다:

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       transmittance up to i
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   local contribution
c_i = eval_SH(SH_i, view_direction)    view-dependent colour
```

이것은 **NeRF의 체적 렌더와 같은 식**이며, 광선을 따라 조밀한 샘플 대신 명시적·희소한 Gaussian 집합 위에서 적분합니다. 그 동일성 때문에 렌더 품질이 NeRF와 맞습니다 — 둘 다 같은 radiance-field 식을 적분합니다.

### 왜 미분 가능한가 (Why this is differentiable)

모든 단계 — 투영, 타일 할당, alpha 합성, SH 평가 — 가 Gaussian 파라미터에 대해 미분 가능합니다. 정답 이미지가 있으면, 렌더된 픽셀 손실을 계산하고, 래스터라이저를 통해 역전파한 뒤, 경사하강법으로 모든 `(mu, q, s, alpha, c_lm)`을 갱신합니다. 약 30,000 반복 동안 Gaussian이 올바른 위치·스케일·색을 찾습니다.

### 밀도화와 가지치기 (Densification and pruning)

고정된 Gaussian 집합으로는 복잡한 장면을 덮을 수 없습니다. 학습에는 두 가지 적응 메커니즘이 있습니다:

- 기울기 크기가 크고 스케일이 작을 때 현재 위치에서 Gaussian을 **복제(Clone)** 합니다 — 여기에 더 세부가 필요합니다.
- 기울기가 크고 스케일이 큰 Gaussian을 두 개의 더 작은 것으로 **분할(Split)** 합니다 — 하나의 큰 Gaussian이 너무 부드러워 영역을 맞추지 못합니다.
- 불투명도가 임계값 아래로 떨어진 Gaussian을 **가지치기(Prune)** 합니다 — 기여하지 않습니다.

밀도화는 N 반복마다 돌아갑니다. 장면은 보통 ~100k 초기 Gaussian(SfM 점에서 시드)에서 학습 끝의 1–5M까지 자랍니다.

### 한 단락으로 보는 구면 조화 (Spherical harmonics in one paragraph)

시야 의존 색은 단위 구 위의 함수 `c(direction)`입니다. 구면 조화는 구의 푸리에 기저입니다. 차수 `L`에서 자르면 채널당 `(L+1)^2`개 기저 함수가 됩니다. 새 뷰에 대한 색 평가는 학습된 SH 계수와 시야 방향에서 평가한 기저의 내적입니다. 차수 0 = 계수 하나 = 상수 색. 차수 3 = 16계수 = Lambertian 셰이딩, 스펙큘러, 약한 반사를 담기에 충분합니다. SD Gaussian Splatting 논문은 기본적으로 차수 3을 씁니다.

### 2026 프로덕션 스택 (The 2026 production stack)

```
1. Capture         smartphone / DJI drone / handheld scanner
2. SfM / MVS       COLMAP or GLOMAP derives camera poses + sparse points
3. Train 3DGS      nerfstudio / gsplat / inria official / PostShot (~10-30 min on RTX 4090)
4. Edit            SuperSplat / SplatForge (clean floaters, segment)
5. Export          .ply -> glTF KHR_gaussian_splatting or .usd (OpenUSD 26.03)
6. View            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D와 생성형 변형 (4D and generative variants)

- **4D Gaussian Splatting** — Gaussian이 시간의 함수입니다; 체적 비디오에 쓰입니다(Superman 2026, A$AP Rocky의 "Helicopter").
- **생성형 splat** — 전체 장면을 환각하는 text-to-splat 모델(World Labs의 Marble).
- **3D Gaussian Unscented Transform** — NVIDIA NuRec의 자율주행 시뮬레이션용 변형.

```figure
cv3-gaussian-splat
```

## 직접 만들기 (Build It)

### Step 1: 2D Gaussian (A 2D Gaussian)

먼저 2D 래스터라이저를 만듭니다. 3D 경우는 투영 후 이것으로 줄어듭니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum`이 모든 (Gaussian, 픽셀) 쌍에 대해 이차 형식 `diff^T Sigma^-1 diff`를 계산합니다.

### Step 2: 2D splatting 래스터라이저 (2D splatting rasteriser)

앞에서 뒤로 Alpha 합성합니다. 2D에서 깊이는 의미가 없으므로, 순서를 위해 Gaussian당 학습된 스칼라를 씁니다.

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

빠르지는 않습니다 — 실제 구현은 타일 기반 CUDA 커널을 씁니다 — 하지만 수학은 정확히 같고 완전히 미분 가능합니다.

### Step 3: 학습 가능한 2D splat 장면 (A trainable 2D splat scene)

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`, `opacity_logit`, `colour_logits`는 모두 제약이 없는 파라미터이며 렌더 시점에 올바른 활성화를 통과합니다. 모든 3DGS 구현의 표준 패턴입니다.

### Step 4: 목표 이미지에 2D Gaussian 맞추기 (Fit 2D Gaussians to a target image)

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

200 스텝 동안 64개 Gaussian이 두 도형에 자리 잡습니다. 아이디어 전체가 이것입니다 — 명시적 기하 프리미티브에 대한 경사하강법.

### Step 5: 2D에서 3D로 (From 2D to 3D)

3D 확장은 같은 루프를 유지합니다. 추가분은 다음과 같습니다:

1. Gaussian당 회전이 단일 각 대신 쿼터니언입니다.
2. 공분산은 쿼터니언으로 만든 `R`과 `S = diag(exp(log_scale))`로 `R S S^T R^T`입니다.
3. 투영 `(mu, Sigma) -> (mu', Sigma')`는 카메라 extrinsics와 `mu`에서의 원근 투영 Jacobian을 씁니다.
4. 색이 구면 조화 전개가 됩니다; 시야 방향에서 평가합니다.
5. 깊이 정렬은 학습된 스칼라 대신 실제 카메라 공간 z입니다.

모든 프로덕션 구현(`gsplat`, `inria/gaussian-splatting`, `nerfstudio`)이 GPU에서 타일 기반 CUDA 커널로 정확히 이렇게 합니다.

### Step 6: 구면 조화 평가 (Spherical harmonics evaluation)

차수 3까지 SH 기저는 채널당 16항입니다. 평가:

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result
```

학습된 `sh_coeffs`는 그 Gaussian의 "모든 방향의 색"을 저장합니다. 렌더 시점에 현재 시야 방향에 대해 평가하면 3-벡터 RGB를 얻습니다.

## 활용하기 (Use It)

실제 3DGS 작업에는 `gsplat`(Meta) 또는 `nerfstudio`를 씁니다:

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto`는 nerfstudio의 3DGS 트레이너입니다. 전형적 장면에서 RTX 4090으로 10–30분 걸립니다.

2026년에 중요한보내기 옵션:

- `.ply` — 원시 Gaussian 구름(이식성 좋음, 파일 가장 큼).
- `.splat` — PlayCanvas / SuperSplat 양자화 포맷.
- glTF `KHR_gaussian_splatting` — Khronos 표준, 뷰어 간 이식(2026년 2월 RC).
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD 네이티브, NVIDIA Omniverse와 Vision Pro 파이프라인용.

4D / 동적 장면에는 `4DGS`와 `Deformable-3DGS`가 시간에 따라 변하는 mean과 opacity로 같은 기계를 확장합니다.

## 결과물 배포 (Ship It)

이 레슨이 만드는 것:

- `outputs/prompt-3dgs-capture-planner.md` — 주어진 장면 유형에 대해 촬영 세션(사진 수, 카메라 경로, 조명)을 계획하는 프롬프트.
- `outputs/skill-3dgs-export-router.md` — 다운스트림 뷰어나 엔진에 맞는보내기 포맷(`.ply` / `.splat` / glTF / USD)을 고르는 스킬.

## 연습 문제 (Exercises)

1. **(Easy)** 위 2D splat 트레이너를 다른 합성 이미지에서 돌립니다. `num_splats`를 `[16, 64, 256]`에서 바꿔 각 MSE vs step을 그리고, 수익 체감이 시작하는 지점을 찾습니다.
2. **(Medium)** 2D 래스터라이저를 확장해 스칼라 "시야 각"에 의존하는 Gaussian당 RGB 색을 차수-2 조화로 지원합니다. 목표 이미지 쌍으로 학습하고 모델이 둘 다 재구성하는지 확인합니다.
3. **(Hard)** `nerfstudio`를 클론하고 가진 장면(책상, 식물, 얼굴, 방)의 20장 촬영으로 `splatfacto`를 학습합니다. glTF `KHR_gaussian_splatting`으로보내고 뷰어(Three.js `GaussianSplats3D`, SuperSplat, Babylon.js V9)에서 엽니다. 학습 시간, Gaussian 수, 렌더 fps를 보고합니다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 3DGS | "Gaussian splat" | Gaussian당 위치·회전·스케일·불투명도·SH 색을 가진 수백만 3D Gaussian으로의 명시적 장면 표현 |
| Covariance | "Gaussian의 형태" | `Sigma = R S S^T R^T`; 한 Gaussian의 방향과 이방성 스케일 |
| Alpha compositing | "뒤에서 앞으로 블렌드" | NeRF 체적 렌더와 같은 식, 이제 명시적 희소 집합 위 |
| Densification | "복제와 분할" | 재구성이 부족한 곳에 새 Gaussian을 적응적으로 추가 |
| Pruning | "낮은 불투명도 삭제" | 학습 중 불투명도가 거의 0으로 붕괴한 Gaussian 제거 |
| Spherical harmonics | "시야 의존 색" | 구 위의 푸리에 기저; 시야 방향의 함수로 색을 저장 |
| Splatfacto | "nerfstudio의 3DGS" | 2026년에 3DGS를 학습하는 가장 쉬운 경로 |
| `KHR_gaussian_splatting` | "glTF 표준" | 뷰어와 엔진 간 3DGS를 이식 가능하게 하는 Khronos 2026 확장 |

## 더 읽을거리 (Further Reading)

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 원 논문
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 프로덕션급 CUDA 래스터라이저
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 참고 학습 레시피
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026 이식 포맷
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` 스키마
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 산업 개요
