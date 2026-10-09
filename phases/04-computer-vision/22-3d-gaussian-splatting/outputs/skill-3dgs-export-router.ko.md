---
name: skill-3dgs-export-router
description: 다운스트림 뷰어나 엔진에 따라 올바른 3DGS 내보내기 형식(.ply / .splat / glTF KHR_gaussian_splatting / USD)을 선택합니다
version: 1.0.0
phase: 4단계
lesson: 22강
tags: [3d-gaussian-splatting, export, glTF, OpenUSD, pipeline]
---

# 3DGS 내보내기 라우터

다운스트림 대상을 올바른 3DGS 파일 형식에 매핑합니다. "불러오기가 안 된다"는 디버깅에 소요되는 몇 시간을 절약해 줍니다.

## 언제 사용해야 하는가

- 3DGS 장면을 학습한 후, 콘텐츠 파이프라인과 공유하기 전에 사용하세요.
- 연구용(.ply)과 생산용(glTF / USD) 형식 중 하나를 선택할 때 사용하세요.
- 파이프라인 핸드오프: 캡처 팀 -> 3DGS 엔지니어 -> 게임 디자이너 / VFX 아티스트 / 웹 개발자.

## 입력

- `target_engine`: unreal | unity | omniverse | blender | vision_pro | three_js | babylon_js | cesium | playcanvas | supersplat
- `priority`: portability | file_size | quality_preservation
- `include_sh_degree`: 0 | 1 | 2 | 3

## 형식 결정

| 대상 | 권장 형식 | 이유 |
|--------|--------------------|-----|
| Unreal Engine (가상 프로덕션) | Volinga 플러그인 또는 glTF KHR_gaussian_splatting | 네이티브 Unreal SDK 경로 |
| Unity (XR / 게임) | .ply (Aras-P Unity-GaussianSplatting 플러그인 사용) | 커뮤니티 표준 Unity 파이프라인 |
| NVIDIA Omniverse, Pixar 도구 | OpenUSD 26.03 (UsdVolParticleField3DGaussianSplat) | 네이티브 USD prim 타입 |
| Apple Vision Pro | OpenUSD 26.03 | visionOS 2.x에 네이티브 |
| Blender | .ply + KIRI Engine 애드온 | 커뮤니티 애드온이 원시 splat을 읽음 |
| Three.js 웹 뷰어 | glTF KHR_gaussian_splatting 또는 .splat | 브라우저 표준, `GaussianSplats3D`와 작동 |
| Babylon.js V9+ | glTF KHR_gaussian_splatting | V9에서 네이티브 지원 추가 |
| Cesium (CesiumJS 1.139+, Cesium for Unreal 2.23+) | glTF KHR_gaussian_splatting | 명시적 지원이 출시됨 |
| PlayCanvas | .splat | PlayCanvas 네이티브 양자화 형식 |
| SuperSplat (편집기) | .ply 또는 .splat | 가져오기 + 내보내기 |

## 양자화 절충안

- `.ply` 전체 정밀도: 가장 큰 파일, 무손실, 모든 뷰어.
- `.splat`: 4x-8x 더 작아짐, SH3 계수에 약간의 품질 손실, PlayCanvas 생태계 표준.
- glTF KHR: EXT_meshopt_compression으로 구성 가능; 가장 작으며 호환성이 가장 높음.
- USD: USDZ 패키징으로 압축; Apple 파이프라인에 가장 작음.

## 출력 보고서

```
[export plan]
  target:         <engine>
  format:         <name>
  sh degree:      <0|1|2|3>
  compression:    <none|meshopt|quantisation|usdz>
  expected size:  <MB>
  compatible with: <list of viewers>

[pipeline]
  1. source: <.ply from training>
  2. optional: SuperSplat cleanup pass
  3. convert: <tool + CLI or API call>
  4. package: <.gltf / .glb / .usd / .usdz / .splat / .ply>
  5. validate: <viewer sanity check>
```

## 규칙

- SH3 계수를 조용히 제거하지 마세요 — specular 반사가 눈에 띄게 변합니다.
- `priority == file_size`인 경우, `.splat` 또는 meshopt가 포함된 glTF를 권장하고 품질 손실에 대해 경고하세요.
- Apple 플랫폼에서는 2026년 glTF보다 USD / USDZ를 선호하세요; USDZ는 visionOS에 일급 지원이 있습니다.
- 대상 뷰어의 3DGS 지원이 표준 이전(pre-Feb 2026)인 경우, `.ply`과 뷰어의 커스텀 로더를 권장하세요; Khronos 표준 glTF는 아직 인식되지 않습니다.
- 인계하기 전에 최소한 하나의 뷰어에서 내보낸 파일을 항상 검증하세요; 양자화 과정에서 조용한 손상이 발생합니다.
