---
name: skill-3dgs-export-router
description: 다운스트림 뷰어나 엔진에 맞는 3DGS보내기 포맷(.ply / .splat / glTF KHR_gaussian_splatting / USD)을 고릅니다
version: 1.0.0
phase: 4
lesson: 22
tags: [3d-gaussian-splatting, export, glTF, OpenUSD, pipeline]
---

# 3DGS보내기 라우터 (3DGS Export Router)

다운스트림 타깃을 올바른 3DGS 파일 포맷에 매핑합니다. "로드되지 않음" 디버깅에 쓰는 시간을 절약합니다.

## 언제 쓰나요 (When to use)

- 3DGS 장면을 학습한 뒤, 콘텐츠 파이프라인과 공유하기 전.
- 연구급(.ply)과 프로덕션급(glTF / USD) 포맷 사이에서 고를 때.
- 파이프라인 핸드오프: 촬영팀 -> 3DGS 엔지니어 -> 게임 디자이너 / VFX 아티스트 / 웹 개발자.

## 입력 (Inputs)

- `target_engine`: unreal | unity | omniverse | blender | vision_pro | three_js | babylon_js | cesium | playcanvas | supersplat
- `priority`: portability | file_size | quality_preservation
- `include_sh_degree`: 0 | 1 | 2 | 3

## 포맷 결정 (Format decision)

| Target | Recommended format | Why |
|--------|--------------------|-----|
| Unreal Engine (virtual production) | Volinga plugin or glTF KHR_gaussian_splatting | Native Unreal SDK path |
| Unity (XR / game) | .ply via Aras-P Unity-GaussianSplatting plugin | Community-standard Unity pipeline |
| NVIDIA Omniverse, Pixar tools | OpenUSD 26.03 (UsdVolParticleField3DGaussianSplat) | Native USD prim type |
| Apple Vision Pro | OpenUSD 26.03 | Native to visionOS 2.x |
| Blender | .ply + KIRI Engine add-on | Community add-on reads raw splats |
| Three.js web viewer | glTF KHR_gaussian_splatting or .splat | Browser-standard, works with `GaussianSplats3D` |
| Babylon.js V9+ | glTF KHR_gaussian_splatting | V9 added native support |
| Cesium (CesiumJS 1.139+, Cesium for Unreal 2.23+) | glTF KHR_gaussian_splatting | Shipped explicit support |
| PlayCanvas | .splat | PlayCanvas native quantised format |
| SuperSplat (editor) | .ply or .splat | Import + export |

## 양자화 트레이드오프 (Quantisation trade-offs)

- `.ply` 전체 정밀도: 파일 가장 큼, 무손실, 모든 뷰어.
- `.splat`: 4x–8x 작음, SH3 계수에서 약간의 품질 손실, PlayCanvas 생태계 표준.
- glTF KHR: EXT_meshopt_compression으로 설정 가능; 호환성 최고이면서 가장 작음.
- USD: USDZ 패키징으로 압축; Apple 파이프라인에서 가장 작음.

## 출력 보고 (Output report)

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

## 규칙 (Rules)

- SH3 계수를 조용히 제거하지 마세요 — 스펙큘러 반사가 눈에 띄게 바뀝니다.
- `priority == file_size`이면 `.splat` 또는 meshopt가 있는 glTF를 추천하고 품질 손실을 경고하세요.
- Apple 플랫폼에서는 2026년에 glTF보다 USD / USDZ를 선호하세요; USDZ가 visionOS 일급 지원을 가집니다.
- 타깃 뷰어의 3DGS 지원이 표준 이전(2026년 2월 이전)이면 `.ply`와 뷰어의 커스텀 로더를 추천하세요; Khronos 표준 glTF는 아직 인식되지 않습니다.
- 핸드오프 전에 최소 한 뷰어에서보낸 파일을 항상 검증하세요; 양자화 중 조용한 손상이 일어납니다.
