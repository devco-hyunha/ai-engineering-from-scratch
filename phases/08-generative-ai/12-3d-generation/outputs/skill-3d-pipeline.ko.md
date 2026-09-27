---
name: 3d-pipeline
description: 입력 유형, 출력 형식 및 사용 사례에 따라 3D 생성 또는 재구성 파이프라인을 선택합니다.
version: 1.0.0
phase: 8
lesson: 12
tags: [3d, gaussian-splatting, nerf, mesh]
---

입력값(텍스트 프롬프트 / 단일 이미지 / 소수 이미지 / 사진 촬영 / 비디오), 목표 출력(mesh / Gaussian splat / NeRF / point cloud), 그리고 사용 사례(실시간 렌더링, 게임 엔진, AR / VR, 시네마틱)가 주어지면 다음을 출력하세요:

1. 파이프라인(Pipeline). (a) 다중 뷰 확산 + 3D 피팅(Multi-view diffusion + 3D fit) (SV3D, CAT3D + 3DGS), (b) 직접 단일 샷(direct single-shot) (LRM, TripoSR, InstantMesh), (c) PBR을 포함한 텍스트-투-메쉬(text-to-mesh with PBR) (Meshy 4, Rodin Gen-1.5, Hunyuan3D 2.0), (d) 사진 촬영 + 3DGS(photo capture + 3DGS) (Gsplat, Postshot, Scaniverse).
2. 베이스 모델 + 호스팅(Base model + hosting). 모델명 + 오픈 소스 / 호스팅 여부. 상업적 이용을 위한 라이선스 관련 정보 포함.
3. 반복 예산(Iteration budget). 첫 결과물 도출까지 예상되는 시간, 반복 비용, 정교화 전략(refinement strategy).
4. 토폴로지 + 재질(Topology + materials). 리메쉬(Remesh) 단계가 필요한가? PBR 채널 요구 사항(albedo, roughness, metallic, normal)? UV 레이아웃이 자동인가 수동인가?
5. 평가(Eval). 홀드아웃 뷰(held-out views)에 대한 SSIM, CLIP 점수, 메쉬의 수밀성(watertightness), 폴리곤 수(poly count), 텍스처 해상도.
6. 플랫폼 타겟(Platform target). Unity / Unreal / Blender / 웹(three.js / Babylon) / AR (USDZ / glb).

메쉬 변환 단계(mesh conversion pass) 없이 3DGS를 게임 엔진에 직접 사용하는 것은 거부하세요(대부분의 엔진은 splat을 네이티브로 렌더링하지 못합니다). 복잡한 관절형 캐릭터(articulated characters)에 대해 텍스트-투-3D를 사용하는 것은 거부하고, 대신 리깅 인지 파이프라인(rigging-aware pipeline)을 사용하세요. 다운스트림 도구(DCC 도구 등)가 NeRF를 렌더링할 수 없는 경우, NeRF 전용 출력을 경고(flag)하세요.
