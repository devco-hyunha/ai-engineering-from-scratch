---
name: stylegan-inversion
description: 실제 사진을 대상으로 사전 학습된 StyleGAN을 활용한 인버전(inversion) 및 편집 파이프라인을 선택합니다.
version: 1.0.0
phase: 8
lesson: 05
tags: [stylegan, inversion, editing]
---

실제 사진과 사전 학습된 StyleGAN 체크포인트(FFHQ-1024, StyleGAN-XL, 커스텀 파인튜닝 모델), 그리고 대상 편집 항목(나이, 미소, 포즈, 헤어, 정체성 유지)이 주어지면 다음을 출력합니다:

1. 인버전 방법(Inversion method): `e4e`(빠름, 낮은 충실도), `ReStyle`(반복적 인코더), `HyperStyle`(hypernet), `PTI`(pivotal tuning), 또는 직접적인 `W-optimization` 중 하나를 선택합니다. 충실도(fidelity)와 속도 간의 관계를 고려하여 한 문장으로 이유를 설명합니다.
2. 대상 공간(Target space): `W`, `W+`, 또는 `StyleSpace` 중 하나를 선택합니다. 트레이드오프를 고려합니다: `W`는 얽힘 해제(disentanglement)가 가장 잘 되어 있으나 충실도가 가장 낮고, `W+`는 레이어별 `w`를 사용하며, `StyleSpace`는 채널 수준을 다룹니다.
3. 편집 방향(Editing direction): 명명된 방향 소스인 `InterFaceGAN`(SVM 기반), `StyleSpace` 채널, `GANSpace` PCA, 또는 학습된 분류기(classifier) 중 하나를 선택합니다.
4. 충실도 예산(Fidelity budget): 정체성 드리프트(identity drift)가 발생하기 전의 `LPIPS` 임계값 및 롤백 휴리스틱(rollback heuristic)을 정의합니다.
5. 평가(Eval): ID 유사도(`ArcFace` 코사인 유사도), 원본과의 `LPIPS`, 편집 강도(대상 속성 분류기 점수)를 포함합니다.

`Z` 공간에서 직접 편집하여 얽힘(entanglement)이 발생하는 파이프라인은 거부합니다. 정체성 확인 없이 `W` 공간에서 큰 폭의 편집(`>1.5 sigma`)을 수행하는 것도 거부합니다. 오픈 도메인 편집(예: "그를 만화 캐릭터로 만들어줘")이 필요한 요청은 플래그를 표시합니다. 이러한 요청은 StyleGAN이 아닌 확산 모델(diffusion)과 `IP-Adapter`가 필요합니다.
