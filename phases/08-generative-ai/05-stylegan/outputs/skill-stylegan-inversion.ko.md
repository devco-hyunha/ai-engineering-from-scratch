---
name: stylegan-inversion
description: 실제 사진에 대해 사전 학습된 StyleGAN에 대한 역전환 및 편집 파이프라인을 선택합니다.
version: 1.0.0
phase: 8단계
lesson: 05강
tags: [stylegan, inversion, editing]
---

실제 사진 + 사전 학습된 StyleGAN 체크포인트(FFHQ-1024, StyleGAN-XL, 사용자 지정 미세 조정)와 목표 편집(나이, 미소, 자세, 머리, 정체성 보존)이 주어지면 다음을 출력합니다:

1. 역전환 방법. e4e (빠름, 낮은 충실도), ReStyle (반복 인코더), HyperStyle (하이퍼넷), PTI (피벗 튜닝) 또는 직접 W-최적화. 충실도 대 속도와 연결된 한 문장 이유를 제시합니다.
2. 목표 공간. W, W+ 또는 StyleSpace. 트레이드오프: W = 가장 분해(disentangled)되지만 충실도는 가장 낮음, W+ = 레이어별 w, StyleSpace = 채널 수준.
3. 편집 방향. 명명된 방향 출처: InterFaceGAN (SVM 기반), StyleSpace 채널, GANSpace PCA 또는 학습된 분류기.
4. 충실도 예산. 정체성 드리프트 전의 LPIPS 임계값; 롤백 휴리스틱.
5. 평가. ID 유사성 (ArcFace 코사인), 원본에 대한 LPIPS, 편집 강도 (목표 속성 분류기 점수).

Z에서 직접 편집하는 파이프라인은 거부합니다 (entangled). W에서 큰 편집(&gt;1.5 sigma)은 정체성 검사 없이 거부합니다. 오픈 도메인 편집이 필요한 요청(예: "그를 만화 캐릭터로 만들어 주세요")은 플래그를 지정합니다 - StyleGAN이 아니라 diffusion + IP-Adapter가 필요합니다.
