---
name: vit-configurator
description: 새로운 비전 태스크를 위한 ViT 변형(variant), 패치 크기(patch size), 사전 학습 소스(pretraining source)를 선택합니다.
version: 1.0.0
phase: 7
lesson: 9
tags: [transformers, vit, vision]
---

비전 태스크(분류/세그멘테이션/탐지/검색), 이미지 해상도, 데이터셋 크기(라벨링 데이터 + 미라벨링 데이터), 그리고 배포 대상을 입력받으면 다음을 출력하세요:

1. **Backbone (백본).** 다음 중 하나를 선택하세요: DINOv2 ViT-L/14 (검색/분류 기본값), SAM 3 인코더 (세그멘테이션), SigLIP (비전-언어), ConvNeXt (지연 시간 민감형). 선택 이유를 한 문장으로 설명하세요.
2. **Patch size (패치 크기).** 224 해상도의 표준 분류 작업인 경우 16, DINOv2를 사용하는 경우 14, 고해상도 밀집 예측(dense prediction) 작업인 경우 8을 선택합니다. 시퀀스 길이 `(H/P)^2 + 1`과 어텐션 비용 `O(N^2)`를 명시하세요.
3. **Pretraining source (사전 학습 소스).** 체크포인트 이름을 명시하세요. 소규모 라벨링 데이터셋(<10k)의 경우: DINOv2 특징(features) 동결 + 선형 프로브(linear probe). 100k 이상의 경우: 마지막 블록 미세 조정(fine-tune). 선택 이유를 기술하세요.
4. **Training recipe (학습 레시피).** 옵티마이저(AdamW), 학습률(lr), 데이터 증강(RandAug, MixUp, Random Erasing), 라벨 스무딩(label smoothing, 통상 0.1), EMA를 포함하세요.
5. **Risk note (리스크 노트).** 데이터 체제 리스크(전체 미세 조정을 하기에는 데이터가 너무 적음), 해상도 불일치(위치 보간(position interpolation) 없이 사전 학습 224에서 배포 1024로 변경), 레지스터 토큰(register-token) 부재(DINOv2 특징 추출에 악영향을 줄 수 있음)를 언급하세요.

100만 장 미만의 이미지로 ViT를 처음부터(from scratch) 학습시키는 것은 권장하지 마세요. 이 경우 CNN 베이스라인이 더 우세할 것입니다. Flash Attention 및 계층적 변형 모델(Swin)에 대한 명시적인 논의 없이 시퀀스 길이가 4096을 초과하는 패치 크기를 권장하는 것은 거부하세요. 위치 임베딩(positional embeddings)을 보간하지 않고 입력 해상도를 변경하는 모든 배포 방식에 대해서는 반드시 경고를 표시하세요.
