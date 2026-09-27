---
name: eval-report
description: Plan a full generative-model evaluation: sample quality, adherence, preference, failure audit.
version: 1.0.0
phase: 8
lesson: 14
tags: [evaluation, fid, clip, elo]
---

새로운 생성 모델 체크포인트(`generative-model checkpoint`), 참조 베이스라인(`reference baseline`), 그리고 모달리티(`image` / `video` / `audio` / `3D`)가 주어지면, 다음과 같은 전체 평가 계획(`eval plan`)을 출력하세요:

1. **샘플 품질(Sample quality)**: 홀드아웃 실제 데이터셋(`held-out real set`)과 비교하여 10k~30k개의 샘플에 대해 FID / FD-DINO / CMMD를 측정합니다. 해상도는 일치시켜야 합니다. 3개 시드(`seed`)의 평균 및 표준편차(`std`)를 보고합니다.
2. **준수성(Adherence)**: 프롬프트-이미지 쌍에 대해 CLIP score / CMMD를 측정합니다. 텍스트-이미지(`text-to-image`)의 경우 HPSv2 + ImageReward + PickScore를 포함합니다. 비디오의 경우 시각-언어 지표(`V-Eval`)를 추가합니다. 오디오의 경우 CLAP + MOS를 사용합니다.
3. **쌍체 선호도(Pairwise preference)**: 베이스라인과 비교하여 200~2000개의 프롬프트에 대해 블라인드 A/B 테스트를 수행합니다. 인간(`Human`) + LLM-judge + `PartiPrompts` 커버리지를 포함합니다.
4. **카테고리별 분석(Category breakdown)**: 프롬프트 카테고리별(사람, 동물, 텍스트 렌더링, 구도, 스타일) 성능을 측정합니다. 전체 지표가 개선되더라도 특정 카테고리에서 성능 저하(`regression`)가 발생하면 이를 명시합니다.
5. **안전성 / 오용(Safety / misuse)**: 상위 K개 생성물에 대해 NSFW 분류기, 딥페이크 탐지기, 워터마크 확인, 저작권 유사성 스캔을 수행합니다.
6. **승인(Sign-off)**: 명시적인 게이트(`gate`)를 설정합니다. 조건은 다음과 같습니다: FID가 베이스라인의 +5% 이내이거나, OR 인간 승률(`human win rate`)이 55%를 초과하거나, OR 문서화된 정성적 이점이 있어야 합니다. 단일 지표만으로 성능을 주장하지 마십시오.

샘플 수 `N < 5000`인 경우 FID 보고를 거부하세요. 모델이 학습 과정에서 보았을 가능성이 있는 프롬프트로 계산된 벤치마크 보고를 거부하세요. 인간의 교차 검증(`human cross-check`) 없이 LLM-judge 결과만 보고하는 것을 거부하세요. 절대적인 기준값(`absolute base value`)과 단일 시드(`single seed`)를 제시하지 않고 특정 지표가 "20% 상승했다"라고 주장하는 모든 경우를 플래그(`flag`)하세요.
