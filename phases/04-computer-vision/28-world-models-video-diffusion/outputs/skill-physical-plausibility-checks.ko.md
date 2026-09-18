---
name: skill-physical-plausibility-checks
description: 배포 전 생성 비디오에 물체 영속성·중력·연속성 자동 검사를 정의합니다
version: 1.0.0
phase: 4
lesson: 28
tags: [video-generation, quality, physics, evaluation]
---

# 물리적 개연성 검사 (Physical Plausibility Checks)

생성 비디오의 프로덕션 배포에는 자동 가드레일이 필요합니다. 사람 검토는 스케일되지 않고; 물리 검사가 고전적 실패 모드를 잡습니다.

## 언제 쓰나요 (When to use)

- 텍스트 또는 이미지 프롬프트에서 비디오를 생성하는 모든 제품.
- 비디오 생성 API 엔드포인트에서 QA 자동화.
- 파인튜닝 또는 기본 모델 업데이트 후 비디오 모델의 품질 드리프트 모니터링.

## 입력 (Inputs)

- `video`: `(T, H, W, 3)` 텐서 또는 mp4 경로.
- 선택적 참고 정보: 예상 객체 수, 초기 장면 설명.

## 검사 (Checks)

### 1. 물체 영속성 (Object permanence)
SAM 3.1 Object Multiplex로 프레임에 걸쳐 모든 검출을 추적합니다. 안정적 트랙이 <=3 프레임 동안 사라졌다가 다시 나타나면 플래그 — 모델이 객체를 일시적으로 잃음. 프레임 가장자리가 아닌 중심 근처에서 객체가 사라지면 hard fail; 가장자리에서는 soft fail.

### 2. 모션 부드러움 (Motion smoothness)
연속 프레임 사이 optical flow는 대부분 연속이어야 합니다. 갑작스러운 픽셀당 flow 스파이크는 텔레포트를 나타냅니다. RAFT로 flow를 계산하고; 99번째 백분위 flow 크기가 중앙값의 10배를 넘는 프레임을 플래그합니다.

### 3. 중력 / 지지 (Gravity / support)
고체(음식, 공, 도구)로 검출된 객체에 대해, 들어 올리는 액션이 없으면 수직 위치가 비증가인지 확인합니다. 객체 근처 "잡는 손"이 검출되지 않으면 위쪽 드리프트를 플래그합니다.

### 4. 정체성 일관성 (Identity consistency)
사람이나 캐릭터에 대해 프레임에 걸쳐 face-recognition 임베딩을 씁니다. 지속 정체성에 대해 5프레임 윈도우에서 코사인 유사도가 > 0.8로 유지되어야 합니다. 임계값 아래는 캐릭터가 변형되었음을 뜻합니다.

### 5. 손과 사지 (Hands and limbs)
포즈 추정기(Lesson 21)를 돌립니다. 손이 보이는 손가락 > 5 또는 < 4인 프레임; 프레임 사이 팔 길이가 두 배가 되는 곳; 사지가 표면을 통과해 몸과 교차하는 곳을 플래그합니다.

### 6. 텍스트 렌더링 (프롬프트가 텍스트를 요청한 경우)
사용자 프롬프트에 따옴표 문자열이 있으면, 생성 프레임을 OCR하고 요청 문자열에 대한 CER을 계산합니다. CER > 20%이면 플래그합니다.

## 보고 (Report)

```
[plausibility]
  video frames:           <T>
  permanence violations:  <N>
  smoothness violations:  <N>
  gravity violations:     <N>
  identity drift:         <N of 5-frame windows>
  limb anomalies:         <N>
  OCR CER vs requested:   <float>

[verdict]
  ship | hold | reject

[samples for review]
  frame ranges where each failure occurred
```

## 규칙 (Rules)

- 단일 검사만으로 hard-block하지 마세요; 점수를 집계하고 총 이상치가 임계값을 넘으면 비디오를 검토용으로 보류하세요.
- 정체성 드리프트와 영속성 위반에 가장 높은 가중치를 주세요 — 사용자가 먼저 알아챕니다.
- 검사별 실패율을 시간에 걸쳐 로깅하세요; 상승 추세는 보통 기본 모델이 업데이트되었거나 프롬프트 분포가 이동했음을 뜻합니다.
- 플래그된 비디오를 절대 삭제하지 마세요; 모델 디버깅과 사후 분석을 위해 보관하세요.
- 민감 콘텐츠(사람, 아동, 공인)에서는 점수와 무관하게 모든 비디오의 사람 검토를 요구하세요.
