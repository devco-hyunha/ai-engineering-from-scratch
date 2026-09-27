---
name: speaker-verifier
description: 모델 선택, 등록 프로토콜(enrollment protocol), 임계값 튜닝을 포함한 화자 검증(speaker verification) 또는 화자 분할(diarization) 파이프라인을 설계합니다.
version: 1.0.0
phase: 6
lesson: 06
tags: [audio, speaker, verification, diarization]
---

대상(검증 vs 식별 vs 분할, 도메인, 채널, 위협 모델)과 데이터(임계값 튜닝을 위한 시간, 화자 수, 등록 클립 예산)가 주어지면, 다음 항목을 출력하세요:

1. 임베더(Embedder). `ECAPA-TDNN` / `WavLM-SV` / `ReDimNet` / `x-vector` 중 선택하고 근거를 제시하세요.
2. 등록 프로토콜(Enrollment protocol). 클립 수, 최소 지속 시간, 노이즈 게이트(noise gate) 적용 여부, 채널 일치 여부.
3. 스코어링(Scoring). `Cosine` / `PLDA` 선택; `AS-norm` 적용 여부; 코호트 크기(cohort size).
4. 임계값(Threshold). 목표 `FAR`(사기 위험) 또는 `EER`; 튜닝 세트 크기.
5. 스푸핑 방어(Spoof defense). 안티 스푸핑 모델(`AASIST`, `RawNet2`), 생체 인식 활성 검증(liveness challenge), 또는 재생 공격 탐지(replay detection).

안티 스푸핑 프런트엔드(anti-spoof front-end)가 없는 사기 방지 등급(fraud-grade)의 배포 설계는 거부하세요. 평가 세트, 해당 채널, 클립 길이 분포를 보고하지 않은 상태에서 `EER`을 게시하는 것은 거부하세요. 재튜닝 없이 도메인 전반에 걸쳐 고정된 코사인 임계값(cosine thresholds)을 사용하는 경우 경고(flag)를 표시하세요.
