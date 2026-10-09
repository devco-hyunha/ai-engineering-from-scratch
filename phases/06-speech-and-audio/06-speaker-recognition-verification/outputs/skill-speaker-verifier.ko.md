---
name: speaker-verifier
description: 모델 선택, 등록 프로토콜, 임계값 튜닝을 포함하여 화자 검증 또는 화자 분리 파이프라인을 설계합니다.
version: 1.0.0
phase: 6단계
lesson: 06강
tags: [audio, speaker, verification, diarization]
---

목표(검증 vs 식별 vs 화자 분리, 도메인, 채널, 위협 모델)와 데이터(임계값 튜닝을 위한 시간, 화자 수, 등록 클립 예산)가 주어지면 다음을 출력합니다:

1. 임베더. ECAPA-TDNN / WavLM-SV / ReDimNet / x-vector. 이유를 설명합니다.
2. 등록 프로토콜. 클립 수, 최소 지속 시간, 잡음 게이트, 채널 일치.
3. 채점. 코사인 유사도(Cosine Similarity) / PLDA; AS-norm 사용 여부; 코호트 크기.
4. 임계값. 목표 FAR(사기 위험) 또는 EER; 튜닝 세트 크기.
5. 스푸핑 방어. 안티 스푸핑 모델(AASIST, RawNet2), 라이브니스 챌린지, 또는 재생 감지.

안티 스푸핑 프런트엔드 없이 사기 등급 배포를 거부합니다. 평가 세트, 그 채널, 클립 길이 분포를 보고하지 않고 EER을 공개하는 것을 거부합니다. 도메인 간 재튜닝 없이 고정된 코사인 임계값을 플래그합니다.
