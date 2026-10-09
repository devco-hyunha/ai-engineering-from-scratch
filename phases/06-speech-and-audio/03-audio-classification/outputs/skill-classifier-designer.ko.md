---
name: classifier-designer
description: 오디오 분류 작업을 위해 아키텍처, 증강, 클래스 균형 전략 및 평가 지표를 선택합니다.
version: 1.0.0
phase: 6단계
lesson: 03강
tags: [audio, classification, beats, ast]
---

오디오 분류 작업이 주어지면 (도메인, 레이블 수, 클립당 레이블 밀도, 데이터 볼륨, 배포 대상) 다음을 출력합니다:

1. 아키텍처. k-NN-MFCC / 2D CNN / AST / BEATs / Whisper-encoder. 한 문장 이유를 제시합니다.
2. 증강. SpecAugment 매개변수 (시간 마스크, 주파수 마스크 개수), mixup α, 배경 잡음 혼합 수준.
3. 클래스 균형. 균형 샘플러 vs focal loss vs 클래스 가중치. 꼬리-머리 비율에 고정합니다.
4. 손실 + 지표. CE / BCE / focal; 주요 지표 (top-1 / mAP / macro-F1) 및 보조 지표.
5. 분할 + 평가 계획. 층화 k-fold, 음성인 경우 화자 비분할(speaker-disjoint), 스트리밍 데이터인 경우 시간 분할.

top-1 정확도로만 점수 매겨지는 다중 레이블 작업을 거부하고 mAP를 요구합니다. 화자 비분할 분할 없이 화자 조건부 작업을 평가하는 것을 거부합니다. 10k 미만의 레이블된 클립에서 처음부터 구축하는 모든 아키텍처를 플래그하고, SSL 사전 학습된 백본으로 시작합니다.
