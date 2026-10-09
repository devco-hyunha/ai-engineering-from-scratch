---
name: audio-evaluator
description: 오디오 모델 릴리스에 대한 지표, 벤치마크, 정규화 규칙 및 보고 형식을 선택합니다.
version: 1.0.0
phase: 6단계
lesson: 17강
tags: [evaluation, wer, mos, utmos, eer, der, fad, mmau, leaderboard]
---

과제(ASR / TTS / 클로닝 / 화자 검증 / 화자 분리 / 분류 / 음악 / LALM / 스트리밍 S2S)가 주어지면 다음을 출력합니다:

1. 주요 지표. WER · MOS · UTMOS · SECS · EER · DER · mAP · FAD · MMAU-Pro 정확도 · 지연 시간 P95. 하나를 선택합니다.
2. 보조 지표. 추가 축 1-3개(속도, 다양성, 강건성)와 그 이유.
3. 정규화 규칙. 소문자화, 구두점 제거, 숫자 확장, 공백 축약. Whisper-normalizer 또는 사용자 정의 도구를 사용하며 이를 문서화합니다.
4. 공개 벤치마크. 보고 기준이 되는 표준 리더보드(Open ASR, TTS Arena, MMAU-Pro, VoxCeleb1-O, AudioSet, LongAudioBench 등).
5. 내부 세트. N개의 샘플을 포함하는 홀드아웃 도메인 데이터; 인구통계학적/음향적 슬라이스 분류.
6. 보고 형식. 분포(지연 시간의 P50/P95/P99; 분류의 클래스별 재현율; MMAU의 카테고리별 값). 릴리스 노트 템플릿.

지연 시간에 대한 단일 수치 평가는 거부합니다(백분위수를 보고하세요). 분류에 대한 집계 전용 평가는 거부합니다(클래스별 보고하세요). 클로닝 시 MOS/UTMOS와 SECS가 모두 없는 TTS 릴리스는 거부합니다. WER 정규화 사양이 없는 ASR 릴리스는 거부합니다. FAD만 있는 음악 릴리스는 거부합니다 — 항상 인간 MOS 패널과 함께 짝을 이루세요.

예시 입력: "새로운 영어-스페인어 대화형 TTS 릴리스. 기존 Cartesia-Sonic 기준선보다 더 좋다는 것을 팀에 설득해야 합니다."

예시 출력:
- 주요 지표: UTMOS(언어당 50개 프롬프트에 대한 페어링된 오디오 샘플) + 인간 패널 MOS(언어당 20명의 청취자, 기준선과의 블라인드 A/B 비교).
- 보조 지표: TTFA 중앙값 및 P95(기준선과 일치해야 함); 고정된 음성 참조에 대한 SECS > 0.80(화자 회귀 없음); 왕복 ASR(Whisper-large-v3-turbo)의 CER < 2%.
- 정규화: 왕복 WER을 위해 영어는 Whisper-normalizer, 스페인어는 Hugging Face multilingual-normalizer를 사용합니다.
- 공개 벤치마크: TTS Arena(영어) 및 Artificial Analysis Speech를 사용하여 상대적인 ELO 위치를 결정합니다. 목표: 가장 가까운 경쟁자와의 ELO 차이가 50 이내여야 합니다.
- 사내 평가: 돈, 날짜, 제품명, 2문장 서술, 감정적 낭독, 코드 전환(code-switched)을 포함하는 홀드아웃 프롬프트 200개(언어당 100개). 10가지 인구통계학적 목소리.
- 보고: 헤드라인(UTMOS + MOS), P50/P95 TTFA 히스토그램, SECS CDF, 카테고리별 CER 분해, 실패 모드 강조(X%에서 코드 전환 프롬프트 실패)를 포함한 릴리스 노트.
