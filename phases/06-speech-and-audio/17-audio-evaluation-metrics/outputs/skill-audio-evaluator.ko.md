---
name: audio-evaluator
description: Pick metrics, benchmarks, normalization rules, and reporting format for any audio model release.
version: 1.0.0
phase: 6
lesson: 17
tags: [evaluation, wer, mos, utmos, eer, der, fad, mmau, leaderboard]
---

주어진 작업(ASR / TTS / cloning / speaker-verif / diarization / classification / music / LALM / streaming S2S)에 대해 다음을 출력하세요:

1. **주요 지표(Primary metric).** WER · MOS · UTMOS · SECS · EER · DER · mAP · FAD · MMAU-Pro accuracy · latency P95 중 하나를 선택합니다.
2. **보조 지표(Secondary metrics).** 1~3개의 추가 축(속도, 다양성, 강건성)과 그 이유를 제시합니다.
3. **정규화 규칙(Normalization rule).** 소문자 변환, 문장 부호 제거, 숫자 확장, 공백 압축 등을 포함합니다. `Whisper-normalizer` 또는 커스텀 방식을 사용하며 이를 문서화합니다.
4. **공개 벤치마크(Public benchmark).** 보고의 기준이 될 표준 리더보드(Open ASR, TTS Arena, MMAU-Pro, VoxCeleb1-O, AudioSet, LongAudioBench 등)를 지정합니다.
5. **사내 데이터셋(In-house set).** N개의 샘플로 구성된 홀드아웃(held-out) 도메인 데이터; 인구통계학적/음향적 슬라이스 분류를 포함합니다.
6. **보고 형식(Reporting format).** 분포(지연 시간의 경우 P50/P95/P99; 분류의 경우 클래스별 재현율; MMAU의 경우 카테고리별 분류) 및 릴리스 노트 템플릿을 포함합니다.

지연 시간(latency)에 대해 단일 수치 평가를 요청하는 경우 거부하십시오(백분위수 보고 필수). 분류(classification) 작업에서 집계 수치만 요청하는 경우 거부하십시오(클래스별 보고 필수). TTS 릴리스 시 MOS/UTMOS와 SECS(클로닝의 경우)가 모두 없는 경우 거부하십시오. ASR 릴리스 시 WER 정규화 사양이 없는 경우 거부하십시오. 음악 릴리스 시 FAD만 있는 경우 거부하십시오(항상 인간 MOS 패널과 병행해야 함).

**입력 예시:** "새로운 영어-스페인어 대화형 TTS 출시. 기존 Cartesia-Sonic 베이스라인보다 성능이 뛰어남을 팀에 설득해야 함."

**출력 예시:**
- **Primary:** UTMOS (언어당 50개 프롬프트에 대한 쌍을 이룬 오디오 샘플) + 인간 패널 MOS (언어당 20명의 청취자, 베이스라인과 블라인드 A/B 테스트).
- **Secondary:** TTFA 중앙값 및 P95 (베이스라인과 일치해야 함); 고정된 음성 참조 대비 SECS > 0.80 (화자 퇴보 방지); 왕복 ASR(Whisper-large-v3-turbo)에서의 CER < 2%.
- **Normalization:** 왕복 WER 측정을 위해 영어는 `Whisper-normalizer`, 스페인어는 Hugging Face `multilingual-normalizer` 사용.
- **Public benchmark:** 상대적 ELO 위치 파악을 위해 TTS Arena(영어) 및 Artificial Analysis Speech 사용. 목표: 가장 가까운 경쟁자와 ELO 차이 50 이내.
- **In-house:** 돈, 날짜, 제품명, 2문장 내레이션, 감정 낭독, 코드 스위칭(code-switched)을 포함하는 200개의 홀드아웃 프롬프트(언어당 100개). 10개의 인구통계학적 음성.
- **Reporting:** 헤드라인(UTMOS + MOS)이 포함된 릴리스 노트, P50/P95 TTFA 히스토그램, SECS CDF, 카테고리별 CER 상세 내역, 실패 모드(failure-mode) 명시(예: 코드 스위칭 프롬프트에서 X% 실패).
