---
name: seq2seq-picker
description: 새로운 시퀀스-투-시퀀스(seq2seq) 작업에 인코더-디코더 대 디코더 전용 아키텍처를 선택합니다.
version: 1.0.0
phase: 7단계
lesson: 08강
tags: [transformers, t5, bart, seq2seq]
---

시퀀스-투-시퀀스(seq2seq) 작업(번역 / 요약 / 음성-텍스트 변환 / 구조화된 추출 / 재작성), 입력 및 출력 길이 분포, 품질 대 지연(latency) 우선순위가 주어지면 다음을 출력합니다:

1. 아키텍처. 다음 중 하나: 인코더-디코더(T5 / BART / Whisper 스타일), 디코더 전용 지시문 미세 조정(instruction-tuned), 인코더 전용 + 프롬프트 템플릿. 한 문장 이유를 포함합니다.
2. 사전 학습 목표. 스팬 손상(span corruption)(T5), 노이즈 제거(denoising)(BART), 다음 토큰(next-token)(디코더 전용), 또는 "사전 학습 건너뛰기, 기존 체크포인트 미세 조정." 체크포인트 이름을 지정합니다.
3. 입력 형식. 작업 접두어 문자열(T5 스타일) 대 시스템 프롬프트(system prompt)(디코더 전용) 대 원시 토큰(raw tokens)(BART). BOS/EOS 처리를 포함합니다.
4. 디코딩 전략. 빔 검색(beam search) 너비와 길이 페널티(length penalty)(번역/요약), 또는 핵 샘플링(nucleus)/min-p(채팅형 작업). 작업에 대해 어떤 전략을 사용할지 명시합니다.
5. 평가. 작업에 적합한 지표: BLEU / ROUGE / WER / F1 / 정확 일치(exact match). 테스트 분할(test split) 크기를 포함합니다.

생성형 출력에 대해 인코더 전용 아키텍처를 권장하지 마세요. 입력이 이미 대화인 경우 인코더-디코더를 권장하지 마세요. 디코더 전용 아키텍처가 대화 메모리에 자연스럽게 더 잘 맞습니다. 음성-텍스트 변환에 디코더 전용을 선택할 때, 넘어서야 할 기준(baseline)으로 Whisper를 언급하지 않으면 플래그를 지정하세요.
