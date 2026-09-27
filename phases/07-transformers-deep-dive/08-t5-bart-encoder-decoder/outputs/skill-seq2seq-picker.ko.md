---
name: seq2seq-picker
description: 새로운 seq2seq 작업을 위해 encoder-decoder와 decoder-only 중 하나를 선택합니다.
version: 1.0.0
phase: 7
lesson: 8
tags: [transformers, t5, bart, seq2seq]
---

seq2seq 작업(번역, 요약, 음성-텍스트 변환, 구조화된 추출, 재작성), 입력 및 출력 길이 분포, 품질 대 지연 시간(latency) 우선순위가 주어지면 다음을 출력하세요:

1. 아키텍처(Architecture): 다음 중 하나를 선택합니다: encoder-decoder (T5 / BART / Whisper 스타일), decoder-only 지시어 미세 조정(instruction-tuned), 또는 encoder-only + 프롬프트 템플릿. 선택 이유를 한 문장으로 설명하세요.
2. 사전 학습 목표(Pretraining objective): Span corruption (T5), denoising (BART), next-token (decoder-only), 또는 "사전 학습 생략, 기존 체크포인트 미세 조정(skip pretraining, fine-tune existing checkpoint)" 중 하나를 선택하고 체크포인트 이름을 명시하세요.
3. 입력 포맷팅(Input formatting): 작업 접두사 문자열(T5 스타일), 시스템 프롬프트(decoder-only), 또는 원시 토큰(BART) 중 선택하세요. BOS/EOS 처리 방식도 포함해야 합니다.
4. 디코딩 전략(Decoding strategy): 빔 서치 너비 및 길이 페널티(번역/요약 작업), 또는 nucleus/min-p(채팅형 작업) 중 해당 작업에 가장 적합한 전략을 명시하세요.
5. 평가(Eval): 작업에 적합한 지표(BLEU, ROUGE, WER, F1, exact match 등)와 테스트 데이터 분할 크기를 포함하세요.

생성형 출력(generative outputs)에 대해 encoder-only를 추천하는 것은 거부하세요. 입력이 이미 대화(conversation) 형태인 경우 encoder-decoder를 추천하는 것도 거부하세요. 대화 메모리는 decoder-only가 자연스럽게 처리합니다. 음성-텍스트 변환(speech-to-text) 작업에서 Whisper를 비교 기준(baseline)으로 언급하지 않고 decoder-only를 선택하는 경우 이를 지적하세요.
