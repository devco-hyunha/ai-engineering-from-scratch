---
name: ner-picker
description: 주어진 추출 작업에 적합한 NER 접근법을 선택합니다.
version: 1.0.0
phase: 5단계
lesson: 06강
tags: [nlp, ner, extraction]
---

작업 설명(도메인, 레이블 세트, 언어, 지연 시간, 데이터 볼륨)이 주어지면 다음을 출력합니다:

1. 접근법. 규칙 기반 + 사전(gazetteer), CRF, BiLSTM-CRF, 또는 트랜스포머 미세 조정(Fine-tuning).
2. 시작 모델. 모델 이름을 지정합니다(spaCy 모델 ID는 `en_core_web_sm` / `en_core_web_trf`, Hugging Face 체크포인트 ID는 `dslim/bert-base-NER`, 또는 "custom, trained from scratch").
3. 레이블링 전략. BIO, BILOU, 또는 스팬(span) 기반. 한 문장으로 근거를 설명합니다.
4. 평가. `seqeval`를 사용합니다. 항상 엔티티 단위 F1을 보고하고, 토큰 단위 F1는 보고하지 않습니다.

사용자가 이미 사전 학습된 도메인 모델(예: 의료용 BioBERT)을 가지고 있지 않은 한, 500개 미만의 레이블이 지정된 예시로는 트랜스포머 미세 조정을 권장하지 않습니다. 중첩된 엔티티는 스팬(span) 기반 또는 다중 패스 모델이 필요하다고 표시합니다. 사용자가 "production scale"을 언급하면서 CoNLL-2003 기본 레이블을 사용하는 경우, 사전(gazetteer) 감사를 요구합니다.
