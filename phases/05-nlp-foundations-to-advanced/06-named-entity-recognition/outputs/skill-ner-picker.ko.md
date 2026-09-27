---
name: ner-picker
description: 주어진 추출 작업에 맞는 NER 접근을 고릅니다.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

작업 설명(도메인, 라벨 집합, 언어, 지연, 데이터 양)이 주어지면 다음을 출력하세요:

1. Approach. 규칙 기반 + 가제티어, CRF, BiLSTM-CRF, 또는 트랜스포머 파인튜닝.
2. Starting model. 이름을 적으세요(spaCy 모델 ID 예: `en_core_web_sm` / `en_core_web_trf`, Hugging Face 체크포인트 ID 예: `dslim/bert-base-NER`, 또는 "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, 또는 스팬 기반. 한 문장으로 선택 이유(근거)를 제시하세요.
4. Evaluation. `seqeval`을 쓰세요. 항상 개체 수준 F1을 보고하고, 토큰 수준은 절대 쓰지 마세요.

라벨 예제가 500개 미만이면, 사용자가 이미 사전학습 도메인 모델(예: 의료용 BioBERT)을 확보하고 있지 않은 한 트랜스포머 파인튜닝을 권장하지 마세요. 중첩 개체는 스팬 기반 또는 멀티패스 모델이 필요하다고 명시하세요. 사용자가 "production scale"을 언급하면서 CoNLL-2003 기본 라벨을 그대로 쓰는 경우에는 가제티어 감사를 필히 요구하세요.
