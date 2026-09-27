---
name: bert-finetuner
description: 새로운 분류(classification), 추출(extraction) 또는 검색(retrieval) 작업을 위한 BERT 미세 조정(fine-tuning) 범위를 설정합니다.
version: 1.0.0
phase: 7
lesson: 6
tags: [bert, fine-tuning, nlp]
---

다운스트림 작업(분류 / NER / 검색 / 재순위화 / NLI), 레이블링된 데이터 크기, 배포 제약 조건(지연 시간, 디바이스)이 주어지면 다음을 출력합니다:

1. 백본(Backbone) 선택. 모델 이름(ModernBERT-base / large, DeBERTa-v3, multilingual-e5 등)과 한 문장으로 된 선정 이유를 포함합니다. 8K 이하의 컨텍스트가 필요한 영어 작업에는 ModernBERT를 우선적으로 권장합니다.
2. 헤드 사양(Head spec). 분류(Classification): `[CLS]` → dropout → linear(num_classes). NER: 토큰별 linear + 선택적 CRF. 검색(Retrieval): mean-pool + contrastive loss.
3. 학습 레시피(Training recipe). 옵티마이저(AdamW, 일반적인 lr 2e-5), warmup % (6–10%), 에포크(epochs, 3–5), 배치 크기(batch size), fp16/bf16.
4. 평가 계획(Eval plan). 작업에 적합한 지표(분류의 경우 accuracy + F1, NER의 경우 entity-level F1, 검색의 경우 MRR/NDCG). 홀드아웃(Held-out) 분할 크기.
5. 실패 모드 점검(Failure mode check). 명명된 하나의 리스크: 레이블 누수(label leakage), 클래스 불균형(class imbalance), 컨텍스트 절단(context truncation), 사전 학습(pretrain)과 미세 조정(fine-tune) 코퍼스 간의 토크나이저 불일치(tokenizer mismatch).

생성형 출력(텍스트 생성)을 위해 BERT를 미세 조정하는 것은 거부하고, 대신 디코더 전용(decoder-only) 모델을 권장하십시오. 소수 클래스가 10% 미만인 경우, 클래스 계층화 평가(class-stratified eval) 없이 미세 조정 모델을 배포하는 것을 거부하십시오. 레이블링된 예시가 1,000개 미만인 상태에서 전체 백본을 언프리즈(unfreeze)하는 모든 미세 조정은 과적합(overfit) 가능성이 높다고 명시하십시오.
