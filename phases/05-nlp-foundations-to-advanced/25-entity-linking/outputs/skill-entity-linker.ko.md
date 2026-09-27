---
name: entity-linker
description: 엔티티 연결(entity linking) 파이프라인 설계 — KB, 후보 생성기, 모호성 해소기, 평가.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

사용 사례(도메인 KB, 언어, 볼륨, 지연 시간 예산)가 주어지면, 다음 항목을 출력하세요:

1. 지식 베이스(Knowledge Base). Wikidata / Wikipedia / 커스텀 KB. 버전 날짜. 갱신 주기.
2. 후보 생성기(Candidate Generator). 별칭 인덱스(Alias-index), 임베딩(embedding), 또는 하이브리드 방식. K개에 대한 타겟 언급 재현율(Target mention recall @ K).
3. 모호성 해소기(Disambiguator). 사전 확률 + 문맥(Prior + context), 임베딩 기반, 생성형, 또는 LLM 프롬프트 방식.
4. NIL 전략. 최고 점수 임계값(Threshold), 분류기, 또는 명시적 NIL 후보.
5. 평가(Evaluation). 30개에 대한 언급 재현율(Mention recall @ 30), Top-1 정확도, 홀드아웃 세트(held-out set)에서의 NIL 탐지 F1 점수.

언급 재현율(mention-recall) 베이스라인이 없는 모든 EL 파이프라인은 거부하세요(후보 생성기가 올바른 엔티티를 찾아냈는지 알 수 없다면 모호성 해소기를 평가할 수 없습니다). 유효한 KB ID로 출력을 제약(constrained output)하지 않는 LLM 프롬프트 기반 EL 파이프라인은 거부하세요. 도메인 미세 조정(fine-tuning) 없이 인기 편향(popularity bias)이 소수 엔티티(예: 이름 충돌)에 영향을 미치는 시스템은 주의(Flag)를 표시하세요.
