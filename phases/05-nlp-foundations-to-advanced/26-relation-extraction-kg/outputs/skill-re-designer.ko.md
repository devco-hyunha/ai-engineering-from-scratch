---
name: re-designer
description: 출처(provenance) 및 정규화(canonicalization)를 포함한 관계 추출(relation extraction) 파이프라인을 설계하십시오.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

코퍼스(도메인, 언어, 규모)와 하위 작업 용도(KG-RAG, 분석, 컴플라이언스)가 주어졌을 때, 다음을 출력하십시오:

1. 추출기(Extractor): 패턴 기반(Pattern-based), 지도 학습(Supervised), LLM, 또는 AEVS 하이브리드 방식 중 선택하십시오. 정밀도(precision)와 재현율(recall) 목표에 근거하여 선택 이유를 제시하십시오.
2. 온톨로지(Ontology): 폐쇄형 속성 목록(Wikidata 또는 도메인 기반) 또는 정규화(canonicalization) 단계를 포함한 개방형 IE(Open IE) 중 선택하십시오.
3. 출처(Provenance): 모든 트리플(triple)은 소스 문자 범위(char-span)와 문서 ID(`doc id`)를 포함해야 합니다. 이는 감사를 위해 타협할 수 없는 필수 사항입니다.
4. 병합 전략(Merge strategy): 정규화된 엔티티 ID(`canonical entity id`) + 관계 ID(`relation id`) + 시간적 한정자(temporal qualifiers)를 활용한 중복 제거(dedup) 정책을 수립하십시오.
5. 평가(Evaluation): 수동으로 라벨링된 200개의 트리플에 대한 정밀도/재현율 및 LLM 추출 샘플에 대한 환각률(hallucination-rate)을 정의하십시오.

스팬 검증(span verification, 소스 출처)이 없는 LLM 기반 RE 파이프라인은 거부하십시오. 정규화 과정 없이 프로덕션 그래프로 유입되는 개방형 IE(Open IE) 출력은 거부하십시오. 시간 제약이 있는 관계(고용주, 배우자, 직책 등)에 시간적 한정자(temporal qualifier)가 없는 파이프라인은 경고(flag)를 표시하십시오.
