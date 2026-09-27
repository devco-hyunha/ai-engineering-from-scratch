---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

다운스트림 과제(정보 추출, 재작성 검증, 질의 분해, 표제어 추출)가 주어지면, 다음을 출력합니다.

1. 태그셋. 영어 전용 레거시 파이프라인은 Penn Treebank, 다국어·교차언어는 Universal Dependencies.
2. 라이브러리. 대부분의 프로덕션은 spaCy(`en_core_web_sm` / `_lg` / `_trf`), 학술급 다국어는 stanza, 최고 UD 정확도는 trankit.
3. 통합 스니펫. 라이브러리를 호출하고 `.pos_`, `.dep_`, `.head`를 소비하는 3–5줄.
4. 테스트할 실패 모드. 명사-동사 중의성(`saw`, `book`, `can`)과 PP-부착 중의성이 고전적 함정입니다. 출력 20개를 샘플링해 눈으로 확인하세요.

직접 파서를 만드는 권고는 거부하세요. 파서를 처음부터 만드는 것은 연구 프로젝트이지 응용 과제가 아닙니다. 소문자/대문자 변형을 처리하지 않고 POS 태그를 소비하는 파이프라인은 취약하다고 표시하세요.
