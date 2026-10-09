---
name: grammar-pipeline
description: downstream NLP 작업을 위한 고전적인 POS + 의존성 파이프라인을 설계합니다.
version: 1.0.0
phase: 5단계
lesson: 07강
tags: [nlp, pos, parsing]
---

downstream 작업 (정보 추출, 재작성 검증, 쿼리 분해, 어근화)이 주어지면 다음을 출력합니다:

1. 태그셋. 영어 전용 레거시 파이프라인에는 Penn Treebank를, 다국어 또는 언어 간 파이프라인에는 Universal Dependencies를 사용합니다.
2. 라이브러리. 대부분의 프로덕션 환경에서는 spaCy (`en_core_web_sm` / `_lg` / `_trf`)를, 학술적 수준의 다국어 처리에는 stanza를, 가장 높은 UD 정확도를 원한다면 trankit를 선택합니다.
3. 통합 스니펫. 라이브러리를 호출하고 `.pos_`, `.dep_`, `.head`를 소비하는 3-5줄의 코드입니다.
4. 테스트해야 할 실패 모드. 명사-동사 모호성 (`saw`, `book`, `can`)과 전치사구 부착 모호성은 고전적인 함정입니다. 출력 20개를 샘플링하여 육안으로 확인해 보세요.

직접 파서를 만들 것을 권하지 마세요. 파서를 처음부터 구축하는 것은 연구 프로젝트이며, 애플리케이션 작업이 아닙니다. 소문자/대문자 변형을 처리하지 않고 POS 태그를 소비하는 모든 파이프라인은 취약하다고 표시하세요.
