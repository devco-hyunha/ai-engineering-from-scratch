---
name: moderation-stack
description: 프로덕션 배포를 위한 모더레이션 스택 구성을 추천합니다.
version: 1.0.0
phase: 18단계
lesson: 29강
tags: [openai-moderation, perspective, llama-guard, layered-moderation, azure-content-safety]
---

프로덕션 배포가 주어졌을 때, 세 계층에 걸친 모더레이션 스택 구성을 추천해 보세요.

다음 내용을 생성합니다:

1. 입력 분류기. OpenAI Moderation, Llama Guard 3/4, 또는 Perspective API 중 하나를 선택합니다. 정책 분류 체계(taxonomy)에 맞춰 선택하세요. 멀티모달 배포의 경우 Llama Guard 4 또는 OpenAI omni-moderation을 사용합니다.
2. 출력 분류기. 입력 분류기와 동일하거나 다른 것을 선택합니다. 임계값을 다운스트림 위험 모델에 맞춰 설정하세요.
3. 커스텀 도메인 규칙. 일반적인 분류기가 잡지 못하는 도메인 특화 규칙을 나열합니다: 금융 조언 면책 조항, 의료 조언 거부, 법률 면책 패턴 등.
4. 경계 사례를 위한 판정자. 인간 에스컬레이션 경로를 지정합니다. 하드 거부는 최종 결정이며, 모호한 사례는 SLA 내 인간 검토로 전달됩니다.
5. 마이그레이션 계획. 스택에 Azure Content Moderator가 포함되어 있다면, 2027년 2월 서비스 종료 전까지 Azure AI Content Safety로 마이그레이션하는 계획을 세우세요.

하드 거부 조건:
- 출력 모더레이션이 없는 배포 (입력만으로는 충분하지 않습니다).
- 규제 대상 표면(금융, 건강, 법률)에 커스텀 도메인 규칙이 없는 배포.
- 현대적인 채팅 애플리케이션에 대해 solely LLM 이전 시대 분류기(Perspective)에만 의존하는 배포.

거부 규칙:
- 사용자가 단일 최적 분류기를 요청하면 거부하세요 — 분류기 선택은 정책 분류 체계(taxonomy)에 따라 다릅니다.
- 사용자가 임계값을 요청하면 단일 숫자를 제시하지 마세요 — 임계값은 위험 허용 범위와 다운스트림 효과에 따라 달라집니다.

출력: 한 페이지 분량의 추천안을 작성하여 다섯 섹션을 채우고, 각 계층의 분류기를 명시하며, 마이그레이션 의무 사항을 표시합니다. OpenAI Moderation 문서와 Llama Guard 3/4 참조를 각각 한 번씩 인용하세요.
