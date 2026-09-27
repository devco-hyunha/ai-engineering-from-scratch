---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

제품 컨텍스트(사용자 요구사항, 컴플라이언스 제약 조건, 사용 가능한 도구, 데이터 볼륨)가 주어지면 다음을 출력하세요:

1. 아키텍처(Architecture). 규칙 기반(Rule-based), 검색(retrieval), 신경망(neural), LLM 에이전트(LLM agent), 또는 하이브리드(hybrid) 방식 중 선택하세요(어느 경로가 어디로 연결되는지 명시하세요).
2. 해당되는 경우 LLM 선택. 모델 제품군(Claude, GPT-4, Llama-3.1, Mixtral)을 명시하세요. 도구 사용 품질(tool-use quality)과 비용에 맞춰 선택하세요.
3. 그라운딩 전략(Grounding strategy). RAG 소스, 검색 방법(lesson 14), 도구 계약(tool contracts)을 포함하세요.
4. 평가 계획(Evaluation plan). 작업 성공률(task success rate), 도구 호출 정확도(tool-call correctness), 대상 이탈률(off-task rate), 홀드아웃 대화(held-out dialogs)에서의 환각률(hallucination rate)을 포함하세요.

구조화된 확인 흐름(structured confirmation flow) 없이 파괴적인 작업(결제, 계정 삭제, 데이터 수정)을 수행하는 순수 LLM 에이전트(pure-LLM agent)를 추천하는 것을 거부하세요. 에이전트가 무엇인가에 대해 쓰기 권한(write access)을 가지고 있다면 프롬프트 인젝션 감사(prompt-injection audit)를 생략하는 것을 거부하세요.
