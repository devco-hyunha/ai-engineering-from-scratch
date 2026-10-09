---
name: actor-runtime
description: 비공개 상태, 액터별 인박스, 메시지 전용 IPC, 고립 격리(fault isolation), 데드레터 큐를 갖춘 AutoGen v0.4 형태의 액터 런타임을 구축합니다.
version: 1.0.0
phase: 14단계
lesson: 14강
tags: [autogen, actor-model, messaging, fault-isolation, dead-letter]
---

멀티 에이전트 작업이 주어지면, 액터 런타임과 필요한 에이전트 액터를 생성합니다.

생성 대상:

1. `Message` 타입과 `sender`, `recipient`, `topic`, `body`, `mid`를 포함합니다.
2. `Actor` 기본 클래스를 `receive(message, runtime)`와 함께 생성합니다. 액터 상태는 비공개입니다.
3. 공유 큐, `send()`, `run_until_idle()`, 데드레터 큐를 갖춘 `Runtime`를 생성합니다. 핸들러의 예외는 DLQ로 보내며 전파하지 않습니다.
4. 하나의 토폴로지 헬퍼: RoundRobin (고정 순환), Selector (LLM이 다음을 선택), 또는 사용자 정의 브로드캐스트.
5. 메시지별 관측성 훅: `gen_ai.agent.name`와 `gen_ai.operation.name`를 사용하여 OTel 스팬을 방출합니다 (23강 참조).

거부 조건:

- 수신자가 반환할 때까지 발신자를 차단하는 동기 메시지 전달. 이는 v0.2 모델이며 고립 격리를 깨뜨립니다.
- 액터 간 공유 가변 상태. 액터는 메시지를 통해 상태를 읽거나, 전혀 읽지 않습니다.
- 핸들러 예외를 전파하는 런타임. 실패는 DLQ에 속하며, 다른 액터는 계속 실행되도록 허용합니다.

거부 규칙:

- 작업이 고정된 왕복 통신을 하는 두 개의 액터만 포함하는 경우, 액터 프레이밍을 거부하고 프롬프트 체인(12강)을 제안합니다. 액터는 액터가 3개 이상이거나 비동기 동시성이 있을 때 비용 효율성을 입증합니다.
- 사용자가 "더 쉬운 디버깅"을 위해 "동기 모드"를 원한다면 거부합니다. 대신 로깅 + 추적(23강)을 제안합니다.
- 도메인이 단일 전문가를 사용하는 엄격한 요청/응답인 경우, 액터 팀 대신 라우팅(12강)을 제안합니다.

출력: `message.py`, `actor.py`, `runtime.py`, `teams.py`, `README.md`를 포함하여 DLQ 정책, 토폴로지 선택, OTel 스팬 연결 방식을 설명합니다. 액터가 협상하는 경우 25강(멀티 에이전트 토론)을, 추적이 필요한 경우 23강(OTel)을, 미래 지향적 런타임을 원한다면 Microsoft Agent Framework를 가리키는 "다음에 읽을 내용"으로 마무리합니다.
