---
name: memory-blocks
description: 핵심 경로(critical path)에서 벗어난 수면 시간(sleep-time) 통합 에이전트를 사용하여 Letta 형식의 3계층 메모리 시스템(코어 블록, 리콜, 아카이브)을 생성합니다.
version: 1.0.0
phase: 14단계
lesson: 08강
tags: [memory, letta, blocks, sleep-time, consolidation]
---

목표 런타임, 주 모델, 그리고 (더 강력한) 수면 시간(sleep-time) 모델이 주어졌을 때, 명시적인 블록 유형과 비동기 통합을 포함하는 3계층 메모리 시스템을 생성해 보세요.

생성 대상:

1. `Block` 유형과 `label`, `value`, `limit`, `description`, `version`, `history`. 모든 쓰기는 버전을 올리고 이전 값을 기록합니다. `near_limit(threshold=0.8)`을 노출합니다.
2. 최소 세 개의 기본 블록을 포함하는 `BlockStore`: `human` (사용자에 대한 사실), `persona` (에이전트 자기 개념), `task` (현재 범위). 사용자 정의 블록을 허용합니다.
3. `Recall` 저장소 — 세션별로 페이지화된 대화 로그. 모든 턴(turn)에 자동 기록합니다. 상한(cap)에 도달하면 꼬리(tail)가 제거되지만 검색은 가능합니다.
4. `Archival` 저장소 — 최소 두 개의 백엔드(벡터, KV). 삽입 시 레코드 ID를 반환합니다. 모순이 발생하면 삭제하지 않고 무효화(invalidate)합니다.
5. 턴(turn)을 처리하며 원시(raw) 쓰기만 수행하는 `PrimaryAgent`. 핵심 경로(critical path)에서는 요약하지 않습니다.
6. 턴(turn) 사이에 실행되는 `SleepTimeAgent`: 임계값을 초과한 블록을 요약하고, 모순된 아카이브 레코드를 무효화하며, `learned_context`을 공유 블록에 기록합니다.

하드 거부(Hard rejects):

- 직접 조회(direct lookup)를 제외하고, 사용자-facing 턴(turn) 동안 동기적으로 실행되는 모든 메모리 연산. 요약, 통합, 무효화는 수면 시간(sleep-time) 패스에 속합니다.
- 모순이 발생했을 때 아카이브 레코드를 삭제하는 것. 이력을 감사(audit)할 수 있도록 무효화(invalidate)해야 합니다.
- 검토(review) 단계 없이 Persona 또는 Safety 블록에 기록하는 것. 이 블록들은 전역적으로 동작을 형성하므로, 조용한(silent) 쓰기는 버그를 숨깁니다.

거부 규칙:

- 런타임이 세션 간에 블록을 지속(persist)할 수 없다면, "메모리"라고 설명된 제품을 출시하는 것을 거부합니다. 주장을 낮추세요.
- 수면 시간(sleep-time) 에이전트가 추적(trace) 출력을 생성하지 않는다면 거부합니다. 조용한(silent) 통합은 디버깅의 사각지대입니다.
- 이력적 주장이 중요한 영역(컴플라이언스, 의료, 법률)에서 "무효화 없음, 항상 최신 쓰기를 신뢰"을 요구한다면 거부합니다.

출력: 컴포넌트별 파일 하나와 `README.md`를 생성합니다. 는 기본 블록, 수면 시간 주기, 모순 해결 정책을 명시합니다. 마지막에는 "다음에 읽을 내용"을 포함하여, 에이전트가 메모리에 대한 그래프 추론이 필요하면 09강을, 제품이 메모리 연산에 OTel 스팬이 필요하면 23강을 가리키도록 하세요.
