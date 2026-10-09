# 캡스톤 10 — 다중 에이전트 소프트웨어 엔지니어링 팀

> 2026년 다중 에이전트 엔지니어링 팀의 형태는 수렴했습니다. 아키텍트가 계획하고, N명의 코더가 병렬 워크트리에서 작업하며, 리뷰어가 게이트 역할을 하고, 테스터가 검증합니다. SWE-AF의 팩토리 아키텍처, MetaGPT의 역할 기반 프롬팅, AutoGen 0.4의 타입 지정된 액터 그래프, Cognition의 Devin, Factory의 Droids는 모두 독립적으로 이 형태에 도달했습니다. 병렬 워크트리는 벽 시계 시간을 처리량으로 변환합니다. 공유 상태와 핸드오프 프로토콜이 실패 표면이 됩니다. 캡스톤은 팀을 구축하고, SWE-bench Pro로 평가하며, 어떤 핸드오프가 얼마나 자주 깨지는지 보고하는 것입니다.

**유형:** Capstone
**언어:** Python / TypeScript (에이전트), Shell (워크트리 스크립트)
**선수 요건:** 11단계 (LLM 엔지니어링), 13단계 (도구), 14단계 (에이전트), 15단계 (자율), 16단계 (다중 에이전트), 17단계 (인프라)

**활용 단계:** P11 · P13 · P14 · P15 · P16 · P17
**시간:** 40시간

## 문제점

단일 에이전트 코딩 하네스는 대규모 작업에서 한계에 부딪힙니다. 개별 에이전트가 약해서가 아니라, 200k 토큰 컨텍스트가 아키텍처 계획, 네 개의 병렬 코드베이스 슬라이스, 리뷰어 코멘터리, 테스트 출력 등을 모두 담을 수 없기 때문입니다. 다중 에이전트 팩토리는 문제를 분할합니다. 아키텍트가 계획을 소유하고, 코더가 병렬 워크트리에서 구현을 소유하며, 리뷰어가 게이트 역할을 하고, 테스터가 검증합니다. SWE-AF의 "팩토리" 아키텍처, MetaGPT의 역할, AutoGen의 타입 지정된 액터 그래프 — 세 가지 프레이밍 모두 동일한 형태를 설명합니다.

실패 표면은 핸드오프입니다. 아키텍트가 코더가 구현할 수 없는 것을 계획합니다. 코더가 충돌하는 diff를 생성합니다. 리뷰어가 환각(Hallucination)된 수정을 승인합니다. 테스터가 아직 작성 중인 코더와 경쟁합니다. 이 팀 중 하나를 구축하고, 50개의 SWE-bench Pro 이슈에서 실행하며, 모든 핸드오프를 추적하고, 사후 분석(Postmortem)을 공개해 보세요.

## 개념

역할은 타입이 지정된 에이전트입니다. **아키텍트** (Claude Opus 4.7)는 이슈를 읽고 계획을 작성하며, 명시적인 인터페이스를 가진 하위 작업으로 분해합니다. **코더** (Claude Sonnet 4.7, N개의 병렬 인스턴스, 각각 `git worktree` + Daytona 샌드박스)는 하위 작업을 독립적으로 구현합니다. **리뷰어** (GPT-5.4)는 병합된 diff를 읽고 승인하거나 특정 변경을 요청합니다. **테스터** (Gemini 2.5 Pro)는 테스트 스위트를 격리된 환경에서 실행하고 산출물과 함께 통과/실패를 보고합니다.

통신은 공유 작업 보드(파일 기반 또는 Redis)를 통해 이루어집니다. 각 역할은 처리가 허용된 작업을 소비합니다. 핸드오프는 A2A 프로토콜 타입의 메시지입니다. 조정 고려 사항: 병합 충돌 해결(코디네이터 역할 또는 자동 3-way 병합), 공유 상태 동기화(코더가 시작하면 계획이 동결되며, 재계획은 별도 이벤트임), 리뷰어 게이트키핑(리뷰어는 자신의 변경이나 제안한 변경을 승인할 수 없음).

토큰 증폭은 숨겨진 비용입니다. 모든 역할 경계는 요약 프롬프트와 핸드오프 컨텍스트를 추가합니다. 단일 에이전트의 40턴 실행은 4개 역할에 걸쳐 총 160턴이 됩니다. 평가 기준은 토큰 효율성을 단일 에이전트 기준선과 비교하여 가중치를 부여합니다. 질문은 "멀티 에이전트가 작동하는가"가 아니라 "달러당 이득이 있는가"이기 때문입니다.

## 아키텍처

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 스택

- 오케스트레이션: 공유 상태 + 에이전트별 서브 그래프를 사용하는 LangGraph
- 메시징: 타입이 지정된 에이전트 간 메시지를 위한 A2A 프로토콜 (Google 2025)
- 모델: Opus 4.7 (아키텍트), Sonnet 4.7 (코더), GPT-5.4 (리뷰어), Gemini 2.5 Pro (테스터)
- 워크트리 격리: 코더별 `git worktree add` + Daytona 샌드박스
- 병합 코디네이터: 커스텀 3-way 병합 + LLM 매개 충돌 해결
- 평가: SWE-bench Pro (50개 이슈), SWE-AF 시나리오, 단위 테스트용 HumanEval++
- 관측 가능성: 역할 태그가 지정된 스팬, 에이전트별 토큰 회계를 사용하는 Langfuse
- 배포: 각 역할을 별도 Deployment로 구성하고 백로그에 HPA를 적용한 K8s

```figure
ce-team-handoff
```

## 구현하기

1. **작업 보드.** 타입이 지정된 메시지를 포함하는 파일 기반 JSONL: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. 에이전트는 태그를 구독합니다.

2. **설계자.** GitHub 이슈를 읽고, 하위 작업 인터페이스(접촉 파일, 공개 함수, 테스트 영향)를 명시적으로 요구하는 계획 템플릿으로 Opus 4.7을 실행합니다. 하위 작업의 DAG가 포함된 `plan_request`을 생성합니다.

3. **코더.** N개의 병렬 워커가 보드에서 하나의 하위 작업을 각각 가져갑니다. 각 워커는 새로운 `git worktree add` 브랜치와 Daytona 샌드박스를 생성합니다. 하위 작업을 구현합니다. 패치와 테스트 변경 사항이 포함된 `diff_ready`을 생성합니다.

4. **병합 조정자.** 모든 코더가 완료되면, N개의 브랜치를 스테이징 브랜치로 3-way 병합합니다. 파일 수준 겹침이 존재할 때만 LLM 중재 충돌 해결을 수행합니다.

5. **리뷰어.** GPT-5.4가 병합된 diff를 읽습니다. 본인이 작성한 diff는 승인할 수 없습니다. `approved` (no-op) 또는 관련 코더에게 라우팅된 특정 변경 요청이 포함된 `review_feedback`을 생성합니다.

6. **테스터.** Gemini 2.5 Pro가 깨끗한 샌드박스에서 테스트 스위트를 실행합니다. 아티팩트를 캡처합니다. 스택트레이스가 포함된 `test_passed` 또는 `test_failed`을 생성합니다. 실패한 테스트는 실패한 하위 작업을 소유한 코더에게 루프백합니다.

7. **핸드오프 회계.** 역할 경계를 넘나드는 모든 메시지는 Langfuse에서 페이로드 크기와 사용된 모델이 포함된 스팬을 받습니다. 하위 작업별 토큰 증폭을 계산합니다 (coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens).

8. **평가.** 50개의 SWE-bench Pro 이슈에서 실행합니다. 단일 에이전트 기준선(단일 워크트리의 Sonnet 4.7)과 비교하여 pass@1 및 해결된 이슈당 비용($)을 비교합니다.

9. **사후 분석.** 각 실패한 이슈에 대해 깨진 핸드오프를 식별합니다(계획이 너무 모호함, 병합 충돌, 리뷰어 오탐 승인, 테스터 불안정성). 핸드오프 실패 히스토그램을 생성합니다.

## 사용하기

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 출시하기

`outputs/skill-multi-agent-team.md`이 산출물입니다. 이슈 URL과 병렬화 수준이 주어지면, 팀은 역할별 토큰 회계와 함께 병합 준비가 된 PR을 생성합니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 매칭된 50개 이슈 하위 집합, pass@1 |
| 20 | 병렬 속도 향상 | 단일 에이전트 기준선 대비 벽시계 시간 |
| 20 | 리뷰 품질 | 주입된 버그 탐지에서의 오탐 승인율 |
| 20 | 토큰 효율성 | 단일 에이전트 대비 해결된 이슈당 총 토큰 수 |
| 15 | 조정 엔지니어링 | 병합 충돌 해결, 핸드오프 실패 히스토그램 |
| **100** | | |

## 연습 문제

1. 실행 중 diff에 명백한 버그를 주입해 보세요 (본문 앞에 `return None`을 추가). 리뷰어의 오탑승(false-approve) 비율을 측정하세요. 오탑승이 5% 미만으로 떨어질 때까지 리뷰어 프롬프트를 조정하세요.

2. 코더를 두 명으로 줄여 보세요 (아키텍트 + 코더 + 리뷰어 + 테스터, 코더가 두 하위 작업을 순차적으로 실행). 벽시계 시간(wall-clock)과 통과율을 비교하세요.

3. 병합 조정자를 단일 작성자 제약으로 대체해 보세요 (하위 작업이 서로 겹치지 않는 파일 집합을 건드림). 아키텍트에게 주어지는 계획 부담을 측정하세요.

4. 리뷰어를 GPT-5.4에서 Claude Opus 4.7로 교체해 보세요. 오탑승 비율과 토큰 비용 차이를 측정하세요.

5. 다섯 번째 역할인 문서 작성자(documenter, Haiku 4.5)를 추가해 보세요. 리뷰 후, 변경 로그 항목을 생성합니다. 문서화 품질이 추가 토큰 비용을 정당화하는지 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| 병렬 워크트리 | "격리된 브랜치" | `git worktree add`이 코더마다 새로운 작업 트리를 생성 |
| 작업 보드 | "공유 메시지 버스" | 에이전트가 구독하는 타입 지정된 메시지의 파일 또는 Redis 저장소 |
| 핸드오프 | "역할 경계" | 한 역할의 컨텍스트에서 다른 역할의 컨텍스트로 전달되는 모든 메시지 |
| 토큰 증폭 | "다중 에이전트 오버헤드" | 같은 작업에 대해 역할 간 총 토큰 수 / 단일 에이전트 토큰 수 |
| A2A 프로토콜 | "에이전트 간 통신" | 타입 지정된 에이전트 간 메시지에 대한 Google의 2025 사양 |
| 병합 조정자 | "통합자(Integrator)" | 3-way 병합을 실행하고 충돌을 중재하는 구성 요소 |
| 오탑승 | "리뷰어 환각" | 리뷰어가 알려진 버그가 있는 diff를 승인하는 것 |

## 추가 읽기

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 2026년 다중 에이전트 팩토리의 참고 자료
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 역할 기반 다중 에이전트 프레임워크
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft의 타입 지정된 액터 프레임워크
- [Cognition AI (Devin)](https://cognition.ai) — 참고 제품
- [Factory Droids](https://www.factory.ai) — 대체 참고 제품
- [Google A2A protocol](https://a2a-protocol.org/latest/) — 에이전트 간 메시지 사양
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — 격리 기반 구조
- [SWE-bench Pro](https://www.swebench.com) — 평가 대상
