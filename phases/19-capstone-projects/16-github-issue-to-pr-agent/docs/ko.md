# 캡스톤 16 — GitHub 이슈에서 PR까지 자율 에이전트

> 이슈에 라벨을 붙이면 PR이 생성됩니다. 2026년 자율 코딩 에이전트의 제품 형태는 다음과 같습니다: 클라우드 샌드박스에서 에이전트를 실행하고, 테스트 통과를 검증한 후, 근거를 담은 리뷰 준비가 완료된 PR을 게시합니다. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules가 모두 이 기능을 제공합니다. 어려운 부분은 저장소의 빌드 환경을 자동으로 재현하는 것, 자격 증명 유출을 방지하는 것, 저장소별 예산을 강제하는 것, 그리고 에이전트가 강제 푸시(force-push)를 할 수 없도록 보장하는 것입니다. 이 캡스톤은 자체 호스팅 버전을 구축하고, 비용 및 통과율 측면에서 호스팅 대안과 비교합니다.

**유형:** Capstone
**언어:** Python (에이전트), TypeScript (GitHub App), YAML (Actions)
**선수 요건:** 11단계 (LLM 엔지니어링), 13단계 (도구), 14단계 (에이전트), 15단계 (자율), 17단계 (인프라)

**활용 단계:** P11 · P13 · P14 · P15 · P17
**시간:** 30시간

## 문제점

비동기 클라우드 코딩 에이전트는 인터랙티브 코딩 에이전트(캡스톤 01)와 다른 제품 카테고리입니다. UX는 GitHub 라벨입니다. 이슈에 라벨을 붙이면 `@agent fix this`, 워커가 클라우드 샌드박스에서 시작되어 저장소를 클론하고, 테스트를 실행하고, 파일을 편집하고, 검증한 후, 에이전트의 근거를 본문에 담은 PR을 엽니다. 인터랙티브 루프도, 터미널도 없습니다. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules, Factory Droids가 모두 이 형태로 수렴합니다.

엔지니어링 과제는 구체적입니다: 환경 재현(에이전트는 캐시된 개발 이미지 없이 저장소를 처음부터 빌드해야 함), 불안정한 테스트(재실행하거나 격리해야 함), 자격 증명 범위 지정(최소 세분화된 권한을 가진 GitHub App), 저장소별 일일 예산 강제, 그리고 강제 푸시 금지 정책. 이 캡스톤은 통과율, 비용, 안전성을 호스팅 대안과 비교하여 측정합니다.

## 개념

트리거는 GitHub 웹훅(이슈 라벨 또는 PR 댓글)입니다. 디스패처가 ECS Fargate 또는 Lambda에 작업을 큐잉합니다. 워커는 저장소로부터 언어 및 프레임워크를 추론하여 생성된 범용 Dockerfile을 사용하여 Daytona 또는 E2B 샌드박스에 저장소를 가져옵니다. 에이전트는 Claude Opus 4.7 또는 GPT-5.4-Codex를 대상으로 mini-swe-agent 또는 SWE-agent v2 루프를 실행합니다. 코드를 읽고, 수정을 제안하고, 패치를 적용하고, 테스트를 실행하는 과정을 반복합니다.

검증은 게이트 단계입니다. PR이 열리기 전에 샌드박스에서 전체 CI가 통과해야 합니다. 커버리지 델타가 계산되며, 임계값을 초과하는 음수 값이 나오면 PR은 열리지만 `needs-review`으로 라벨이 붙습니다. 에이전트는 근거를 PR 설명에 게시하며, 리뷰어가 후속 조치를 위해 ping할 수 있는 `@agent` 스레드를 남깁니다.

안전성은 두 가지 다른 GitHub 인터페이스를 통해 범위가 지정됩니다. 앱은 `workflows: read`과 좁은 저장소 내용/PR 범위를 가진 짧은 수명의 설치 토큰을 제공하며; 브랜치 보호(앱 권한이 아님)는 "`main`에 대한 직접 쓰기 금지"와 "강제 푸시 금지"를 강제합니다 — 앱은 우회 목록에 추가되지 않습니다. `.github/workflows`에 대한 경로 범위 읽기 전용 접근은 실제 GitHub App 기본 기능이 아니므로, 에이전트의 파일 편집 허용 목록이 워커 수준에서 이를 강제해야 합니다. 저장소별 일일 예산 상한은 디스패처에서 강제됩니다(예: 저장소당 하루 최대 5개 PR, PR당 $20).

## 아키텍처

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 스택

- 트리거: 세밀한 토큰을 사용하는 GitHub App; Lambda 또는 Fly.io를 통한 웹훅 수신기
- 워커: ECS Fargate 태스크(또는 GitHub Actions 셀프 호스트 러너)
- 샌드박스: 태스크별 Daytona devcontainer 또는 E2B 샌드박스
- 에이전트 루프: Claude Opus 4.7 / GPT-5.4-Codex 기반의 mini-swe-agent 기본값 또는 SWE-agent v2
- 검색: tree-sitter 저장소 맵 + ripgrep
- 검증: 샌드박스 내 전체 CI + 커버리지 델타 게이트
- 관측 가능성: PR 본문에 연결된 PR별 추적 아카이브가 포함된 Langfuse
- 예산: 저장소별 일일 달러 상한; 저장소당 하루 최대 PR 수

```figure
cf-issue-to-pr
```

## 구현하기

1. **GitHub App.** 세밀한 설치 토큰: issues 읽기+쓰기, pull_requests 쓰기, contents 읽기+쓰기, workflows 읽기. 브랜치 보호(이 작업을 수행할 수 있는 유일한 인터페이스)는 "`main`에 대한 직접 푸시 금지"와 "강제 푸시 금지"를 강제합니다; 앱은 우회 목록에 포함되지 않습니다. GitHub App 권한은 경로 범위가 아니므로, 워커는 제안된 diff에 대한 허용 목록 체크로서 "`.github/workflows` 아래에 쓰기 금지"를 강제합니다.

2. **웹훅 수신기.** Lambda 함수는 issue 라벨 / PR 댓글 웹훅을 수신합니다. 라벨 `@agent fix this`으로 필터링합니다. SQS에 큐잉합니다.

3. **디스패처.** SQS에서 작업을 가져옵니다. 저장소별 일일 예산을 강제합니다. 저장소 URL, 이슈 본문, 새로운 Daytona 샌드박스를 사용하여 ECS Fargate 작업을 실행합니다.

4. **환경 추론.** 언어(Python, Node, Go, Rust)와 패키지 관리자(uv, pnpm, go mod, cargo)를 감지합니다. Dockerfile이 존재하지 않으면 즉시 생성합니다.

5. **에이전트 루프.** Claude Opus 4.7을 사용하는 mini-swe-agent 또는 SWE-agent v2. 도구: ripgrep, tree-sitter 저장소 맵, read_file, edit_file, run_tests, git. 하드 제한: $20 비용, 30분 벽시계 시간, 30 에이전트 턴.

6. **검증.** 루프가 종료된 후 샌드박스 내에서 전체 테스트 스위트를 실행합니다. jacoco / coverage.py를 통해 커버리지 델타를 계산합니다. CI가 빨간색이면: 중단하고 PR을 열지 마세요. 커버리지가 2% 이상 감소하면: `needs-review` 라벨로 PR을 엽니다.

7. **PR 게시.** 에이전트 브랜치를 푸시합니다. GitHub API를 통해 제목, 근거, diff 요약, 추적 URL, 비용, 턴 수와 함께 PR을 엽니다.

8. **자격 증명 위생.** 워커는 단기 GitHub App 설치 토큰으로 실행됩니다. 아카이빙 전에 로그에서 비밀을 제거합니다.

9. **평가.** 난이도가 다양한 30개의 시드 내부 이슈. 통과율, PR 품질(diff 크기, 스타일, 커버리지), 비용, 지연 시간을 측정합니다. 동일한 이슈에서 Cursor Background Agents 및 AWS Remote SWE Agents와 비교합니다.

## 사용하기

```
# github.com에서
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 출시하기

`outputs/skill-issue-to-pr.md`는 산출물입니다. 라벨이 지정된 이슈를 비용이 제한되고 자격 증명이 범위가 지정된 리뷰 준비가 완료된 PR로 변환하는 GitHub App + 비동기 클라우드 워커입니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 30개 이슈 통과율 | 엔드투엔드 성공(CI 초록 + 커버리지 OK) |
| 20 | PR 품질 | diff 크기, 커버리지 델타, 스타일 준수 |
| 20 | 해결된 이슈당 비용 및 지연 시간 | PR당 $ 및 벽시계 시간 |
| 20 | 안전성 | 범위 지정된 토큰, 저장소별 예산, force-push 금지, 자격 증명 위생 |
| 15 | 운영자 UX | 근거 댓글, 재시도 기능, @-멘션 후속 조치 |
| **100** | | |

## 연습 문제

1. "불안정한 테스트 수정" 모드를 추가하세요: 라벨 `@agent stabilize-flake TestX`이 샌드박스 내에서 테스트를 50번 실행하고, 이를 안정화하는 최소한의 변경 사항을 제안합니다.

2. 세 개의 공유 이슈에 대해 Cursor Background Agents와 비용을 비교하세요. 어떤 도구가 어떤 부분에서 우세한지 보고하세요.

3. 비용 대시보드를 구현하세요: 저장소별 일일 비용, 사용자별 비용. 이상 징후가 발생하면 알림을 보내세요.

4. CI를 실행하지 않고 초안 PR을 여는 "드라이 런(dry-run)" 모드를 구축하세요. 리뷰어가 계획을 저렴하게 검토할 수 있도록 하기 위함입니다.

5. 보존 정책을 추가하세요: 병합되지 않은 7일 이상 된 PR 브랜치는 자동으로 삭제됩니다.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| GitHub App | "범위가 지정된 봇 신원" | 세밀한 권한 + 짧은 수명의 설치 토큰을 가진 앱 |
| 비동기 클라우드 에이전트 | "백그라운드 에이전트" | 터미널이 아닌 클라우드 샌드박스에서 실행되는 비인터랙티브 워커 |
| 환경 추론 | "Dockerfile 합성" | 언어 + 패키지 매니저를 감지하고, Dockerfile이 없으면 생성 |
| 검증 | "샌드박스 내 CI" | PR을 열기 전에 워커 내에서 전체 테스트 스위트를 실행 |
| 커버리지 델타 | "커버리지 보존" | 베이스에서 에이전트 브랜치까지의 테스트 커버리지 % 변화 |
| 저장소별 예산 | "일일 상한" | 디스패처에서 강제되는 달러 및 PR 수 상한 |
| 근거 | "PR 본문 설명" | 에이전트가 변경 사항과 이유를 요약한 내용; PR 본문에 필수 |

## 추가 읽기

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 표준 비동기 클라우드 에이전트 참조
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 참조
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 상업적 대안
- [OpenAI Codex (cloud)](https://openai.com/codex) — 호스팅 경쟁사
- [Google Jules](https://jules.google) — Google의 호스팅 버전
- [Factory Droids](https://www.factory.ai) — 대체 상업적 참조
- [GitHub App documentation](https://docs.github.com/en/apps) — 범위가 지정된 봇 신원
- [Daytona cloud sandboxes](https://daytona.io) — 참조 샌드박스
