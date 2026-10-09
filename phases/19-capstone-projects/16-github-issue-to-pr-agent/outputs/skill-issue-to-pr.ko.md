---
name: issue-to-pr
description: 클라우드 샌드박스에서 실행되며, 빌드를 재현하고 테스트를 검증하며, 저장소별 엄격한 예산 내에서 리뷰 가능한 PR을 여는 비동기 GitHub 이슈-to-PR 에이전트를 구축합니다.
version: 1.0.0
phase: 19단계
lesson: 16강
tags: [capstone, async-agent, github, fargate, daytona, swe-bench, budget, safety]
---

라벨이 `@agent fix this`인 이슈가 있는 GitHub 저장소가 주어지면, 각 라벨이 지정된 이슈를 범위화된 자격 증명과 제한된 비용으로 리뷰 가능한 PR로 변환하는 자체 호스팅 클라우드 에이전트를 출시하세요.

구축 계획:

1. 세밀한 토큰을 사용하는 GitHub App: 이슈 읽기/쓰기, PR 쓰기, 내용 읽기/쓰기, 워크플로 읽기. 강제 푸시(force-push)는 금지됩니다. main 브랜치 보호 설정으로 직접 쓰기를 방지합니다.
2. 웹훅 수신기(Lambda 또는 Fly.io)가 라벨 / PR-댓글 이벤트를 필터링하고 SQS에 큐잉합니다.
3. 디스패처는 저장소별 일일 $ 및 PR 개수 상한을 강제하며, 허용된 작업마다 ECS Fargate 태스크를 실행합니다.
4. 환경 추론: 저장소 내용에서 언어 + 패키지 관리자 + 런타임을 감지합니다. Dockerfile이 없으면 즉시 합성합니다.
5. 태스크마다 Daytona 또는 E2B 샌드박스를 사용합니다. 저장소를 새로운 `git worktree` + 에이전트 브랜치로 클론합니다.
6. 에이전트 루프(mini-swe-agent 또는 SWE-agent v2, Claude Opus 4.7 또는 GPT-5.4-Codex 사용). 도구: ripgrep, tree-sitter 저장소 맵, read_file, edit_file, run_tests, git. 상한: $20, 30 턴(turns), 30분.
7. 검증: 샌드박스 내 전체 CI; jacoco / coverage.py를 통한 커버리지 델타; 델타가 -2% 미만이면 라벨 `needs-review`을 붙이고, CI가 빨간색이면 중단합니다.
8. GitHub API를 통해 근거, diff 요약, 추적(trace) URL, 비용, 턴(turns)을 포함하여 PR을 엽니다.
9. 관측 가능성: PR별 Langfuse 추적(trace); 시크릿(secret)을 제거하는 로그 스크럽; 저장소별 예산 대시보드.
10. 30개의 시드(seed)된 내부 이슈로 평가; 세 개의 이슈가 포함된 공유 하위 집합에서 Cursor Background Agents 및 AWS Remote SWE Agents와 비교합니다.

평가 루브릭:

| 가중치 | 기준 | 측정 |
|:-:|---|---|
| 25 | 30개 이슈의 통과율 | 엔드투엔드 성공(CI green + 커버리지 OK) |
| 20 | PR 품질 | Diff 크기, 커버리지 델타, 스타일 준수 |
| 20 | 해결된 이슈당 비용 및 지연 | $/PR 및 실시간(wall-clock)/PR |
| 20 | 안전성 | 범위 한정 토큰, 저장소별 예산, 강제 푸시 금지, 자격 증명 위생 |
| 15 | 운영자 UX | 근거 주석, 재시도 기능, @-멘션 후속 조치 |

하드 거부 조건:

- 강제 푸시가 가능한 모든 에이전트. 하드 제외 대상입니다.
- 예산 검사를 건너뛰는 디스패처. 폭주 루프는 전형적인 실패 사례입니다.
- 샌드박스 내에서 전체 CI가 통과되지 않은 상태로 열린 PR.
- 익명화되지 않은 토큰이나 PII가 포함된 추적 아카이브.

거부 규칙:

- main 브랜치 보호가 설정되지 않은 경우 설치를 거부합니다.
- 저장소별 일일 예산(달러 및 PR 수)이 없는 경우 실행을 거부합니다.
- 실패한 실행을 자동으로 재시도하지 않습니다. 모든 재시도는 사람이 라벨을 다시 적용해야 합니다.

출력: GitHub App, 웹훅 수신기, 디스패처 + 예산 원장, Fargate 태스크 정의, 샌드박스 수명 주기 관리자, mini-swe-agent 루프, 30개 이슈 평가 실행, Cursor Background Agents 및 AWS Remote SWE Agents와의 나란 비교, 그리고 상위 세 가지 빌드 추론 실패와 각각을 줄인 Dockerfile 합성 변경을 명시한 문서가 포함된 저장소입니다.
