---
name: skill-catalog-builder
description: 명시적 발견 범위 전반에 걸쳐 경계가 설정된 에이전트 스킬 카탈로그를 구축하고, 지침 본문 로드 전에 충돌을 보고합니다.
license: MIT
metadata:
  lesson: "24"
---

# 스킬 카탈로그 빌더

에이전트 호스트가 여러 스킬 디렉터리 전반에 걸쳐 결정적인 발견이 필요할 때 이 스킬을 사용하세요.

1. `references/discovery-contract.md`를 읽어 보세요.
2. `assets/scope-policy.json`의 예시 호스트 정책을 검토하세요. 그 순서가 보편적이라고 가정하지 마세요.
3. 우선순위가 높은 것부터 낮은 것까지 범위를 나열하여 `python3 scripts/build_catalog.py project=PATH user=PATH`를 실행하세요.
4. 스킬을 활성화하기 전에 JSON `collisions` 및 `omitted` 배열을 검사하세요.
5. 선택된 SKILL.md 본문만 로드하세요. 직접 참조는 해당 본문이 이를 명시할 때만 로드하세요.

발견 중에 번들된 스크립트를 절대 실행하지 마세요. 우선순위가 같은 중복 항목을 파일 시스템의 우연한 순서에 따라 선택하지 마세요.

카탈로그 예산, 선택된 항목, 충돌 해결 및 생략된 항목을 반환하세요.
