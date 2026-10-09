---
name: skill-invocation-router
description: 에이전트 스킬 카탈로그(Agent Skill catalog)에 대해 명시적 인간, 암시적 모델 또는 에이전트, 프로그래밍 방식 애플리케이션, 제한된 스킬 조합, 하네스 활성화 정책을 설계하고 테스트합니다.
license: MIT
metadata:
  lesson: "25"
---

# 스킬 호출 라우터(Skill invocation router)

호스트가 단일 구분 없는 `invocable` 플래그가 아닌 감사 가능한 활성화 정책을 필요로 할 때 이 스킬을 사용하세요.

1. `references/invocation-model.md`를 읽고 요청된 채널을 분류하세요.
2. `assets/host-policy.json`를 이식 가능한 표준이 아닌 예시 어댑터 구성으로 검토하세요.
3. `python3 scripts/simulate_invocation.py --policy assets/host-policy.json --actor ACTOR --name NAME --description DESCRIPTION --query QUERY [--explicit-name NAME] [--caller-name NAME] [--depth N] [--user-invocable true|false] [--disable-model-invocation true|false]`를 실행하세요.
4. 인간, 애플리케이션, 스킬, 또는 하네스 요청의 경우, 정확히 발견된 이름과 해당 채널별 허용 목록을 요구하세요.
5. 스킬 호출자의 경우, 호출자 신원, 비순환적 대상, 그리고 제한된 조합 깊이를 추가로 요구하세요.
6. 모델 또는 자율 에이전트 요청의 경우, 행위자(actor)나 인식된 호스트 확장 기능이 부적격하게 만드는 후보를 제거하세요.
7. 남은 설명만 점수화하세요. 가장 강력한 적격 매치를 선택하거나, 적격 후보가 임계값을 넘지 못하면 중립(abstain)하세요.
8. 어댑터, 채널, 점수, 정책 이유를 포함한 JSON 결정을 반환하세요.

활성화는 지침을 로드합니다. 도구, 파일 시스템 변경, 네트워크 접근, 비밀 사용, 또는 번들된 스크립트를 승인하지는 않습니다.
