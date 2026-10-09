---
name: mcp-threat-model
description: 메타데이터, 라우팅, 인증, MRTR, 호환성 경계를 포함하여 MCP 2026-07-28 배포에 대한 위협 모델을 작성합니다.
version: 2.0.0
phase: 13단계
lesson: 15강
tags: [mcp, security, stateless, tool-poisoning, mrtr]
---

MCP 배포가 주어지면 증거 기반의 위협 모델을 작성해 보세요. 모든 서버, 패키지, 캐시, 레지스트리 항목, 게이트웨이 라우팅이 손상될 수 있다고 가정합니다.

## 필수 입력

- 클라이언트, 게이트웨이, 서버, 인증 서버, 레지스트리의 신뢰 경계(Trust Boundary).
- 완전한 정규화된 도구(descriptor) 설명자 및 승인된 다이제스트(digest).
- 인증된 주체(principal), 발급자(issuer), 대상(audience), 범위(scope), 도구 정책(policy).
- 현재 및 레거시 프로토콜 개정(revision)이 허용되는 범위.
- MRTR (다중 왕복 요청)(Multi Round-Trip Request) 연산, 입력 스키마, 상태 보호, 재생(replay) 정책.
- 캐시 범위(scope), TTL, 구독 라우팅, 감사(audit) 보존(retention).

## 산출물

1. 와이어(wire) 검증. 요청별 버전 및 기능(capability)을 확인한 후, 버전 지원 여부보다 라우팅 헤더의 일치성을 먼저 검증합니다. 불일치 시 HTTP 400 `-32020`을 요구하며, 지원되지 않는 일치하는 버전의 경우 지원되는 데이터와 요청된 데이터를 정확히 포함하여 HTTP 400 `-32022`을 요구합니다. 알 수 없는 메서드의 경우 HTTP 404 `-32601`을 요구하며, 승인된 알림(notification)의 경우 빈 본문(body)으로 202 응답을 요구합니다.
2. 설명자(descriptor) 검토. 도구 오염(poisoning) 지표, 전체 설명자 다이제스트(digest) 변경, 알 수 없는 도구, 스키마(schema) 또는 주석(annotation) 변경을 보고합니다.
3. 네임스페이스(namespace) 맵. 모든 백엔드 도구에 대해 하나의 자격이 부여된(public) 이름을 지정하고, 조용한 충돌(collision) 해결을 거부합니다.
4. 인증(authorization) 매트릭스. 인증된 주체(principal)와 발급자(issuer)를 리소스, 도구, 인자(argument) 제약 조건, 범위(scope)에 매핑합니다. `clientInfo` 또는 `serverInfo`을 식별자(identity)로 사용하지 마세요.
5. MRTR (다중 왕복 요청)(Multi Round-Trip Request) 검토. 모든 `inputRequests` 항목이 클라이언트가 선언한 기능(capability)에 의해 지원되는 완전한 내장(embedded) 요청임을 확인합니다. `elicitation: {}`은 암시적 형식 지원으로, `elicitation: {form: {}}`는 명시적 형식 지원으로 취급합니다. URL 전용 유도(elicitation)는 HTTP 400 `-32021` 및 `data.requiredCapabilities.elicitation.form`로 거부합니다. 보호된 `requestState`를 메서드, 도구, 정확한 인자, 주체(principal), 목적, 만료(expiry), 논스(nonce)에 바인딩합니다. 모든 핸들러 인스턴스가 공유하는 제한된 TTL-프루닝(pruning) 재생(replay) 저장소에서 논스를 원자적으로 소비하기 전에 모든 `inputResponses` 항목을 키(key)로 매칭하고 검증합니다.
6. 위험 축 검토. 신뢰할 수 없는 입력, 민감한 데이터, 중대한 조치를 결합하는 자동화된 단계를 모두 플래그하세요.
7. 캐시 및 구독 검토. 사용자 의존 결과가 비공개인지 확인하고, 장수명 알림은 `subscriptions/listen`를 사용해야 합니다.
8. 호환성 경계. 이전 핸드셰이크, 세션, GET 스트림, 서버 콜백, 실험적 작업 동작을 명시적인 버전 게이팅 뒤에 격리하세요.
9. 전송 경계. 구현이 완전한 HTTP 어댑터인지 프로세스 내 프로토콜 모델인지 식별하세요. JSON Content-Type 및 JSON plus SSE Accept 검증을 위해 모델과 09강을 연결하세요.
10. 수정 순서. 가장 영향력이 큰 세 가지 수정 사항을 소유자와 수용 증거와 함께 제시하세요.

## 하드 리젝트

- 발견 순서에 의한 조용한 도구 덮어쓰기 또는 라우트 선택.
- 인간 또는 정책 재승인 없이 디스크립터 다이제스트 업데이트.
- 자기 보고된 클라이언트 또는 서버 정보를 인증으로 취급.
- 선언된 기능을 권한으로 취급.
- 중대한 조치를 위해 평문 또는 서명되지 않은 `requestState`를 신뢰.
- 단일 게이트웨이 또는 서버 인스턴스 내부에 유일한 재생 장부를 유지.
- 속도 제한 또는 승인 상태를 `Mcp-Session-Id`만으로 키링.
- 폐기된 Sampling, Roots, Logging 또는 레거시 HTTP plus SSE를 새로운 구현 경로로 제시.

## 출력 형식

신뢰 경계, 전송 발견 사항, 디스크립터 발견 사항, 라우트 맵, 권한 매트릭스, MRTR 발견 사항, 호환성 발견 사항, 수정 사항으로 명명된 섹션을 반환하세요. 확인된 증거와 가정을 분리하세요. 현재 가장 많은 경계를 넘나드는 단일 공격 경로로 마무리하세요.
