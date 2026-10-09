# 보안 — 시크릿, API 키 로테이션, 감사 로그, 가드레일

> 중앙 집중식 볼트(HashiCorp Vault, AWS Secrets Manager, Azure Key Vault)를 통해 시크릿 난립(secret sprawl)을 제거하세요. 자격 증명을 설정 파일, VCS의 env 파일, 스프레드시트에 저장하지 마세요. 정적 키보다 IAM 역할을 사용하며, CI/CD에는 OIDC를 사용하세요. AI 게이트웨이 패턴은 2026년 솔루션입니다: 앱 → 게이트웨이 → 모델 제공자 순으로 연결되며, 게이트웨이가 런타임에 볼트에서 자격 증명을 가져옵니다. 볼트에서 로테이션하면 모든 앱이 몇 분 내에 새로운 키를 가져갑니다 — 재배포가 필요 없고, Slack의 "새 키는 누가 가지고 있어?"라는 메시지도 없습니다. 로테이션 정책은 90일 이하로 설정하세요. 모든 커밋 시 TruffleHog / GitGuardian / Gitleaks로 스캔하세요. 제로 트러스트(Zero Trust): MFA, SSO, RBAC/ABAC, 짧은 수명의 토큰, 디바이스 보안 상태(posture)를 적용하세요. PII 제거는 엔티티 인식을 사용하여 PHI/PII를 전달하기 전에 마스킹합니다. 일관된 토큰화(Mesh 접근 방식)는 민감한 값을 안정적인 플레이스홀더에 매핑하여 LLM이 코드/관계의 의미를 보존하도록 합니다. 네트워크 아웃바운드: LLM 서비스는 전용 VPC/VNet 서브넷에 배치하고 `api.openai.com`, `api.anthropic.com` 등만 화이트리스트에 포함하며, 모든 다른 아웃바운드 트래픽은 차단하세요. 2026년 사고의 원인: Vercel 공급망 공격은 손상된 CI/CD 자격 증명을 통해 수천 개의 고객 배포에 걸친 env 변수를 유출했습니다.

**유형:** Learn
**언어:** Python (stdlib, toy PII-scrubber + audit-log writer)
**선수 요건:** 17단계 · 19단계 (AI 게이트웨이), 17단계 · 13단계 (관측 가능성)
**시간:** 약 60분

## 학습 목표

- 네 가지 시크릿 관리 안티 패턴(VCS의 설정 파일, 하드코딩된 env, 스프레드시트, 정적 키)을 나열하고, 각각의 대체 방안을 지정하세요.
- AI 게이트웨이가 볼트에서 자격 증명을 가져오는 패턴을 2026년 프로덕션 표준으로 설명하세요.
- 일관된 토큰화(같은 값 → 같은 플레이스홀더)를 사용하여 의미가 보존되도록 PII 스크러버를 구현하세요.
- 2026년 Vercel 공급망 사고의 명칭과, 이 사고가 CI/CD 자격 증명 위생에 대해 가르쳐 준 내용을 나열하세요.

## 문제점

인턴이 API 키가 포함된 `.env`를 커밋합니다. 그들은 빠르게 삭제합니다. 키는 이미 git 히스토리에 남아 있습니다. GitGuardian 스캔이 이를 포착하고, 로테이션 프로세스는 "팀에 Slack으로 알리고, 40개 설정 파일을 업데이트하고, 모든 서비스를 재배포하는 것"입니다. 8시간 후, 서비스의 절반은 라이브 상태이고, 절반은 배포 창을 기다리고 있습니다.

별도로, 사용자 프롬프트에 "My SSN is 123-45-6789."가 포함되어 있습니다. 프롬프트는 OpenAI로 전송됩니다. BAA는 있지만 내부 정책상 PII를 마스킹한 후 전달해야 합니다. 이를 수행하지 않았습니다.

별도로, EKS 클러스터의 LLM 파드는 인터넷의 모든 호스트에 도달할 수 있습니다. 누군가 공격자가 제어하는 도메인에 대한 DNS 조회를 통해 데이터를 유출합니다. 이를 차단한 것이 없습니다.

LLM 서비스의 보안은 세 가지 벡터 모두를 다루어야 합니다. Vault 기반 자격 증명, PII 스크럽, 네트워크 아웃바운드 필터링, 감사 로그입니다.

## 개념

### 중앙 집중식 Vault + IAM 역할 가져오기

**Vault**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. 단일 진실 공급원입니다.

**IAM 역할**: 앱/게이트웨이는 정적 키가 아닌 IAM 신원을 통해 인증합니다. Vault는 토큰의 수명 동안 시크릿을 반환합니다.

**AI 게이트웨이 패턴**: 게이트웨이는 요청 시 Vault에서 `OPENAI_API_KEY`를 가져옵니다. Vault에서 회전하면 다음 요청은 새 키를 받습니다. 재배포가 필요 없습니다.

### 회전 정책 ≤ 90일

모든 API 키, Vault 루트 토큰, CI/CD 자격 증명. 가능한 경우 자동 회전. 수동 회전은 기록하고 추적합니다.

### 시크릿 스캐닝

- **TruffleHog** — 커밋에 대한 정규식 + 엔트로피 검사.
- **GitGuardian** — 상용 제품, 높은 정확도.
- **Gitleaks** — 오픈 소스, CI에서 실행.

모든 커밋에서 실행합니다. 새 시크릿이 감지되면 PR을 차단합니다.

### 제로 트러스트(Zero Trust) 태세

- 모든 계정에 MFA가 필수입니다.
- SAML/OIDC를 통한 SSO.
- 세밀한 접근 제어(RBAC(역할 기반) 또는 ABAC(속성 기반)).
- 단기 수명 토큰 (일 단위, 시간 단위).
- 디바이스 태세 — 디스크 암호화가 있는 기업 디바이스만 허용.

### PII / PHI 스크럽

프롬프트가 인프라를 떠나기 전에:

1. 엔티티 인식 (spaCy NER, Presidio, 상용 도구).
2. 매칭된 엔티티 마스킹: `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`.
3. 일관된 토큰화 (Mesh 접근 방식): 동일한 값이 동일한 플레이스홀더로 매핑되어 LLM이 관계를 보존합니다.
4. LLM 응답을 위한 선택적 역매핑.

정적 정규식 필터는 기본 패턴을 잡습니다. NER은 더 많은 것을 잡습니다. 둘 다 사용하세요.

### 입력 + 출력 가드레일

입력: 알려진 제일브레이크, 금지된 주제를 차단하고 사용자별 속도 제한을 적용하세요.

출력: 유출된 비밀(API 키 패턴, 거절 컨텍스트의 이메일 패턴)에 대한 정규식 스크럽, 정책 위반에 대한 분류기를 사용하세요.

### 네트워크 아웃바운드 화이트리스트

LLM 서비스는 전용 서브넷에 배치하세요:
- 화이트리스트: `api.openai.com`, `api.anthropic.com`, 벡터 DB 엔드포인트, 볼트 엔드포인트.
- 그 외 모든 트래픽은 드롭하세요.
- DNS는 화이트리스트 전용 리졸버를 통해 처리하세요(DNS 터널링을 통한 데이터 유출 방지).

### 감사 로그

모든 LLM 호출에 대한 변경 불가능한 로그에는 다음이 포함됩니다:
- 타임스탬프.
- 사용자 / 테넌트.
- 프롬프트 해시(개인정보 보호를 위해 원본 프롬프트는 저장하지 않음).
- 모델 + 버전.
- 토큰 수.
- 비용.
- 응답 해시.
- 가드레일 트리핑 여부.

규제 요건에 따라 보존 기간을 설정하세요(SOC 2는 1년, HIPAA는 6년).

### 2026년 Vercel 사건

공급망 공격: 손상된 CI/CD 자격 증명이 수천 개의 고객 배포 환경의 환경 변수를 유출했습니다. 교훈: CI/CD 자격 증명은 프로덕션 환경과 동등합니다. 볼트에 저장하고 범위를 좁게 설정하며 aggressively 로테이션하세요.

### 기억해야 할 숫자

- 로테이션 정책: ≤ 90일.
- 모든 커밋 시 스캔: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: CI/CD 자격 증명 손상 → 수천 개의 고객 환경 변수 유출.
- 감사 로그 보존 기간: SOC 2 = 1년, HIPAA = 6년.

```figure
i4-vault-rotation
```

## 사용하기

`code/main.py`는 일관된 토큰화와 append-only 감사 로그를 갖춘 PII 스크러버를 구현합니다.

## 출시하기

이 강의는 `outputs/skill-llm-security-plan.md`을 생성합니다. 규제 범위와 현재 상태를 고려하여 볼트 마이그레이션, 스크러버, 아웃바운드, 감사 로그를 계획합니다.

## 연습 문제

1. `code/main.py`을 실행하세요. 동일한 SSN을 참조하는 두 프롬프트를 보내세요. 두 경우 모두 동일한 플레이스홀더가 반환되는지 확인하세요.
2. OpenAI + Anthropic + Weaviate를 호출하는 vLLM-on-EKS 배포를 위한 네트워크 아웃바운드(egress) 정책을 설계하세요.
3. git 히스토리에서 2년 전의 키를 발견했습니다. 올바른 대응은 키를 회전(rotating)하는 것, 히스토리를 삭제(scrubbing)하는 것, 아니면 둘 다 하는 것일까요? 근거를 설명하세요.
4. 감사 로그가 하루에 10 GB씩 증가합니다. 보존 계층(hot 30일, warm 12개월, cold 6년)을 설계하세요.
5. LLM 응답에 실제 값을 다시 넣는 역토큰화(reverse-tokenization)가 플레이스홀더를 유지하는 것보다 복잡도 대비 가치가 있는지 논쟁하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Vault | "시크릿 저장소" | 중앙화된 자격 증명 관리 서비스 |
| IAM role | "신원 기반 인증" | 앱이 가정한(role assumed) 역할; 짧은 수명의 자격 증명을 반환 |
| OIDC for CI/CD | "클라우드 발급 토큰" | CI에 정적 키 없음 — OIDC를 통해 신원 확인 |
| TruffleHog / GitGuardian / Gitleaks | "시크릿 스캐너" | 커밋 시점의 시크릿 탐지 |
| RBAC / ABAC | "접근 제어" | 역할 기반 vs 속성 기반 |
| PII scrubbing | "데이터 마스킹" | 민감한 엔티티를 제거하거나 토큰화 |
| Consistent tokenization | "안정적인 플레이스홀더" | 동일한 값 → 매번 동일한 토큰 |
| Mesh approach | "Mesh 토큰화" | 의미론적 보존(semeantic-preserving) 토큰화 패턴 |
| Egress whitelist | "아웃바운드 허용 목록" | 허용된 도메인만 도달 가능 |
| Audit log | "불변 히스토리" | 컴플라이언스를 위한 추가 전용(append-only) 기록 |

## 추가 읽기

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 탐지 및 익명화.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
