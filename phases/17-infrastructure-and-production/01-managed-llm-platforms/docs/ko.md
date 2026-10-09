# 관리형 LLM 플랫폼 — Bedrock, Vertex AI, Azure OpenAI

> 세 개의 하이퍼스케일러, 세 가지 서로 다른 전략. AWS Bedrock은 모델 마켓플레이스입니다. Claude, Llama, Titan, Stability, Cohere가 하나의 API 뒤에 있습니다. Azure OpenAI는 OpenAI와의 독점 파트너십이며, 전용 용량에 대한 Provisioned Throughput Units (PTUs)를 제공합니다. Vertex AI는 Gemini 우선 전략으로, 가장 긴 컨텍스트와 멀티모달 스토리를 자랑합니다. 2026년 Artificial Analysis는 Llama 3.1 405B 동등 모델에서 Azure OpenAI의 중앙값을 약 50 ms, Bedrock을 약 75 ms로 측정했습니다. PTUs가 이 격차를 설명합니다. 전용 용량이 공유 온디맨드보다 더 빠르기 때문입니다. 결정 규칙은 "어디가 가장 빠른가"가 아니라 "어떤 모델 카탈로그와 FinOps 표면이 내 제품에 맞는가"입니다. 이 강의는 직관(vibes)이 아닌, 명시된 트레이드오프를 바탕으로 선택하는 방법을 가르쳐 줍니다.

**유형:** Learn
**언어:** Python (표준 라이브러리, 장난감 비용 및 지연 비교 도구)
**선수 요건:** 11단계 (LLM 엔지니어링), 13단계 (도구 및 프로토콜)
**시간:** 약 60분

## 학습 목표

- 세 가지 플랫폼 전략(마켓플레이스 vs 독점 vs Gemini 우선)을 나열하고, 각각을 제품 사용 사례에 매칭해 보세요.
- Azure OpenAI에서 Provisioned Throughput Units (PTUs)가 무엇을 구매하는지, 그리고 405B 규모에서 온디맨드 Bedrock이 왜 일반적으로 약 25 ms 더 느린지 설명해 보세요.
- 각 플랫폼의 FinOps 귀속 표면(Bedrock Application Inference Profiles vs Vertex 팀별 프로젝트 vs Azure 범위 + PTU 예약)을 다이어그램으로 그려 보세요.
- "두 공급자 최소 기준" 정책을 작성하고, 단일 벤더 종속(lock-in)이 2026년에서 비싼 실수인 이유를 설명해 보세요.

## 문제점

제품에 Claude 3.7 Sonnet을 선택했습니다. 이제 이를 서빙해야 합니다. Anthropic API를 직접 호출할 수도 있고, AWS Bedrock을 통해 호출할 수도 있으며, 게이트웨이를 통해 호출할 수도 있습니다. 직접 API가 가장 단순합니다. Bedrock은 BAAs, VPC 엔드포인트, IAM, CloudWatch 귀속을 추가합니다. 게이트웨이는 공급자 간 페일오버, 통합 청구, 속도 제한을 추가합니다.

더 깊은 질문은 카탈로그입니다. 하나의 제품에서 Claude, Llama, Gemini를 모두 사용해야 한다면, Bedrock과 Vertex, Azure OpenAI를 동시에 사용하는 것이 아닌 한 한 곳에서 모두 구매할 수 없습니다. 하이퍼스케일러는 상호 교체 가능하지 않습니다. 각 하이퍼스케일러는 모델 레이어의 소유 주체에 대해 서로 다른 전략을 취했습니다.

이 강의에서는 세 가지 전략, 지연 시간 격차, FinOps 격차, 그리고 락인 위험을 매핑합니다.

## 개념

### 세 가지 전략

**AWS Bedrock** — 마켓플레이스입니다. Claude (Anthropic), Llama (Meta), Titan (AWS 자체 모델), Stability (이미지), Cohere (임베딩), Mistral, 그리고 이미지 및 임베딩 하위 카탈로그가 포함됩니다. 하나의 API, 하나의 IAM 표면, 하나의 CloudWatch 내보내기. Bedrock의 전략은 고객이 단일 모델보다 선택권을 더 원한다는 것입니다.

**Azure OpenAI** — 독점 파트너십입니다. Azure 데이터센터에서 GPT-4 / 4o / 5 / o-series, DALL·E, Whisper, 그리고 OpenAI 모델의 미세 조정(fine-tuning)을 사용할 수 있습니다. "Azure OpenAI Service" 카탈로그에는 비-OpenAI 모델이 없습니다. 이러한 모델은 Azure AI Foundry (별도 제품)로 이동합니다. Azure의 전략은 OpenAI가 최전선(frontier)을 유지하며, 고객이 그 특정 관계에 대해 엔터프라이즈 제어 기능을 원한다는 것입니다.

**Vertex AI** — Gemini가 우선이고, 나머지는 차선입니다. Gemini 1.5 / 2.0 / 2.5 Flash 및 Pro, 그리고 Model Garden (서드파티)이 포함됩니다. Vertex의 전략은 멀티모달 롱 컨텍스트입니다. 1M 토큰 Gemini 컨텍스트가 차별점입니다.

### 대규모에서의 지연 시간 격차

Artificial Analysis는 연속 벤치마크를 실행합니다. 동등한 Llama 3.1 405B 배포(공용 온디맨드)에서 Azure OpenAI의 중앙값 첫 토큰 지연 시간은 약 50 ms이며, Bedrock은 약 75 ms입니다. 이 격차는 AWS의 실패가 아니라 용량 모델의 차이입니다. Azure는 PTU (Provisioned Throughput Units)를 판매하며, 이는 귀사의 테넌트를 위해 GPU 용량을 예약합니다. Bedrock의 동등한 기능(Provisioned Throughput)도 존재하지만, 단위당 시간당 약 $21부터 시작하며, 대부분의 고객은 공용 온디맨드를 유지합니다.

온디맨드 공용 용량은 다른 모든 고객의 트래픽과 경쟁합니다. 전용 용량은 그렇지 않습니다. 제품 SLA가 P99에서 TTFT < 100 ms라면, Azure에서 PTU를 구매하거나, Bedrock Provisioned Throughput를 구매하거나, 기본 변동을 수용해야 합니다.

### Provisioned Throughput 경제학

Azure PTU: 예약된 추론 컴퓨팅 블록입니다. 예측 가능한 워크로드의 경우 온디맨드 대비 최대 약 70%의 절감 효과가 있습니다. 트래픽과 관계없이 시간당 고정 비용이 발생하며, 유휴 상태에서도 예약 비용이 청구됩니다. 손익분기점은 일반적으로 지속적 사용률이 40-60%인 지점입니다.

Bedrock 프로비저닝 처리량: 모델 및 지역에 따라 시간당 $21-$50입니다. 계산 방식은 유사하며, 손익분기점은 피크 사용률의 절반 정도입니다. 월간 커밋이 필요합니다.

Vertex 프로비저닝 용량은 Gemini SKU 단위로 판매되며, 모델 및 지역에 따라 가격이 달라지고 공개적으로 광고되는 경우가 적습니다.

### FinOps 표면 — 진정한 차별화 요소

**Bedrock 애플리케이션 추론 프로필**은 마켓플레이스에서 가장 깔끔한 귀속(attribution)을 제공합니다. 프로필에 `team`, `product`, `feature`을 태그하고, 모든 모델 호출을 이를 통해 라우팅하면, 후처리 없이 CloudWatch가 프로필별 비용을 분리해 줍니다. 2025년에 추가되었으며, 여전히 가장 세분화된 하이퍼스케일러 네이티브 기능입니다.

**Vertex**의 귀속은 팀별 프로젝트와 모든 곳에 적용하는 레이블을 기반으로 합니다. 각 팀을 GCP 프로젝트로 모델링하고, 모든 리소스에 레이블을 붙이며, BigQuery Billing Export와 DataStudio를 사용하여 롤업(rollup)을 수행합니다. 작업량이 더 많지만, BigQuery를 통해 비용 데이터에 임의의 SQL을 적용할 수 있습니다.

**Azure**는 구독/리소스 그룹 범위와 태그에 의존하며, PTU 예약은 일급(first-class) 비용 객체로 취급됩니다. 태그는 요청이 아닌 리소스 그룹에서 상속되므로, 요청 단위 귀속은 Application Insights 커스텀 메트릭이나 헤더를 찍어내는 게이트웨이를 필요로 합니다.

패턴: Bedrock은 네이티브로 가장 깔끔하고, Vertex는 BigQuery를 통해 가장 유연하며, Azure는 계측(instrument)하지 않는 한 가장 불투명합니다.

### 2026년의 리스크는 락인(lock-in)입니다

하나의 모델이 지배하던 시절에는 단일 하이퍼스케일러 커밋이 문제없었습니다. 2026년에는 프론티어가 매월 이동합니다 — 한 분기는 Claude 3.7, 다음 분기는 Gemini 2.5, 그 다음 분기는 GPT-5입니다. 하나의 플랫폼에 락인하면 프론티어의 3분의 2가 차단됩니다.

작업 팀이 채택하는 패턴: 제품적으로 중요한 모든 LLM 호출에 대해 최소 두 개의 제공자를 사용합니다. Bedrock과 Azure OpenAI가 일반적인 조합입니다 — 한쪽에서 Claude, 다른 쪽에서 GPT를 가져오고, 둘 사이에서 페일오버(failover)를 수행하며, 동일한 게이트웨이를 사용합니다. 게이트웨이가 최적 라우팅을 수행하므로 비용 상승은 미미하며, 장애 시 가용성 상승(예: Azure OpenAI의 2025년 1월 사건, AWS us-east-1 장애)은 결정적입니다.

### 데이터 상주, BAA, 규제 산업

Bedrock: 대부분의 지역에서 BAA 지원; VPC 엔드포인트; 가드레일. 일반적인 핀테크 기본 설정입니다.
Azure OpenAI: HIPAA, SOC 2, ISO 27001; EU 데이터 상주; 기업 규제 기본 설정입니다.
Vertex: HIPAA, GDPR, 지역별 데이터 상주; Google Cloud의 컴플라이언스 스택입니다.

세 가지 모두 기본 체크박스를 충족합니다. 차이점은 데이터 보존 정책, 로그 처리 방식, 남용 모니터링이 트래픽을 읽는지 여부(대부분 기본 옵트인; 기업용 옵트아웃 가능)에 있습니다.

### 기억해야 할 수치

- Llama 3.1 405B급 모델의 Azure OpenAI 중위 TTFT: ~50 ms (PTU 사용 시).
- Bedrock 온디맨드 중위 TTFT: ~75 ms.
- Bedrock 프로비저닝 처리량: $21-$50/hr per unit.
- Azure PTU 손익분기점: ~40-60% 지속적 활용률.
- 높은 활용률에서 PTU의 온디맨드 대비 절감액: 최대 70%.

```figure
i4-platform-lanes
```

## 사용하기

`code/main.py`는 세 플랫폼을 합성 워크로드에서 비교합니다 — 온디맨드 대 PTU 경제성, TTFT 변동성, 비용 귀속 충실도를 모델링합니다. 실행하여 PTU가 어디에서 효과가 있는지, TTFT 격차보다 마켓플레이스의 모델 범위가 어디에서 더 중요한지 확인해 보세요.

## 출시하기

이 강의는 `outputs/skill-managed-platform-picker.md`를 생성합니다. 워크로드 프로필(필요한 모델, TTFT SLA, 일일 볼륨, 컴플라이언스 요구 사항)이 주어지면 주요 플랫폼, 폴백, FinOps 계측 계획을 권장합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 70B급 모델에 대해 Azure PTU가 온디맨드를 능가하는 지속적 활용률은 몇 %인가요? 손익분기점을 계산하고 광고된 40-60% 범위와 비교하세요.
2. 제품이 Claude 3.7 Sonnet과 GPT-4o를 필요로 합니다. 두 공급자 배포를 설계하세요 — 어느 것을 어느 하이퍼스케일러에 배치하고, 앞에 어떤 게이트웨이를 두며, 폴백 정책은 무엇인가요?
3. 규제 대상 헬스케어 고객이 BAA, US-East 데이터 상주, P99 TTFT 100ms 미만을 요구합니다. 플랫폼을 선택하고 세 가지 특정 기능으로 정당화하세요.
4. Bedrock 청구서가 트래픽 변화 없이 이번 달에 4배 증가한 것을 발견했습니다. 애플리케이션 추론 프로필이 없다면 범인을 어떻게 찾겠습니까? 프로필이 있다면 얼마나 걸립니까?
5. Azure OpenAI 및 Bedrock 가격 페이지를 읽어 보세요. 월 1억 토큰의 Claude 워크로드에 대해, Anthropic API 직접 사용, Bedrock 온디맨드, Bedrock Provisioned Throughput 중 어느 것이 더 저렴한가요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Bedrock | "AWS LLM 서비스" | Claude, Llama, Titan, Mistral, Cohere를 아우르는 모델 마켓플레이스 |
| Azure OpenAI | "Azure의 ChatGPT" | 엔터프라이즈 컨트롤이 적용된 Azure 데이터센터 전용 OpenAI 모델 |
| Vertex AI | "Google의 LLM" | 서드파티 모델을 위한 Model Garden을 갖춘 Gemini 우선 플랫폼 |
| PTU | "전용 용량" | Provisioned Throughput Unit — 시간당 과금되는 예약된 추론 GPU |
| Application Inference Profile | "Bedrock 태깅" | 태그가 포함된 제품별 비용/사용량 프로필, CloudWatch 네이티브 |
| Model Garden | "Vertex 카탈로그" | Gemini와 분리된 Vertex AI의 서드파티 모델 섹션 |
| Two-provider minimum | "LLM 이중화" | 모든 중요한 LLM 경로를 ≥2개의 하이퍼스케일러에서 실행하는 정책 |
| BAA | "HIPAA 서류" | Business Associate Agreement; PHI에 필수; 세 제공자 모두 제공 |
| Abuse monitoring | "로그 감시자" | 프롬프트/출력에 대한 제공자 측 안전 스캔; 엔터프라이즈에서 옵트아웃 가능 |

## 추가 읽기

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 권위 있는 요금표 및 Provisioned Throughput 가격.
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/azure-openai/) — PTU 경제성 및 요금표.
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 등급 및 Model Garden 추가 요금.
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — 제공자 간 지속적인 지연 및 처리량 벤치마크.
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 엔터프라이즈 의사결정 프레임워크.
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 귀속 메커니즘 비교.
