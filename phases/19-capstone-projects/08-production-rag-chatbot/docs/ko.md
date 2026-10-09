# 캡스톤 08 — 규제 산업용 프로덕션 RAG 챗봇

> Harvey, Glean, Mendable, LlamaCloud는 2026년 모두 동일한 프로덕션 형태를 운영합니다. docling이나 Unstructured, 그리고 시각 자료용 ColPali로 데이터를 수집합니다. 하이브리드 검색(Hybrid Retrieval)을 수행합니다. bge-reranker-v2-gemma로 리랭킹(Reranker)합니다. 60-80%의 적중률로 프롬프트 캐시(Prompt Cache)를 활용하여 Claude Sonnet 4.7로 합성합니다. Llama Guard 4와 NeMo Guardrails로 보호합니다. Langfuse와 Phoenix로 모니터링합니다. 200개 질문의 골든 세트(golden set)로 RAGAS를 사용하여 채점합니다. 규제 산업(법률, 임상, 보험)에서 하나를 구축하면, 골든 세트 통과, 레드 티밍(Red Teaming), 그리고 드리프트(drift) 대시보드가 캡스톤의 통과 기준이 됩니다.

**유형:** Capstone
**언어:** Python (파이프라인 + API), TypeScript (챗 UI)
**선수 요건:** 5단계 (NLP), 7단계 (트랜스포머), 11단계 (LLM 엔지니어링), 12단계 (멀티모달), 17단계 (인프라), 18단계 (안전)

**활용 단계:** P5 · P7 · P11 · P12 · P17 · P18
**시간:** 30시간

## 문제점

규제 산업 RAG (법률 계약, 임상 시험 프로토콜, 보험 정책)는 ROI가 명확하고 위험이 구체적이기 때문에 2026년 가장 많이 배포된 프로덕션 형태입니다. Harvey (Allen & Overy)는 법률용으로 이를 구축했습니다. Mendable은 개발자 문서 형태를 배포합니다. Glean은 엔터프라이즈 검색을 커버합니다. 패턴은 다음과 같습니다: 고충실도(high-fidelity)로 데이터를 수집하고, 리랭킹(Reranker)과 함께 하이브리드 검색(Hybrid Retrieval)을 수행하며, 인용 강제(citation enforcement)와 프롬프트 캐시(Prompt Cache)를 통해 합성하고, 여러 안전 계층으로 보호하며, 드리프트(drift)를 지속적으로 모니터링합니다.

난이도는 모델에 있지 않습니다. 난이도는 관할권 인식 컴플라이언스(HIPAA, GDPR, SOC2), 인용 수준 감사 가능성, 비용 제어 (적중률이 높을 때 프롬프트 캐시(Prompt Cache)는 60-90% 할인을 제공합니다), RAGAS 충실도(faithfulness)를 통한 환각(Hallucination) 탐지, 그리고 인덱스가 따라잡지 못해 소스 문서가 업데이트될 때의 드리프트(drift) 탐지에 있습니다. 이 캡스톤은 레드 티밍(Red Teaming) 스위트와 함께 200개 질문의 골든 세트(golden set)에서 이 모든 것을 배포하는 것을 요구합니다.

## 개념

파이프라인은 두 가지 측면으로 구성됩니다. **인제스트(Ingestion)**: docling 또는 Unstructured가 구조화된 문서를 파싱하고, ColPali가 시각적으로 풍부한 문서를 처리합니다. 청크(chunks)는 요약, 태그 및 역할 기반 접근 라벨이 부여됩니다. 벡터는 pgvector + pgvectorscale (50M 벡터 미만) 또는 Qdrant Cloud에 저장되며, 희소 BM25가 병행하여 실행됩니다. **대화(Conversation)**: LangGraph가 메모리 및 다중 턴(multi-turn)을 처리하며, 각 쿼리는 하이브리드 검색(Hybrid Retrieval)을 수행하고, bge-reranker-v2-gemma-2b로 리랭킹(reranks)하며, Claude Sonnet 4.7 (프롬프트 캐시됨)으로 합성(synthesizes)하고, 출력은 Llama Guard 4 및 NeMo Guardrails를 통과하여 인용이 연결된 응답(citation-anchored response)을 생성합니다.

평가 스택은 네 가지 계층으로 구성됩니다. **골든 세트(Golden set)** (인용이 포함된 200개의 라벨링된 Q/A)는 정확성을 위해 사용됩니다. **레드 티밍(Red team)** (제일브레이크, PII 추출 시도, 도메인 밖 질문)은 안전성을 위해 사용됩니다. **RAGAS**는 각 턴(turn)마다 충실도(faithfulness) / 답변 관련성(answer relevance) / 컨텍스트 정밀도(context precision)를 자동으로 평가합니다. **드리프트 대시보드(Drift dashboard)** (Arize Phoenix)는 주간 단위로 검색 품질과 환각(Hallucination) 점수를 모니터링합니다.

프롬프트 캐싱(Prompt Caching)은 비용 절감의 핵심 수단입니다. Claude 4.5+ 및 GPT-5+는 시스템 프롬프트(System Prompt) + 검색된 컨텍스트의 캐싱을 지원합니다. 60-80%의 적중률(hit rate)에서는 쿼리당 비용이 3-5배 감소합니다. 파이프라인은 높은 캐시 적중률을 달성하기 위해 안정적인 접두어(prefixes) (시스템 프롬프트 + 리랭킹된 컨텍스트가 먼저)로 설계되어야 합니다.

## 아키텍처

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 스택

- 인제스트: 구조화된 문서의 경우 Unstructured.io 또는 docling; 시각적으로 풍부한 PDF의 경우 ColPali
- 벡터 DB: 50M 벡터 미만인 경우 pgvector + pgvectorscale; 그 외의 경우 Qdrant Cloud
- 희소 검색: 필드 가중치가 적용된 Tantivy BM25
- 오케스트레이션: LlamaIndex Workflows (인제스트) + LangGraph (대화)
- 리랭커: 자체 호스팅된 bge-reranker-v2-gemma-2b 또는 호스팅된 Voyage rerank-2
- LLM: 프롬프트 캐싱을 사용하는 Claude Sonnet 4.7; 폴백(fallback)으로 자체 호스팅된 Llama 3.3 70B
- 평가: 온라인 RAGAS 0.2, 환각 및 제일브레이크 스위트에 대한 DeepEval
- 관측 가능성: 주석 큐(annotation queue)가 포함된 Langfuse 자체 호스팅; 드리프트 모니터링을 위한 Arize Phoenix
- 가드레일: Llama Guard 4 입력/출력 분류기, NeMo Guardrails v0.12 정책, Presidio PII 스크럽(scrub)
- 컴플라이언스: 청크(chunks)에 대한 역할 기반 접근 라벨; GDPR/HIPAA를 위한 관할권 태그

```figure
canary-rollout
```

## 구현하기

1. **수입.** Unstructured 또는 docling으로 코퍼스(1000-10000 문서의 serious build)를 파싱하세요. 스캔된 페이지나 시각적 요소가 많은 페이지는 ColPali를 통해 라우팅하세요. 요약, 역할 레이블, 관할권 태그가 포함된 청크를 생성하세요.

2. **인덱싱.** Dense embeddings (Voyage-3 또는 Nomic-embed-v2)를 pgvector + pgvectorscale에 저장하세요. BM25 보조 인덱스는 Tantivy를 통해 생성하세요. 역할 및 관할권 필터는 페이로드로 포함하세요.

3. **하이브리드 검색.** 먼저 역할+관할권으로 필터링하세요. 그 다음 dense + BM25를 병렬로 수행하고, 상호 랭킹 융합 (RRF)(Reciprocal Rank Fusion (RRF))으로 병합하세요. 상위 20개를 리랭커(Reranker)로 보내고, 상위 5개를 합성(synth) 단계로 보내세요.

4. **프롬프트 캐싱으로 합성하세요.** 시스템 프롬프트(System Prompt)와 정적 정책은 캐시 헤더에 포함하세요. 리랭킹된 컨텍스트는 캐시 확장으로, 사용자 질문은 캐시되지 않은 접미사로 처리하세요. 정상 상태(steady state)에서 60-80%의 캐시 적중률을 목표로 하세요.

5. **가드레일(Guardrails).** 입력에는 Llama Guard 4를 사용하세요. NeMo Guardrails의 rails는 도메인 밖 질문이나 정책 금지 주제를 차단하세요. Presidio는 출력에 실수로 포함된 PII를 제거하세요. 인용 강제(citation enforcement)는 후처리 필터로 적용하세요.

6. **골든 세트.** 도메인 전문가가 (answer, citations)로 라벨링한 200 Q/A 쌍을 준비하세요. 에이전트(Agent)를 정확 인용 일치, 답변 정확성, 충실성(faithfulness) (RAGAS)으로 점수화하세요.

7. **레드 티밍(Red Teaming).** 50개의 적대적 프롬프트: 제일브레이크(Jailbreak) (PAIR, TAP), PII 유출 시도, 도메인 밖 질문, 관할권 간 누출. pass/fail과 심각도로 점수화하세요.

8. **드리프트 대시보드.** Arize Phoenix는 검색 품질(nDCG, 인용 충실성)을 주간 단위로 추적하세요. 5% 감소 시 알림을 설정하세요.

9. **비용 보고서.** Langfuse: 프롬프트 캐싱 적중률, 쿼리당 토큰 수, 단계별 $/쿼리 분해.

## 사용하기

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 출시하기

`outputs/skill-production-rag.md`는 산출물을 설명합니다. 컴플라이언스 라벨이 붙은 규제 대상 도메인 챗봇이 배포되어, 평가 기준(rubric)을 통과하고, 실시간 드리프트 모니터링으로 관측됩니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | RAGAS 충실성 + 답변 관련성 | 골든 세트(200 Q/A)의 온라인 점수 |
| 20 | 인용 정확성 | 검증 가능한 출처 앵커가 있는 답변의 비율 |
| 20 | 가드레일 커버리지 | Llama Guard 4 통과율 + 제일브레이크 스위트 결과 |
| 20 | 비용 / 지연 엔지니어링 | 프롬프트 캐시 적중률, p95 지연, $/쿼리 |
| 15 | 드리프트 모니터링 대시보드 | 주간 검색 품질 추세를 보여주는 Phoenix 라이브 대시보드 |
| **100** | | |

## 연습 문제

1. 다른 관할권(예: GDPR과 함께 HIPAA) 아래에 두 번째 코퍼스 슬라이스를 구축해 보세요. 20개 질문의 관할권 간 탐지에서 역할+관할권 필터링이 교차 유출을 방지하는지 시연해 보세요.

2. 1주간의 프로덕션 트래픽에서 프롬프트 캐시 적중률을 측정해 보세요. 캐시 접두어를 깨는 쿼리를 식별하고 재구조화해 보세요.

3. 10k 토큰 요약 버퍼를 사용하여 다중 턴 메모리를 추가해 보세요. 대화가 길어짐에 따라 충실도가 떨어지는지 측정해 보세요.

4. Claude Sonnet 4.7을 자체 호스팅된 Llama 3.3 70B로 교체해 보세요. 쿼리당 비용($)과 충실도 델타를 측정해 보세요.

5. "불확실" 모드를 추가해 보세요: 상위 리랭크 점수가 임계값 미만이면 에이전트가 답변하는 대신 "확신할 만한 인용이 없습니다"라고 말합니다. 허위 확신 감소량을 측정해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 프롬프트 캐싱 | "캐시된 시스템 + 컨텍스트" | Claude/OpenAI 기능: 적중 시 캐시된 접두어 토큰이 60-90% 할인됨 |
| RAGAS | "RAG 평가기" | 충실도, 답변 관련성, 컨텍스트 정밀도의 자동화된 점수화 |
| 골든 세트 | "라벨링된 평가" | 인용이 포함된 200개 이상의 전문가 라벨링 Q/A; 정답 |
| 관할권 태그 | "컴플라이언스 라벨" | 청크에 첨부된 GDPR/HIPAA/SOC2 범위; 검색 필터로 강제 적용 |
| 인용 충실도 | "그라운딩된 답변 비율" | 검색 가능한 출처 스팬으로 뒷받침되는 주장의 비율 |
| 드리프트 | "검색 품질 감소" | 주간 nDCG 또는 인용 점수 변화; 알림 임계값 5% |
| 레드 티밍 | "적대적 평가" | 릴리스 전 제일브레이크, PII 추출, 오프 도메인 탐지 |

## 추가 읽기

- [Harvey AI](https://www.harvey.ai) — 참조 법률 프로덕션 스택
- [Glean enterprise search](https://www.glean.com) — 엔터프라이즈 규모 RAG 참조
- [Mendable documentation](https://mendable.ai) — 개발자 문서 RAG 참조
- [LlamaCloud Parse + Index](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) — 관리형 인제스트
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 비용 레버 참조
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 표준 RAG 평가 프레임워크
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 참조 드리프트 관측 가능성
- [Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 2026 안전 분류기
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 정책 레일 프레임워크
