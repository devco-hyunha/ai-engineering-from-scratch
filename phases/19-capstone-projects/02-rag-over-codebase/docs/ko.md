# 캡스톤 02 — 코드베이스에 대한 RAG (저장소 간 시맨틱 검색)

> 2026년, 모든 진지한 엔지니어링 조직은 단순한 문자열이 아닌 의미를 이해하는 내부 코드 검색을 운영합니다. Sourcegraph Amp, Cursor의 코드베이스 답변, Augment의 엔터프라이즈 그래프, Aider의 repomap, Pinterest의 내부 MCP는 모두 동일한 형태를 띱니다. 여러 저장소를 수집하고, tree-sitter로 파싱하며, 함수 및 클래스 단위 청크를 임베딩하고, 하이브리드 검색을 수행한 후, 리랭킹을 통해 인용문과 함께 답변을 생성합니다. 이 캡스톤에서는 10개 저장소에 걸쳐 200만 줄의 코드를 처리하고, 모든 git push 시 증분 재인덱싱을 견디는 시스템을 구축하는 것을 목표로 합니다.

**유형:** Capstone
**언어:** Python (수집), TypeScript (API + UI)
**선수 요건:** 5단계 (NLP 기초), 7단계 (트랜스포머), 11단계 (LLM 엔지니어링), 13단계 (도구), 17단계 (인프라)

**활용 단계:** P5 · P7 · P11 · P13 · P17
**시간:** 30시간

## 문제점

2026년까지 모든 프론티어 코딩 에이전트(Coding Agent)는 컨텍스트 윈도우(Context Window)만으로는 저장소 간 질문에 대한 해답을 제공하지 못하기 때문에 코드베이스 검색 레이어를 탑재하고 있습니다. Claude의 100만 토큰 컨텍스트는 도움이 되지만, 순위 기반 검색의 필요성을 없애지는 못합니다. 원시 청크에 대한 단순 코사인 유사도(Cosine Similarity) 검색은 생성된 코드, 모노레포 중복, 그리고 잘 임포트되지 않는 심볼의 롱테일(long tail)에서 결과를 오염시킵니다. 생산 환경에서의 정답은 AST-aware 청크에 대한 하이브리드 (밀집 검색(Dense Retrieval) + BM25) 검색을 수행하고, 심볼 참조 그래프를 백업으로 사용하여 리랭커(Reranker)를 적용하는 것입니다.

이 개념은 하나의 튜토리얼 저장소가 아닌 실제 플릿(fleet)을 인덱싱하고, MRR@10, 인용 충실성(citation faithfulness), 증분 신선도(incremental freshness)를 측정함으로써 학습됩니다. 실패 모드들은 인프라적입니다: 10만 개 파일의 모노레포, 절반의 파일을 수정하는 push, 그리고 네 개의 저장소를 가로질러야 정답을 얻을 수 있는 쿼리(query)가 그 예입니다.

## 개념

AST-aware 수집 파이프라인은 tree-sitter로 각 파일을 파싱하고, 함수 및 클래스 노드를 추출하며, 고정된 토큰 윈도우가 아닌 노드 경계에서 청킹(Chunking)을 수행합니다. 각 청크는 세 가지 표현을 가집니다: 밀집 임베딩(Embedding) (Voyage-code-3 또는 nomic-embed-code), 희소 BM25 용어, 그리고 짧은 자연어 요약. 이 요약은 세 번째 검색 가능한 모달리티(Modality)를 추가합니다. 사용자가 "X가 어떻게 인증되는가"라고 묻으면, 코드가 `check_permission`만 포함하고 있더라도 요약은 "authz"를 언급합니다.

검색은 하이브리드입니다. 쿼리가 밀집 검색(Dense Retrieval)과 BM25 검색을 모두 실행하고, 상위 k개를 병합한 후 교차 인코더 리랭커(Cohere rerank-3 또는 bge-reranker-v2-gemma-2b)에 유니온을 전달합니다. 재랭킹된 목록은 긴 컨텍스트 합성기(Claude Sonnet 4.7 프롬프트 캐싱 사용 또는 Llama 3.3 70B 자체 호스팅)로 전달되며, 모든 주장을 파일 및 줄 범위로 인용하도록 지시합니다. 인용이 없는 답변은 사후 필터(post-filter)에 의해 거부됩니다.

점진적 신선도 유지가 인프라 문제입니다. Git 푸시가 diff를 트리거합니다: 어떤 파일이 변경되었는지, 어떤 심볼이 변경되었는지. 영향을 받은 청크만 재임베딩합니다. 영향을 받은 파일 간 심볼 간선(import, 메서드 호출)이 재계산됩니다. 매 커밋마다 200만 줄을 재처리하지 않고도 인덱스가 일관성을 유지합니다.

## 아키텍처

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 스택

- 파싱: tree-sitter와 17개 언어 문법(Python, TS, Rust, Go, Java, C++ 등)
- 밀집 임베딩: Voyage-code-3 (호스팅) 또는 nomic-embed-code-v1.5 (자체 호스팅), bge-code-v1 폴백
- 희소 인덱스: Tantivy (Rust)와 BM25F, 심볼 이름과 본문에 대한 필드 가중치 적용
- 벡터 DB: Qdrant 1.12 하이브리드 검색 사용, 또는 5천만 벡터 미만 팀을 위한 pgvector + pgvectorscale
- 청크 요약 모델: Claude Haiku 4.5 또는 Gemini 2.5 Flash, 프롬프트 캐싱 적용
- 리랭커: Cohere rerank-3 또는 bge-reranker-v2-gemma-2b 자체 호스팅
- 오케스트레이션: LlamaIndex Workflows (인제스천용), LangGraph (쿼리 에이전트용)
- 합성기: Claude Sonnet 4.7 (1M 컨텍스트) 프롬프트 캐싱 적용
- 심볼 그래프: import 및 호출 간선을 위한 Neo4j (관리형) 또는 kuzu (임베디드)
- 관측 가능성: 검색 및 합성 단계별 Langfuse 스팬

```figure
ce-hybrid-retrieval
```

## 구현하기

1. **인제스천 워커.** 푸시 훅마다 Git 히스토리를 반복합니다. 변경된 파일을 수집합니다. 각 파일에 대해 tree-sitter로 파싱하고, 전체 소스 범위를 가진 함수 및 클래스 노드를 추출합니다. 청크 레코드를 방출합니다 `{repo, path, start_line, end_line, symbol, body}`.

2. **청크 요약기.** 청크를 Haiku 4.5 호출로 배치하고, 시스템 프리앰블에 프롬프트 캐싱을 적용합니다. 프롬프트: "이 함수의 공개 계약과 사이드 이펙트를 명시하여 한 문장으로 요약하세요." 요약본을 청크와 함께 저장합니다.

3. **임베딩 풀.** 두 개의 병렬 큐: 밀집(Voyage-code-3, 배치 크기 128) 및 요약(동일 모델, 요약 문자열 사용). 벡터를 Qdrant에 페이로드 `{repo, path, start_line, end_line, symbol, kind}`와 함께 저장합니다.

4. **BM25 인덱스.** 필드 가중치가 적용된 Tantivy 인덱스: 심볼 이름 가중치 4, 심볼 본문 가중치 1, 요약 가중치 2. "X라는 이름의 함수 찾기" 쿼리와 "X를 수행하는 함수 찾기" 쿼리를 모두 지원할 수 있습니다.

5. **심볼 그래프.** 각 청크에 대해 엣지를 기록합니다: 임포트(이 파일이 저장소 Z의 심볼 Y를 사용), 호출(이 함수가 클래스 C의 메서드 M을 호출), 상속. kuzu에 저장합니다. 쿼리 시 저장소 경계를 넘어 검색을 확장하는 데 사용됩니다.

6. **쿼리 에이전트.** 세 개의 노드를 가진 LangGraph. `retrieve`는 밀집 검색과 BM25를 병렬로 실행하고, (저장소, 경로, 심볼) 기준으로 중복을 제거합니다. `rerank`는 상위 50개에 교차 인코더를 실행하고 상위 10개를 유지합니다. `synth`는 재정렬된 청크를 컨텍스트로 포함하여 Claude Sonnet 4.7을 호출하며, 시스템 프롬프트를 캐싱하고 file:line 인용을 요구합니다.

7. **인용 강제.** 모델 출력을 파싱합니다; `(repo/path:start-end)` 앵커가 없는 모든 주장은 재질문 대상이나 삭제 대상으로 플래그가 지정됩니다. 인용된 답변만 사용자에게 반환합니다.

8. **증분 재인덱싱.** 각 웹훅에서 심볼 수준 차이를 계산합니다. 텍스트가 변경된 청크만 재임베딩합니다. 임포트가 변경된 청크의 심볼 엣지를 재계산합니다. 측정: 200만 LOC 규모 함대에서 50개 파일 푸시를 60초 미만으로 재인덱싱합니다.

9. **평가.** 100개의 저장소 간 질문에 gold file:line 답변을 라벨링합니다. MRR@10, nDCG@10, 인용 충실도(검증 가능한 앵커가 있는 주장의 비율), p50/p99 지연 시간을 측정합니다.

## 사용하기

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 출시하기

산출물 스킬 `outputs/skill-codebase-rag.md`. 저장소 코퍼스를 입력으로 받아, 인제스트 파이프라인, 하이브리드 인덱스, 쿼리 에이전트를 구축하고, 모든 저장소 간 질문에 대해 인용된 답변을 반환합니다. 평가 기준:

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 검색 품질 | 100개 질문 홀드아웃 세트에 대한 MRR@10 및 nDCG@10 |
| 20 | 인용 충실도 | 검증 가능한 file:line 앵커가 있는 답변 주장의 비율 |
| 20 | 지연 시간 및 규모 | 인덱싱된 코퍼스 크기의 10k QPS에서 p95 쿼리 지연 시간 |
| 20 | 점진적 인덱싱 정확성 | 50개 파일 커밋에서 `git push`부터 검색 가능 상태까지의 시간 |
| 15 | UX 및 답변 형식 | 인용 클릭 가능성, 스니펫 미리보기, 후속 조치 제공 |
| **100** | | |

## 연습 문제

1. Voyage-code-3을 nomic-embed-code 셀프 호스트로 교체하세요. MRR@10 델타를 측정하세요. 리랭킹이 활성화된 상태에서 격차가 좁혀지는지 보고하세요.

2. 코퍼스에 생성된 코드(LLM이 만든 보일러플레이트)를 20% 주입하고 재평가하세요. 검색 오염을 관찰하세요. 페이로드에 "generated" 플래그를 추가하고 해당 검색 결과의 가중치를 낮추세요.

3. 코퍼스 크기에 따라 Qdrant 하이브리드 검색(Hybrid Retrieval)과 pgvector + pgvectorscale을 벤치마킹하세요. 배치 크기 1에서 p99를 보고하세요.

4. 샘플링 기반 드리프트 체크를 추가하세요. 주간 단위로 100개 질문 평가 세트를 재실행하세요. MRR@10이 5% 이상 떨어지면 알림을 보내세요.

5. 교차 언어 심볼 해석으로 확장하세요. gRPC로 Go 서비스를 호출하는 Python 함수를 고려하세요. 심볼 그래프를 사용하여 이들을 연결하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| AST 인식 청킹(Chunking) | "함수 단위 분할" | 고정된 토큰 윈도우 대신 tree-sitter 노드 경계에서 코드를 자르는 것 |
| 하이브리드 검색(Hybrid Retrieval) | "밀집 + 희소" | BM25와 벡터 검색을 병렬로 실행하고, top-k를 병합한 후 리랭킹(Reranker)하는 것 |
| 교차 인코더 리랭킹(Reranker) | "2단계 랭킹" | 각 (쿼리, 후보) 쌍을 함께 점수화하는 모델로, 코사인 유사도(Cosine Similarity)보다 더 정확함 |
| 프롬프트 캐시(Prompt Cache) | "캐시된 시스템 프롬프트(System Prompt)" | 반복되는 접두어 토큰을 최대 90% 할인하는 2026 Claude / OpenAI 기능 |
| 심볼 그래프 | "코드 그래프" | 파일 및 저장소 간 import, 호출, 상속을 나타내는 엣지 |
| 인용 충실성 | "그라운딩된 답변 비율" | 앵커를 클릭하고 참조된 범위를 읽어 사용자가 검증할 수 있는 주장의 비율 |
| 점진적 재인덱싱 | "push-to-search 시간" | `git push`부터 변경된 심볼이 쿼리 가능해질 때까지의 실시간 |

## 추가 읽기

- [Sourcegraph Amp](https://ampcode.com) — 프로덕션 교차 저장소 코드 인텔리전스
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 이 캡스톤에 대한 참고 심층 분석
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 랭킹 저장소 뷰
- [Augment Code enterprise graph](https://www.augmentcode.com) — 상용 심볼 그래프 RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 참조 구현
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 세부 사항
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 크로스 인코더 참조
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — 내부 플랫폼 참조
