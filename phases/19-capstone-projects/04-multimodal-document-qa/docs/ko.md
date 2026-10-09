# 캡스톤 04 — 멀티모달 문서 QA (비전 우선 PDF, 표, 차트)

> 2026년 문서 QA의 최전선은 OCR 후 텍스트 처리 방식에서 비전 우선 late interaction 방식으로 이동했습니다. ColPali, ColQwen2.5, ColQwen3-omni는 각 PDF 페이지를 이미지로 취급하고, multi-vector late interaction으로 임베딩하며, 쿼리가 패치에 직접 어텐션하도록 합니다. 금융 10-K 보고서, 과학 논문, 손글씨 노트에서는 이 패턴이 OCR 우선 방식보다 큰 격차로 우월합니다. 10k 페이지에 대해 파이프라인을 처음부터 끝까지 구축하고, OCR 후 텍스트 방식과의 병렬 비교 결과를 공개하세요.

**유형:** Capstone
**언어:** Python (파이프라인), TypeScript (뷰어 UI)
**선수 요건:** 4단계 (컴퓨터 비전), 5단계 (NLP), 7단계 (트랜스포머), 11단계 (LLM 엔지니어링), 12단계 (멀티모달), 17단계 (인프라)

**활용 단계:** P4 · P5 · P7 · P11 · P12 · P17
**시간:** 30시간

## 문제점

기업은 OCR 파이프라인이 망가뜨리는 PDF를 쌓고 있습니다. 회전된 표가 있는 스캔된 10-K 보고서, 방정식이 밀집된 과학 논문, 이미지로만 의미가 통하는 차트, 손글씨 주석 등이 그 대상입니다. 이를 텍스트 우선 방식으로 처리하면 신호의 절반을 잃게 됩니다. 2026년의 해답은 원본 페이지 이미지에 대한 late-interaction multi-vector 검색입니다. ColPali (Illuin Tech)가 이를 도입했으며, ColQwen2.5-v0.2와 ColQwen3-omni가 정확도를 높였습니다. ViDoRe v3에서 비전 우선 검색은 OCR 후 텍스트 방식보다 의미 있는 격차로 높은 점수를 기록하며, 차트, 표, 손글씨에서는 격차가 더 커집니다.

상충 관계는 저장 공간과 지연 시간입니다. ColQwen 임베딩은 페이지당 단일 1024차원 벡터가 아니라 약 2048개의 패치 벡터를 가집니다. 원본 저장 공간이 급증합니다. DocPruner (2026)는 측정 가능한 정확도 손실 없이 50%의 프루닝을 제공합니다. 10k 페이지를 인덱싱하고, ViDoRe v3 nDCG@5를 측정하며, 2초 이내로 답변을 서빙하고, OCR 후 텍스트 기준선과 직접 비교하세요.

## 개념

Late interaction은 모든 쿼리 토큰이 모든 패치 토큰에 대해 점수를 매기고, 쿼리 토큰별 최대 점수를 합산하는 방식입니다. 단일 풀드 벡터가 필요 없이 세밀한 매칭을 얻을 수 있습니다. Multi-vector 인덱스 (Vespa, Qdrant multi-vector, AstraDB)는 패치별 임베딩을 저장하고 검색 시 MaxSim을 실행합니다.

답변 생성기는 비전-언어 모델(VLM)로, 쿼리와 상위 k개 검색된 페이지를 이미지로 입력받아 증거 영역(바운딩 박스 또는 페이지 참조)과 함께 답변을 작성합니다. Qwen3-VL-30B, Gemini 2.5 Pro, InternVL3가 2026년 최첨단 선택지입니다. 수식 및 과학적 표기법을 위해 OCR 폴백(Nougat, dots.ocr)이 선택적 텍스트 채널로 결합됩니다.

평가는 2차원 매트릭스입니다. 한 축은 콘텐츠 유형(평문 단락, 밀집 표, 막대/선 차트, 손글씨 노트, 수식)이고, 다른 축은 검색 접근 방식(비전 우선 후기 상호작용 vs OCR 후 텍스트 vs 하이브리드)입니다. 각 셀은 nDCG@5와 답변 정확도를 얻습니다. 보고서가 산출물입니다.

## 아키텍처

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 스택

- 페이지 렌더링: PyMuPDF (fitz)를 180 DPI로 사용하며, 세로 방향 표준화(portrait-normalized) 적용
- 후기 상호작용 모델: ColQwen2.5-v0.2 또는 ColQwen3-omni (Hugging Face의 vidore 팀)
- 인덱스: 다중 벡터 필드를 지원하는 Vespa, Qdrant 다중 벡터, 또는 MaxSim을 지원하는 AstraDB
- 프루닝: DocPruner 2026 정책 (고변량 패치 유지, 정확도 손실 < 0.5%로 50% 압축)
- OCR 폴백 (수식 / 밀집 표): dots.ocr 또는 Nougat
- VLM 답변 생성기: Qwen3-VL-30B 셀프 호스팅 또는 Gemini 2.5 Pro 호스팅; InternVL3는 폴백으로 사용
- 평가: ViDoRe v3 벤치마크, 다중 페이지 추론을 위한 M3DocVQA
- 뷰어 UI: 증거 영역을 위한 캔버스 오버레이가 포함된 Next.js 15

```figure
ce-late-interaction
```

## 구현하기

1. **수입.** 10-K 보고서, 과학 논문, 스캔된 문서에 걸친 10k PDF 페이지 코퍼스를 순회합니다. 각 페이지를 1536x2048 PNG로 렌더링합니다. `{doc_id, page_num, image_path}`를 저장합니다.

2. **임베딩.** 각 페이지 이미지에서 ColQwen2.5-v0.2를 실행합니다. 출력 형태는 차원 128의 약 2048개 패치 임베딩입니다. DocPruner를 적용하여 가장 신호가 강한 절반을 유지합니다. Vespa 다중 벡터 필드 또는 Qdrant 다중 벡터에 기록합니다.

3. **쿼리.** 각 입력 쿼리에 대해 쿼리 타워(토큰 수준 임베딩)로 임베딩합니다. 인덱스에 대해 MaxSim을 실행합니다: 모든 쿼리 토큰에 대해 페이지 패치 임베딩 중 최대 점 곱을 취하고 합산합니다. 상위 k개 페이지를 반환합니다.

4. **합성.** 쿼리와 상위 5개 페이지 이미지를 Qwen3-VL-30B에 전달합니다. 프롬프트: "제공된 페이지만을 사용하여 답변하세요. 각 주장을 (doc_id, page)로 인용하고 영역(그림, 표, 단락)을 명시하세요."

5. **증거 영역.** 답변에서 인용된 영역을 추출하도록 후처리합니다. VLM이 바운딩 박스를 출력하는 경우(Qwen3-VL은 이를 수행합니다), 뷰어에 오버레이로 렌더링하세요.

6. **OCR 폴백.** 수식이 밀집된 페이지로 식별된 경우(이미지 분산 기반 휴리스틱), Nougat 또는 dots.ocr을 실행하고 OCR 텍스트를 이미지와 함께 추가 채널로 전달하세요.

7. **평가.** ViDoRe v3(검색 nDCG@5)와 M3DocVQA(다중 페이지 QA 정확도)를 실행합니다. 동일한 코퍼스에 동일한 합성기를 사용하여 OCR 후 텍스트 파이프라인도 실행하세요. 콘텐츠 유형 × 접근 방식 매트릭스를 생성하세요.

8. **UI.** 먼저 Streamlit 프로토타입을 만들고, 페이지별 증거 영역 오버레이가 있는 Next.js 15 프로덕션 뷰어를 구축하세요.

## 사용하기

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 출시하기

`outputs/skill-doc-qa.md`는 산출물을 설명합니다: 특정 코퍼스에 맞춰 튜닝된 비전 우선 멀티모달 문서 QA 시스템으로, ViDoRe v3에서 OCR 후 텍스트 기준선과 비교하여 평가됩니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 정확도 | OCR 텍스트 기준선 및 공개 리더보드 대비 벤치마크 수치 |
| 20 | 증거 영역 그라운딩 | 인용된 영역 중 실제로 답변 스팬을 포함하는 영역의 비율 |
| 20 | 저장 및 지연 엔지니어링 | DocPruner 압축률, 인덱스 p95, 답변 p95 |
| 20 | 다중 페이지 추론 | 수동으로 라벨링된 100개 다중 페이지 질문 세트의 정확도 |
| 15 | 출처 검토 UX | 뷰어 명확성, 오버레이 충실도, 나란히 비교 도구 |
| **100** | | |

## 연습 문제

1. 동일한 코퍼스에서 ColQwen2.5-v0.2와 ColQwen3-omni를 측정하세요. 한 모델이 맞춘 페이지와 다른 모델이 놓친 페이지는 무엇인가요? 인덱스에 "콘텐츠 클래스" 태그를 추가하여 유형에 따라 라우팅하세요.

2. 임베딩을 공격적으로 프루닝하세요(75%, 90%). 압축 클리프를 찾으세요: ViDoRe nDCG@5가 OCR 기준선 아래로 떨어지는 지점입니다.

3. 하이브리드를 구축해 보세요: OCR 후 텍스트 처리와 ColQwen을 병렬로 실행하고, RRF로 융합한 뒤 크로스 인코더로 리랭킹하세요. 하이브리드가 각각 단독으로 사용할 때보다 성능이 더 좋은가요? 가장 도움이 되는 부분은 어디인가요?

4. Qwen3-VL-30B를 더 작은 VLM(Qwen2.5-VL-7B)으로 교체하세요. 비용 대비 정확도 곡선을 측정해 보세요.

5. 손글씨 노트 지원 기능을 추가하세요. 손글씨 코퍼스를 렌더링하고, ColQwen으로 임베딩한 뒤 검색 성능을 측정하세요. 손글씨 OCR 파이프라인과 비교해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|-----------------|------------------------|
| 후기 상호작용 | "ColPali 스타일 검색" | 쿼리 토큰이 페이지 패치와 독립적으로 점수를 매기며, MaxSim이 이를 집계합니다 |
| 다중 벡터 | "패치별 임베딩" | 각 문서가 하나의 풀링된 벡터가 아닌 여러 벡터를 가집니다 |
| MaxSim | "후기 상호작용 점수 매기기" | 모든 쿼리 토큰에 대해 문서 벡터 중 최대 유사도를 취하고 합산합니다 |
| DocPruner | "패치 압축" | 2026년 pruning 기법으로, 정확도 손실이 거의 없이 패치의 50%를 유지합니다 |
| ViDoRe v3 | "문서 검색 벤치마크" | 시각 문서 검색을 측정하기 위한 2026년 표준입니다 |
| 증거 영역 | "인용된 경계 상자" | 원본 페이지에서 답변 범위를 위치시키는 bbox입니다 |
| OCR 폴백 | "수식 채널" | 수식이나 표가 많은 페이지에서 비전과 함께 사용되는 텍스트 파이프라인입니다 |

## 추가 읽기

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 후기 상호작용 문서 검색 참고 자료
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 기초 방법론 논문
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — 프로덕션 준비 체크포인트
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — 다중 페이지 멀티모달 RAG 기준선
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — 참고 서빙 스택
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 대체 인덱스
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 대체 관리형 인덱스
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 수식 지원 OCR 폴백
