---
name: vision-rag-designer
description: ColPali / ColQwen2 / VisRAG를 사용하여 비전 네이티브 문서 RAG를 설계하고, 저장소 추정 및 생성기 선택을 포함합니다.
version: 1.0.0
phase: 12단계
lesson: 23강
tags: [colpali, colqwen2, visrag, late-interaction, vidore]
---

문서 RAG 프로젝트(코퍼스 크기, 쿼리 지연 목표, 저장소 예산, 쿼리당 비용)가 주어지면 비전 네이티브 RAG 구성을 생성합니다.

다음 내용을 생성합니다:

1. 리트리버 선택. ColPali (PaliGemma 기반), ColQwen2 (Qwen2-VL 기반, 더 높은 품질), ColSmol (엣지용 1B), 또는 VisRAG (바이인코더, 더 저렴한 저장소).
2. 저장소 추정. N_docs * N_p_per_doc * D * 4 bytes 원시 데이터; PQ의 경우 8로 나눕니다.
3. 지연 시간 추정.
   - 리트리벌 SLA: 쿼리 임베딩 + top-k 리트리벌(MaxSim 또는 ANN)에 약 10ms 소요되며, 인덱스 크기에 따라 달라집니다.
   - 전체 답변 SLA: 리트리벌 지연 시간 + 생성기 200-500ms (모델 및 하드웨어에 따라 달라짐).
4. 생성기 선택. 오픈 소스용 Qwen2.5-VL-72B, 프론티어용 Claude Opus 4.7.
5. 압축 계획. PQ / OPQ 비율 목표는 8-16x; 빠른 ANN을 위해 HNSW 인덱스 사용.
6. 텍스트 RAG에서의 마이그레이션 경로. A/B 테스트 방법 및 완전 전환 시점.

거부 조건:
- 10k 페이지 이상의 코퍼스에 PQ 압축 없이 ColPali를 사용하는 경우. 저장소가 급증합니다.
- 바이인코더 리트리벌이 문서 리콜에서 ColBERT MaxSim과 일치한다고 주장하는 경우. ViDoRe에서는 일치하지 않습니다.
- 차트 + 테이블 워크로드에 텍스트 RAG를 권장하는 경우. 텍스트 RAG는 대부분의 신호를 잃습니다.

거부 규칙:
- 코퍼스가 순수 텍스트(위키, 채팅 로그)인 경우, 비전 네이티브 RAG를 거부하고 표준 텍스트 RAG를 권장합니다.
- 리트리벌 SLA가 100ms 미만인 경우, ColPali MaxSim보다 VisRAG (바이인코더)를 선호합니다.
- 전체 답변 SLA가 100ms 미만인 경우, 생성형 RAG를 완전히 거부하고 리트리벌 전용 UX 또는 캐시된 답변을 권장합니다.
- 저장소 예산이 1 GB 미만이고 코퍼스가 100k 페이지 이상인 경우, 완전한 충실도의 ColPali를 거부하고 공격적인 PQ 또는 VisRAG를 제안합니다.

출력: 리트리버 선택, 저장소 추정, 지연 시간, 생성기, 압축, 마이그레이션을 포함한 한 페이지 RAG 설계. arXiv 2407.01449 (ColPali), 2410.10594 (VisRAG)로 마무리합니다.
