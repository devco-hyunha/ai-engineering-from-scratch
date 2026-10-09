---
name: video-qa
description: 장면 분할, 다중 벡터 인덱싱, 시간적 접지, 타임스탬프 인용을 포함한 비디오 이해 파이프라인을 구축하세요.
version: 1.0.0
phase: 19단계
lesson: 12강
tags: [capstone, video, multimodal, gemini, qwen-vl, molmo, transnet, qdrant]
---

100시간의 비디오가 주어졌을 때, (시작, 종료) 타임스탬프와 프레임 미리보기로 자연어 질문에 답하는 인제스트 파이프라인 및 질의 시스템을 구축하세요.

구축 계획:

1. 비디오를 인제스트하세요 (YouTube URL 또는 MP4); 필요 시 720p로 다운스케일하세요.
2. TransNetV2 또는 PySceneDetect를 사용하여 장면 분할을 수행하세요; `[{scene_id, start_ms, end_ms, keyframe_path}]`를 생성하세요.
3. Whisper-v3-turbo (faster-whisper)로 ASR (자동 음성 인식)(Automatic Speech Recognition (ASR))를 수행하여 단어 단위 타임스탬프를 생성하세요; 장면별로 슬라이스하세요.
4. Gemini 2.5 Pro, Qwen3-VL-Max 또는 Molmo 2를 사용하여 VLM (비전-언어 모델)(Vision-Language Model (VLM)) 캡셔닝을 수행하세요; 캡션 + 프레임 임베딩(Embedding)을 생성하세요.
5. Qdrant 다중 벡터 인덱스를 구축하세요. 각 장면마다 세 개의 명명된 벡터 (caption_emb, frame_emb, transcript_emb)와 페이로드 {video_id, scene_id, start_ms, end_ms, keyframe_url}를 포함하세요.
6. 질의: 세 개의 병렬 밀집 검색(Dense Retrieval)을 수행하세요; 상호 랭킹 융합 (RRF)(Reciprocal Rank Fusion (RRF))로 병합하세요; top-k=5 장면을 선택하세요.
7. 시간적 접지(Grounding) (TimeLens 어댑터 또는 VideoITG)로 상위 장면 내의 (시작, 종료) 시간을 정밀화하세요.
8. 질의 + 상위 3개 장면 클립 + 전사문을 사용하여 VLM (비전-언어 모델)(Vision-Language Model (VLM)) 합성(Gemini 2.5 Pro)을 수행하세요; `(video_id, start_ms, end_ms)` 인용을 요구하세요.
9. ActivityNet-QA, NeXT-GQA 및 100개 질문의 수동 라벨 지정 커스텀 세트에서 평가(Evaluation (Eval))하세요. 전체 정확도와 질문 클래스별 (서술형, 개수 세기, 행동 유형) 정확도를 보고하세요.

평가 기준표:

| 가중치 | 기준 | 측정 |
|:-:|---|---|
| 25 | 시간적 접지(Grounding) IoU | 홀드아웃 접지(Grounding) 세트에서의 IoU |
| 20 | QA 정확도 | NeXT-GQA 및 100개 질문 커스텀 세트 |
| 20 | 인제스트 처리량 | 달러당 인덱싱된 비디오 시간 |
| 20 | UI 및 인용 UX | 타임스탬프 링크, 썸네일 스트립, 프레임으로 이동 |
| 15 | 환각(Hallucination)률 | 개수 세기 및 행동 유형 정확도를 별도로 보고 |

하드 리젝트:

- 장면당 단일 벡터를 풀링하는 파이프라인. 클래스 구분을 드러내기 위해 다중 벡터가 필수입니다.
- (시작, 끝) 인용이 없는 답변.
- 카운팅/동작 하위 집합 분해 없이 전체 정확도만 보고하는 것.
- 장면 프레임을 직접 받지 못하는 VLM 합성 (텍스트 전용 입력은 시각적 접지(Visual Grounding)를 잃습니다).

거부 규칙:

- 라이선스 출처가 불분명한 비디오는 서빙을 거부합니다. 모든 video_id에 라이선스 태그를 요구합니다.
- 측정된 처리량(throughput)을 초과하는 인제스트(rate)에서 "실시간" 응답을 주장하는 것을 거부합니다.
- 카운팅/동작 환각(Hallucination) 수치를 전체 정확도 수치 안에 숨기는 것을 거부합니다.

출력: 장면 분할(segmentation) + ASR + 캡셔닝 파이프라인, 다중 벡터 Qdrant 컬렉션, 시간적 접지(temporal grounding) 어댑터, 타임스탬프 딥 링크가 포함된 Next.js 15 뷰어, 세 가지 벤치마크 평가 결과 (ActivityNet-QA, NeXT-GQA, 사용자 정의), 그리고 관찰한 세 가지 카운팅 또는 동작 유형 실패 클래스와 각각을 감소시킨 검색(retrieval) 또는 합성(synthesis) 변경 사항을 명시한 문서가 포함된 저장소(repo).
