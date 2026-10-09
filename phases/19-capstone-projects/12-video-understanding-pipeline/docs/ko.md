# 캡스톤 12 — 영상 이해 파이프라인 (장면, QA, 검색)

> Twelve Labs는 Marengo + Pegasus를 상품화했습니다. VideoDB는 영상용 CRUD API를 출시했습니다. AI2의 Molmo 2는 오픈 VLM 체크포인트를 공개했습니다. Gemini의 긴 컨텍스트는 수 시간의 영상을 네이티브로 처리합니다. TimeLens-100K는 대규모 시간적 접지(temporal grounding)를 정의했습니다. 2026년 파이프라인은 정립되었습니다: 장면 분할, 장면별 캡션 + 임베딩, 전사 정렬, 다중 벡터 인덱스, 그리고 (시작, 종료) 타임스탬프와 프레임 미리보기로 답변하는 쿼리입니다. 캡스톤은 100시간의 영상을 입력하여 공개 벤치마크에 도달하고, 개수 세기 및 행동 관련 질문에 대한 환각(hallucination)을 측정하는 것입니다.

**유형:** Capstone
**언어:** Python (파이프라인), TypeScript (UI)
**선수 요건:** 4단계 (CV), 6단계 (음성), 7단계 (트랜스포머), 11단계 (LLM 엔지니어링), 12단계 (멀티모달), 17단계 (인프라)

**활용 단계:** P4 · P6 · P7 · P11 · P12 · P17
**시간:** 30시간

## 문제점

긴 영상 QA는 2026년 규모에서 가장 대역폭을 많이 소비하는 멀티모달 문제입니다. Gemini 2.5 Pro는 2시간 영상을 네이티브로 읽을 수 있지만, 100시간의 영상을 쿼리 가능한 코퍼스로 입력하려면 장면 수준 인덱스가 필요합니다. 프로덕션 형태는 장면 분할(TransNetV2 또는 PySceneDetect), VLM을 이용한 장면별 캡션 생성(Gemini 2.5, Qwen3-VL-Max, 또는 Molmo 2), 전사 정렬(Whisper-v3-turbo의 단어 타임스탬프), 그리고 캡션, 프레임 임베딩, 전사를 나란히 저장하는 다중 벡터 인덱스를 결합합니다. 쿼리 파이프라인은 (시작, 종료) 타임스탬프와 프레임 미리보기로 답변합니다.

벤치마크는 공개되어 있습니다(ActivityNet-QA, NeXT-GQA) 및 자체 100개 쿼리 커스텀 세트가 있습니다. 개수 세기 및 행동 유형 질문에 대한 환각(hallucination)은 알려진 어려운 실패 클래스이며, 캡스톤은 이를 명시적으로 측정합니다.

## 개념

세 개의 파이프라인이 입력 시 병렬로 실행됩니다. **장면 분할**은 영상을 장면으로 잘라냅니다. **VLM 캡션 생성**은 장면별 캡션과 키프레임에서 프레임 임베딩을 생성합니다. **ASR 정렬**은 단어 수준 타임스탬프를 생성합니다. 세 스트림은 (scene_id, 시간 범위)로 결합됩니다. 각 장면은 다중 벡터 인덱스(Qdrant)에서 세 가지 벡터 유형을 가집니다: 캡션 임베딩, 키프레임 임베딩, 전사 임베딩.

쿼리 시점에 자연어 질문이 세 벡터 모두에 대해 실행됩니다. 결과는 RRF로 병합되며, 시간적 접지 어댑터(TimeLens 스타일)가 상위 장면 내의 (시작, 종료) 윈도우를 정제합니다. VLM 합성기(Gemini 2.5 Pro 또는 Qwen3-VL-Max)는 쿼리 + 상위 장면 + 잘린 프레임을 입력받아 인용된 타임스탬프와 프레임 미리보기로 답변합니다.

환각(Hallucination) 측정이 중요합니다. 세기("방에 몇 명이 들어가는가?") 및 행동 유형("셰프가 저어하기 전에 따르는가?") 질문은 notoriously 신뢰도가 낮습니다. 서술형 질문과 별도로 정확도를 보고해 보세요.

## 아키텍처

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 스택

- 장면 분할: TransNetV2 (2024-26 최신 기술) 또는 PySceneDetect
- 자동 음성 인식 (ASR)(Automatic Speech Recognition (ASR)): faster-whisper를 통해 Whisper-v3-turbo 사용, 단어 타임스탬프 포함
- VLM 캡셔너 + 답변자: Gemini 2.5 Pro 또는 Qwen3-VL-Max 또는 Molmo 2
- 시간적 접지: TimeLens-100K 학습된 어댑터 또는 VideoITG
- 인덱스: 다중 벡터 지원(Qdrant) (캡션 / 프레임 / 전사)
- UI: HTML5 비디오 플레이어 및 장면 썸네일이 포함된 Next.js 15
- 평가(Evaluation (Eval)): ActivityNet-QA, NeXT-GQA, 사용자 정의 100개 질문 수동 라벨링 세트
- 환각(Hallucination) 벤치마크: 세기 및 행동 유형 하위 세트, 수동 라벨링 포함

```figure
cf-scene-index
```

## 구현하기

1. **인제스트 워커.** YouTube URL 또는 로컬 MP4를 허용합니다. 필요 시 720p로 다운스케일합니다. `{video_id, file_path}`를 저장합니다.

2. **장면 분할.** TransNetV2 또는 PySceneDetect를 실행하여 `[{scene_id, start_ms, end_ms, keyframe_path}]`를 생성합니다. 100시간 목표: 약 6k-8k 장면.

3. **ASR 패스.** 오디오에 Whisper-v3-turbo를 실행합니다. 단어 단위 타임스탬프를 내보내고, 장면별 전사 슬라이스로 분할합니다.

4. **VLM 캡셔닝.** 장면마다 Gemini 2.5 Pro (또는 Qwen3-VL-Max)를 호출하여 키프레임과 짧은 캡션 템플릿을 사용합니다. 캡션 + 프레임 임베딩을 생성합니다.

5. **다중 벡터 인덱스.** 세 개의 명명된 벡터를 가진 Qdrant 컬렉션. 페이로드: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **쿼리.** 자연어 질문이 세 개의 밀집 검색(Dense Retrieval)을 실행합니다. 상호 랭킹 융합 (RRF)(Reciprocal Rank Fusion (RRF))로 병합하며, top-k=5 장면을 선택합니다.

7. **시간적 접지.** 상위 장면에 TimeLens 스타일 어댑터를 실행하여 장면 내의 (시작, 종료) 윈도우를 정제합니다.

8. **VLM 합성.** 쿼리 + 상위 3개 장면 클립(이미지 또는 짧은 클립) + 전사문을 포함하여 Gemini 2.5 Pro를 호출합니다. `(video_id, start_ms, end_ms)` 인용을 요구합니다.

9. **평가.** ActivityNet-QA와 NeXT-GQA를 실행합니다. 100개 쿼리의 사용자 정의 세트를 구축합니다. 전체 정확도 + 클래스별 분류(세는 것, 행동, 설명적)를 보고합니다.

## 사용하기

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 출시하기

`outputs/skill-video-qa.md`는 산출물입니다. YouTube URL이나 업로드된 비디오가 주어지면 파이프라인이 장면을 인덱싱하고 타임스탬프가 있는 인용으로 질문에 답변합니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 시간적 접지 IoU | 유지된 접지 세트에 대한 교차-합 비율 |
| 20 | QA 정확도 | NeXT-GQA 및 사용자 정의 100개 쿼리 |
| 20 | 인제스트 처리량 | 지출된 비용당 비디오 시간 |
| 20 | UI 및 인용 UX | 타임스탬프 링크, 썸네일 스트립, 프레임으로 이동 |
| 15 | 환각(Hallucination) 비율 | 세는 것과 행동 유형 정확도를 별도로 |
| **100** | | |

## 연습 문제

1. 캡셔닝 패스에서 Gemini 2.5 Pro를 Qwen3-VL-Max로 교체합니다. 인간이 평가한 50개 장면 샘플에서 캡션 품질 차이를 보고합니다.

2. 장면별 프레임 임베딩을 다중 벡터 대신 하나의 풀링된 벡터로 줄입니다. 검색 회귀를 측정합니다.

3. "세는 것 엄격(strict)" 모드를 구축합니다: 합성기가 각 세는 인스턴스를 타임스탬프와 함께 추출하고 사용자가 클릭하여 검증합니다. 사용자 검증이 환각(Hallucination)을 줄이는지 측정합니다.

4. 인제스트 비용을 벤치마킹합니다: 세 가지 VLM 선택에 대해 비용당 비디오 시간을 비교합니다. 최적점을 선택합니다.

5. 화자 분리 전사를 추가합니다: 오디오에 pyannote 화자 분리를 실행하고 화자별 전사를 임베딩합니다. "Alice가 X에 대해 뭐라고 말했나요?" 쿼리를 시연합니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 장면 분할 | "샷 감지" | 샷 경계에서 비디오를 장면으로 자르기 |
| 다중 벡터 인덱스 | "캡션 + 프레임 + 전사문" | 표현별 명명된 벡터를 가진 Qdrant 컬렉션 |
| 시간적 접지 | "정확히 언제 발생했는가" | 쿼리 답변에 대한 (시작, 종료) 윈도우를 정밀화 |
| 프레임 임베딩 | "시각적 표현" | 키프레임의 벡터 임베딩; 장면-시각 유사성 계산에 사용 |
| RRF 융합 | "상호 랭킹 융합 (RRF)(Reciprocal Rank Fusion (RRF))" | 여러 랭킹된 목록을 병합하는 전략; 하이브리드 검색의 고전적인 기법 |
| 개수 환각 | "개수 오인식" | "X가 몇 개인가" 질문에서 VLM의 알려진 실패 모드 |
| ActivityNet-QA | "비디오-QA 벤치마크" | 롱폼 비디오 QA 정확도 벤치마크 |

## 추가 읽기

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 오픈 VLM 체크포인트
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 대규모 시간적 접지
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 호스팅된 참조
- [VideoDB](https://videodb.io) — 비디오 CRUD API 참조
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 상용 참조
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 장면 분할 모델
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 고전적인 오픈 대안
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 참조 평가 벤치마크
