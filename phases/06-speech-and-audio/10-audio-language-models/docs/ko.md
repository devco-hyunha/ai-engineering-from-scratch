# 오디오-언어 모델 (Audio-Language Models) — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> 2026년의 오디오-언어 모델은 음성, 환경음, 음악을 바탕으로 추론합니다. Qwen2.5-Omni-7B는 MMAU-Pro에서 GPT-4o Audio와 대등한 성능을 보여줍니다. Audio Flamingo Next는 LongAudioBench에서 Gemini 2.5 Pro를 능가합니다. 멀티 오디오(multi-audio) 작업(현재 모든 모델이 무작위 수준에 머물러 있는 분야)을 제외하면, 오픈 소스와 폐쇄형 모델 사이의 격차는 사실상 사라졌습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## 문제 (The Problem)

5초 분량의 오디오가 있습니다: 개가 짖고, 누군가 "멈춰!"라고 외친 뒤, 정적이 흐릅니다. 유용한 질문들은 다음과 같이 여러 축에 걸쳐 있습니다:

- **전사 (Transcription).** "무엇이라고 말했는가?" — ASR(자동 음성 인식)의 영역입니다.
- **의미론적 추론 (Semantic reasoning).** "그 사람은 위험한 상황인가?" — 개 짖는 소리, 외침, 정적을 통합적으로 이해해야 합니다.
- **음악적 추론 (Music reasoning).** "멜로디를 연주하는 악기는 무엇인가?"
- **장기 오디오 검색 (Long-audio retrieval).** "이 90분짜리 강의 중 강사가 경사 하강법(gradient descent)을 설명한 부분은 어디인가?"

이 모든 질문에 단일 프롬프트로 답할 수 있는 모델을 **오디오-언어 모델 (audio-language model, LALM / ALM)**이라고 합니다. 순수 ASR과는 별개로, LALM은 단순한 전사 결과가 아닌 자유 형식의 자연어 답변을 생성합니다.

## 개념 (The Concept)

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 3요소 템플릿 (The three-component template)

모든 2026년형 LALM은 동일한 골격을 가집니다:

1. **오디오 인코더 (Audio encoder).** Whisper encoder, BEATs, CLAP, WavLM 또는 모델별 커스텀 인코더.
2. **프로젝터 (Projector).** 오디오 인코더의 특징(features)을 LLM의 토큰 임베딩 공간으로 연결하는 `Linear` 또는 `MLP`.
3. **LLM.** Llama, Qwen, Gemma 기반 디코더. 텍스트와 오디오 토큰이 교차된(interleaved) 입력을 받아 텍스트를 생성합니다.

학습 과정 (Training):

- **1단계 (Stage 1).** 인코더와 LLM을 동결(freeze)하고, ASR / 캡셔닝(captioning) 데이터로 프로젝터만 학습합니다.
- **2단계 (Stage 2).** 지시 이행(instruction-following) 오디오 작업(QA, 추론, 음악 이해)에 대해 전체 파라미터 또는 LoRA 미세 조정(fine-tune)을 수행합니다.
- **3단계 (Stage 3, 선택 사항).** 음성 입력/출력(Voice-in / voice-out)을 위해 음성 디코더를 추가합니다. Qwen2.5-Omni와 AF3-Chat이 이 방식을 사용합니다.

### 2026년 모델 지도 (The 2026 model map)

| 모델 (Model) | 백본 (Backbone) | 오디오 인코더 (Audio encoder) | 출력 모달리티 (Output modality) | 액세스 (Access) |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### 벤치마크 현실 점검 (Benchmark reality check, 2026)

**MMAU-Pro.** 음성, 소리, 음악, 혼합 오디오를 아우르는 1,800개의 QA 쌍으로 구성되었습니다. 멀티 오디오(Multi-audio) 서브셋이 포함되어 있습니다.

| 모델 | 전체 (Overall) | 음성 (Speech) | 소리 (Sound) | 음악 (Music) | 멀티 오디오 (Multi-audio) |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA 달성 | — | — | — | — |

**멀티 오디오(multi-audio) 열은 모든 모델에게 치명적입니다.** 4지 선다형 문제에서 무작위로 찍었을 때의 확률은 25%이며, 대부분의 모델이 그 근처의 점수를 기록하고 있습니다. LALM은 여전히 두 개의 클립을 비교하는 데 어려움을 겪고 있습니다.

### 2026년 LALM(Large Audio Language Models)의 활용 분야

- **콜센터 녹취록의 컴플라이언스 감사(Compliance audit).** "상담원이 필수 고지 사항을 언급했는가?"
- **접근성(Accessibility).** 청각 장애 사용자에게 소리 이벤트를 설명(단순 전사를 넘어선 맥락적 설명).
- **콘텐츠 모더레이션(Content moderation).** 폭력적인 언어, 위협적인 어조, 배경 맥락 탐지.
- **팟캐스트 / 회의 챕터 구분(Chaptering).** 단순 화자 전환이 아닌 의미론적 요약 제공.
- **음악 카탈로그 분석.** "B-섹션에서 키 변화(key change)가 있는 모든 트랙을 찾아줘."

### 아직 유용하지 않은 분야 (Where they are NOT (yet) useful)

- 미세한 음악 이론 (코드 레벨 미만).
- 긴 대화에서의 화자 식별 추론 (10분이 지나면 성능 저하).
- 다중 오디오 비교 (22-26%로 무작위 선택보다 약간 높은 수준).
- 실시간 스트리밍 추론 (대부분 오프라인 배치 추론 방식).

```figure
v4-alm-tokens
```

## 직접 구현해 보기 (Build It)

### 1단계: Qwen2.5-Omni 쿼리하기 (query Qwen2.5-Omni)

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 2단계: 프로젝터 패턴 (the projector pattern)

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

이것이 전부입니다. 프로젝터는 보통 1~3개의 선형 레이어(`linear layers`)로 구성됩니다. ASR 쌍(오디오 → 전사 데이터)을 사용하여 이를 학습시키는 것이 1단계 사전 학습 작업(Stage-1 pretext task)입니다.

### 3단계: MMAU / LongAudioBench 벤치마킹(benchmarking)

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

카테고리별(음성(speech) / 소리(sound) / 음악(music) / 다중 오디오(multi-audio))로 각각 보고하세요. 전체 수치만 합산하면 모델이 구체적으로 어디에서 실패하는지 파악하기 어렵습니다.

## 사용 방법 (Use It)

| 작업 (Task) | 2026년 추천 (2026 pick) |
|------|-----------|
| 자유 형식 오디오 질의응답 (Free-form audio QA, 오픈 소스) | Qwen2.5-Omni-7B |
| 긴 오디오 처리 최적 오픈 소스 (Best open on long audio) | Audio Flamingo Next |
| 최적의 폐쇄형 모델 (Best closed) | Gemini 2.5 Pro |
| 음성 입출력 에이전트 (Voice-in / voice-out agent) | Qwen2.5-Omni 또는 GPT-4o Audio |
| 음악 추론 (Music reasoning) | Audio Flamingo 3 또는 2 (음악 특화 AF-CLAP) |
| 콜센터 감사 (Call-center audit) | API를 통한 Gemini 2.5 Pro, 정책 문서 기반 RAG 활용 |

## 주의 사항 (Pitfalls)

- **멀티 오디오에 대한 과도한 신뢰 (Over-trust on multi-audio).** 만약 작업 목표가 "어느 클립에 X가 포함되어 있는가"를 찾는 것이라면, 성능이 무작위 추측(random-chance) 수준에 머물 수 있습니다.
- **장기 오디오 성능 저하 (Long-audio degradation).** 10분이 넘어가면 대부분의 모델에서 화자 할당(speaker attribution) 기능이 저하됩니다. 먼저 화자 분리(Diarize, 레슨 6 참고)를 수행한 후 요약하세요.
- **무음 구간에서의 환각 (Hallucinations on silence).** Whisper 인코더를 사용하는 LALM에서 발생하는 고질적인 문제입니다. VAD(Voice Activity Detection) 게이트를 사용하세요.
- **벤치마크 체리피킹 (Benchmark cherry-picking).** 제조사의 블로그 포스트는 대개 최상의 결과가 나오는 카테고리만을 강조합니다. MMAU-Pro의 멀티 오디오 서브셋을 직접 실행하여 검증해 보세요.

## Ship It (실행하기)

`outputs/skill-alm-picker.md`로 저장하세요. 주어진 오디오 이해(audio-understanding) 작업에 대해 LALM + 벤치마크 서브셋 + 출력 모달리티(텍스트 vs 음성)를 선택합니다.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행하여 간단한 프로젝터 패턴(toy projector pattern)과 (오디오 임베딩, 텍스트 토큰)이 출력 토큰으로 전달되는 가짜 LALM 라우팅(fake LALM routing) 과정을 확인해 보세요.
2. **중간 (Medium).** 100개의 MMAU-Pro 음성 항목에 대해 Qwen2.5-Omni-7B의 점수를 측정해 보세요. 논문에 보고된 수치와 비교해 보세요.
3. **어려움 (Hard).** 최소한의 오디오 캡셔닝(audio-captioning) 베이스라인을 구축해 보세요: BEATs 인코더 + 2계층 프로젝터(2-layer projector) + 동결된(frozen) Llama-3.2-1B를 사용합니다. AudioCaps 데이터셋에서 프로젝터만 미세 조정(fine-tune)해 보세요. Clotho-AQA 데이터셋에서 SALMONN과 성능을 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| LALM | 오디오 ChatGPT | 오디오 인코더(Audio encoder) + 프로젝터(projector) + LLM 디코더(LLM decoder). |
| Projector | 어댑터(Adapter) | 오디오 특징을 LLM 임베딩 공간으로 매핑하는 작은 MLP. |
| MMAU | 벤치마크 | 음성, 소리, 음악을 아우르는 1만 개의 오디오-QA 쌍. |
| MMAU-Pro | 고난도 MMAU | 1,800개의 멀티 오디오 / 추론 중심 질문. |
| LongAudioBench | 장기 오디오 평가 | 의미론적 질의가 포함된 수 분 길이의 클립. |
| Voice-in / voice-out | 음성 네이티브(Speech-native) | 텍스트를 거치지 않고 모델이 음성을 입력받아 음성을 출력함. |

## 추가 읽을거리 (Further Reading)

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 참조 아키텍처(reference architecture).
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 음성 입력-음성 출력(speech-in-speech-out) 모델.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 오픈 소스 장기 오디오(long-audio) 분야의 선두주자.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA(최고 성능).
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 이중 인코더(dual-encoder)의 선구자.
- [MMAU-Pro 리더보드(leaderboard)](https://mmaubenchmark.github.io/) — 2026년 실시간 순위.
