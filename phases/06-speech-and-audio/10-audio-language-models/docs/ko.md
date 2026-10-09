# 오디오-언어 모델 — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> 2026년 오디오-언어 모델은 음성 + 환경 소리 + 음악에 대해 추론합니다. Qwen2.5-Omni-7B는 MMAU-Pro에서 GPT-4o Audio와 동일한 성능을 보입니다. Audio Flamingo Next는 LongAudioBench에서 Gemini 2.5 Pro를 능가합니다. 오픈 모델과 클로즈드 모델 간의 격차는 사실상 해소되었습니다 — 다만 멀티 오디오 작업에서는 모든 모델이 랜덤에 가까운 성능을 보인다는 예외가 있습니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 6단계 · 04강 (ASR), 12단계 · 03강 (비전-언어 모델), 7단계 · 10강 (오디오 트랜스포머)
**시간:** 약 45분

## 문제점

5초의 오디오가 있습니다: 개가 짖고, 누군가 "stop!"이라고 외치고, 그 후 침묵이 이어집니다. 유용한 질문은 여러 축에 걸쳐 있습니다:

- **전사.** "무엇이 말해졌는가?" — ASR 영역입니다.
- **의미론적 추론.** "사람이 위험한가?" — 짖는 소리 + 외침 + 침묵의 종합적인 이해가 필요합니다.
- **음악 추론.** "멜로디는 어떤 악기가 연주하는가?"
- **긴 오디오 검색.** "이 90분 강의에서 강사가 경사 하강법을 설명한 부분은 어디인가?"

이 모든 질문에 하나의 프롬프트로 답하는 단일 모델은 **오디오-언어 모델**(LALM / ALM)입니다. 순수 ASR과 구별됩니다: LALM은 전사뿐만 아니라 자유 형식의 자연어 답변을 생성합니다.

## 개념

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 세 가지 구성 요소 템플릿

2026년 모든 LALM은 동일한 골격을 가집니다:

1. **오디오 인코더.** Whisper 인코더 · BEATs · CLAP · WavLM · 또는 모델별 커스텀 인코더.
2. **프로젝터.** 오디오 인코더의 특징(features)을 LLM의 토큰 임베딩 공간으로 연결하는 선형 또는 MLP.
3. **LLM.** Llama / Qwen / Gemma 기반 디코더. 인터리브된(interleaved) 텍스트 + 오디오 토큰을 입력받아 텍스트를 생성합니다.

학습:

- **1단계.** 인코더 + LLM을 동결(freeze)하고, ASR / 캡셔닝 데이터로 프로젝터만 학습합니다.
- **2단계.** 지시문 따르기(audio QA, 추론, 음악 이해) 오디오 작업에 대해 전체 / LoRA 미세 조정(fine-tune)을 수행합니다.
- **3단계 (선택).** 음성 입력 / 음성 출력은 음성 디코더를 추가합니다. Qwen2.5-Omni와 AF3-Chat이 이를 수행합니다.

### 2026년 모델 지도

| 모델 | 백본 | 오디오 인코더 | 출력 모달리티 | 접근성 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | 텍스트 + 음성 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | 텍스트 + 음성 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 텍스트 | NVIDIA 비영리 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 텍스트 | NVIDIA 비영리 |
| SALMONN | Vicuna | Whisper + BEATs | 텍스트 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 텍스트 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 텍스트 | Apache-2.0 |
| Gemini 2.5 Flash/Pro (비공개) | Gemini | 독점 | 텍스트 + 음성 | API |
| GPT-4o Audio (비공개) | GPT-4o | 독점 | 텍스트 + 음성 | API |

### 벤치마크 현실 점검 (2026)

**MMAU-Pro.** 음성 / 소리 / 음악 / 혼합을 다루는 1800개 QA 쌍. 다중 오디오 하위 세트 포함.

| 모델 | 종합 | 음성 | 소리 | 음악 | 다중 오디오 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench에서 SOTA | — | — | — | — |

**다중 오디오 열은 모든 모델에 치명적입니다.** 4지선다 객관식에서 무작위 확률은 25%이며, 대부분의 모델이 이 수준에 머무릅니다. LALM은 여전히 두 클립을 비교하는 데 어려움을 겪습니다.

### 2026년 LALM이 유용한 영역

- **콜센터 녹음의 컴플라이언스 감사.** "에이전트가 필수 고지를 언급했는가?"
- **접근성.** 청각 장애 사용자에게 소리 이벤트를 설명 (단순 전사뿐만 아니라).
- **콘텐츠 Moderation.** 폭력적 언어 + 위협적 톤 + 배경 컨텍스트 감지.
- **팟캐스트 / 회의 챕터링.** 화자 순서뿐만 아니라 시맨틱 요약.
- **음악 카탈로그 분석.** "B 섹션의 키 변경이 있는 모든 트랙을 찾아라."

### 아직 유용하지 않은 영역

- 세밀한 음악 이론 (화음 수준 이하).
- 긴 대화에 대한 화자 식별 추론 (10분 이상에서 성능이 저하됨).
- 다중 오디오 비교 (22-26%는 랜덤 수준과 거의 같음).
- 실시간 스트리밍 추론 (대부분 오프라인 배치 추론임).

```figure
v4-alm-tokens
```

## 구현하기

### 1단계: Qwen2.5-Omni 질의

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

### 2단계: 프로젝터 패턴

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

이것이 전부입니다. 프로젝터는 보통 1-3개의 선형 레이어로 구성됩니다. ASR 쌍 (오디오 → 전사)으로 이를 학습하는 것이 Stage-1 사전 학습 작업입니다.

### 3단계: MMAU / LongAudioBench 벤치마킹

```python
from datasets import load_dataset
mmau = load_dataset("gamma-lab-umd/MMAU-Pro", split="test")
mcq = mmau.filter(lambda item: len(item["choices"] or []) > 1)

correct = 0
for item in mcq:
    answer = call_model(item["audio_path"], item["question"], item["choices"])
    if answer == item["answer"]:
        correct += 1
print(f"Accuracy: {correct / len(mcq):.3f}")
```

`audio_path` 데이터를 데이터셋 저장소의 `data.zip` (약 47 GB)에 넣습니다. 점수 계산 전에 다운로드하고 압축을 해제하세요. 이 정확 일치 루프는 샌티티 체크(sanity check)일 뿐, 벤치마킹 스코어가 아니므로 그 수치는 공개된 MMAU-Pro 결과와 비교할 수 없습니다. 공식 평가기는 임베딩 유사성(NV-Embed-v2)으로 객관식 답을 매칭하고, LLM 판정자로 개방형 답을 채점하며, 정규식 규칙으로 지시문 따르기 답을 확인합니다: 예측값을 `model_output` 열에 기록하고 [MMAU-Pro repo](https://github.com/sonalkum/MMAUPro)에서 `evaluate_mmau_pro_comprehensive.py`을 실행하세요. 각 `category` (음성, 소리, 음악, 다중 및 기타)를 개별적으로 보고하세요. 집계된 수치는 모델이 실패하는 부분을 숨깁니다.

## 사용하기

| 작업 | 2026년 선택 |
|------|-----------|
| 자유 형식 오디오 QA (개방형) | Qwen2.5-Omni-7B |
| 긴 오디오에 대한 최선의 오픈 모델 | Audio Flamingo Next |
| 최선의 클로즈드 모델 | Gemini 2.5 Pro |
| 음성 입력 / 음성 출력 에이전트 | Qwen2.5-Omni 또는 GPT-4o Audio |
| 음악 추론 | Audio Flamingo 3 또는 2 (음악 특화 AF-CLAP) |
| 콜센터 감사 | API를 통한 Gemini 2.5 Pro, 정책 문서에 대한 RAG 사용 |

## 문제점

- **다중 오디오에 대한 과도한 신뢰.** "어떤 클립에 X가 있는가"가 필요한 작업이라면, 랜덤 수준의 성능이 실제입니다.
- **긴 오디오 성능 저하.** 10분 이상에서 대부분의 모델의 화자 식별이 깨집니다. 먼저 화자 분리(Diarize)를 수행하고(6강), 그 후 요약하세요.
- **침묵에서의 환각.** Whisper 인코더를 사용하는 LALM이 상속받은 동일한 Whisper 스타일 문제입니다. VAD 게이트를 사용하세요.
- **벤치마크 선별.** 벤더 블로그는 최상의 카테고리만 강조합니다. MMAU-Pro 멀티 오디오 하위 집합을 직접 실행해 보세요.

## 출시하기

`outputs/skill-alm-picker.md`로 저장하세요. 주어진 오디오 이해 작업에 대해 LALM + 벤치마크 하위 집합 + 출력 모달리티(텍스트 vs 음성)를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하여 장난감 프로젝터 패턴과 (오디오 임베딩, 텍스트 토큰) → 출력 토큰의 가짜 LALM 라우팅을 확인해 보세요.
2. **중간.** Qwen2.5-Omni-7B를 100개의 MMAU-Pro 음성 항목으로 채점하세요. 논문에서 보고한 수치와 비교해 보세요.
3. **어려움.** 최소한의 오디오 캡셔닝 기준선을 구축하세요: BEATs 인코더 + 2층 프로젝터 + 동결된 Llama-3.2-1B. AudioCaps에서 프로젝터만 미세 조정하세요. Clotho-AQA에서 SALMONN과 비교해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| LALM | 오디오 ChatGPT | 오디오 인코더 + 프로젝터 + LLM 디코더. |
| 프로젝터 | 어댑터 | 오디오 특징을 LLM 임베딩 공간으로 매핑하는 작은 MLP. |
| MMAU | 벤치마크 | 음성, 소리, 음악에 걸친 10k 오디오-QA 쌍. |
| MMAU-Pro | 더 어려운 MMAU | 1800개의 멀티 오디오 / 추론 중심 질문. |
| LongAudioBench | 장문 평가 | 시맨틱 쿼리가 포함된 다중 분 클립. |
| 음성 입력 / 음성 출력 | 음성 네이티브 | 모델이 음성을 입력받아 텍스트 우회 없이 음성을 출력합니다. |

## 추가 읽기

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 참조 아키텍처.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 음성 입력 음성 출력.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 오픈 롱 오디오 리더.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 이중 인코더 선구자.
- [MMAU-Pro leaderboard](https://sonalkum.github.io/mmau-pro/) — 라이브 2026 순위.
