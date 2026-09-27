# Whisper — 아키텍처 및 미세 조정 (Architecture & Fine-Tuning)

> Whisper는 68만 시간의 다국어 약지도 학습(weakly-supervised) 오디오-텍스트 쌍으로 학습된 30초 윈도우 트랜스포머 인코더-디코더(transformer encoder-decoder) 모델입니다. 하나의 아키텍처로 다양한 작업을 수행하며, 99개 언어에 걸쳐 강력한 성능을 발휘합니다. 2026년 기준 표준 ASR 모델입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 문제점 (The Problem)

2022년 9월 OpenAI가 출시한 Whisper는 범용 상품(commodity)으로 출시된 최초의 ASR(자동 음성 인식) 모델입니다. 오디오를 입력하기만 하면 99개 언어의 텍스트를 얻을 수 있으며, 소음에 강하고 노트북에서도 실행 가능합니다. 2024년까지 OpenAI는 `Large-v3` 및 `Turbo` 변형 모델을 출시했으며, 2026년에 이르러 Whisper는 팟캐스트 전사부터 음성 비서, YouTube 자막에 이르기까지 모든 분야의 기본 베이스라인(default baseline)이 되었습니다.

하지만 Whisper를 단순히 블랙박스(black box)처럼 취급하는 파이프라인으로만 여겨서는 안 됩니다. 도메인 변화(Domain shift) — 즉, 전문 용어, 화자의 억양, 고유 명사, 짧은 클립, 무음 구간 — 는 모델의 성능을 저하시킵니다. 여러분은 다음 사항을 반드시 알아야 합니다:

1. 모델의 내부 구조가 실제로 어떻게 구성되어 있는가.
2. 청크(chunked), 스트리밍(streaming) 또는 긴 형식(long-form)의 오디오를 어떻게 올바르게 입력하는가.
3. 언제 미세 조정(fine-tune)을 해야 하며, 그 방법은 무엇인가.

## 개념 (The Concept)

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**아키텍처 (Architecture).** 표준 트랜스포머 인코더-디코더(transformer encoder-decoder) 구조입니다.

- 입력: 30초 분량의 log-mel spectrogram, 80 mels, 10 ms hop → 3000 프레임. 더 짧은 클립은 제로 패딩(zero-padded) 처리하고, 더 긴 클립은 청크(chunk) 단위로 나눕니다.
- 인코더: conv-downsample (stride 2) + `N`개의 트랜스포머 블록. Large-v3의 경우: 32개 레이어, 1280차원, 20개 헤드.
- 디코더: 인과적 셀프 어텐션(causal self-attn)과 인코더 출력에 대한 크로스 어텐션(cross-attn)을 포함한 `N`개의 트랜스포머 블록. 인코더와 동일한 크기입니다.
- 출력: 51,865개의 토큰 어휘 사전(vocab)을 기반으로 한 BPE 토큰.

Large-v3는 15.5억(1.55B) 개의 파라미터를 가집니다. Turbo 모델은 (32개 중) 4개의 디코더 레이어만 사용하여, WER(Word Error Rate) 손실을 1% 미만으로 유지하면서 지연 시간(latency)을 8배 단축했습니다.

**프롬프트 형식 (The prompt format).** Whisper는 디코더 프롬프트 내의 특수 토큰에 의해 제어되는 멀티태스크(multitask) 모델입니다.

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 언어 태그; 번역(translation)과 전사(transcription) 동작을 결정합니다.
- `<|transcribe|>` 또는 `<|translate|>` — 임의의 언어 입력으로부터 영어 출력을 번역하거나, 있는 그대로 전사합니다.
- `<|notimestamps|>` — 단어 단위 타임스탬프를 생략합니다 (더 빠름).

프롬프트는 하나의 모델이 여러 작업을 수행할 수 있게 해주는 핵심 요소입니다. `<|en|>`을 `<|fr|>`로 바꾸면 프랑스어를 전사합니다.

**30초 윈도우 (30-second window).** 모든 작업은 30초 단위로 고정됩니다. 더 긴 클립은 청킹(chunking)이 필요하며, 더 짧은 클립은 패딩(padding)됩니다. 윈도우는 기본적으로 스트리밍 방식이 아닙니다. 이것이 WhisperX, Whisper-Streaming, 그리고 faster-whisper가 존재하는 이유입니다.

**Log-mel 정규화 (Log-mel normalization).** `(log_mel - mean) / std` 공식을 사용하며, 이때 통계값은 Whisper 자체의 학습 코퍼스에서 가져옵니다. `librosa.feature.melspectrogram`이 아닌 반드시 Whisper의 전처리 방식(`whisper.audio.log_mel_spectrogram`)을 사용해야 합니다.

### 2026년 변형 모델(Variants)

| 변형 모델(Variant) | 파라미터(Params) | 지연 시간(Latency, A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× 실시간(realtime) | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 스트리밍(streaming) | 2.0% |

### 미세 조정 (Fine-tuning)

2026년의 표준 워크플로우(Canonical workflow):

1. 정렬된 전사 데이터(aligned transcripts)가 포함된 10~100시간 분량의 대상 도메인 오디오를 수집합니다.
2. `generate_with_loss` 콜백을 사용하여 `transformers.Seq2SeqTrainer`를 실행합니다.
3. 매개변수 효율적 방식(Parameter-efficient): 어텐션 레이어의 `q_proj`, `k_proj`, `v_proj`에 LoRA를 적용하면 WER 비용을 0.3 미만으로 유지하면서 GPU 메모리를 4배 절감할 수 있습니다.
4. 데이터가 10시간 미만인 경우 인코더(encoder)를 동결(freeze)하세요. 디코더(decoder)만 튜닝합니다.
5. Whisper 고유의 토크나이저(tokenizer)와 프롬프트 형식을 사용하세요. 토크나이저를 교체해서는 안 됩니다.

커뮤니티 결과: 20시간의 의료 받아쓰기 데이터를 사용하여 Medium 모델을 미세 조정하면 의료 어휘에 대한 WER이 12%에서 4.5%로 감소합니다. 4시간의 아이슬란드어 데이터를 사용하여 Turbo 모델을 미세 조정하면 WER이 18%에서 6%로 감소합니다.

```figure
sp-asr-attention
```

## 직접 구현해 보기 (Build It)

### 1단계: Whisper를 즉시 실행하기 (run Whisper out of the box)

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 무한 반복 방지
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

항상 재정의(override)해야 하는 주요 기본값은 다음과 같습니다: `temperature=0.0` (샘플링 기본값은 0.0 → 0.2 → 0.4 ... 순의 폴백 체인을 따름), `condition_on_previous_text=False` (연쇄적 환각(cascading hallucination) 문제 방지), 그리고 `no_speech_threshold=0.6` (침묵 감지)입니다.

### 2단계: 청크 단위 롱폼 (chunked long-form)

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX는 (1) Silero VAD 게이팅(gating), (2) `wav2vec 2.0`을 통한 단어 단위 정렬(word-level alignment), (3) `pyannote.audio`를 통한 화자 분리(diarization) 기능을 추가합니다. 2026년 프로덕션급 전사(transcription) 작업을 위한 핵심 도구입니다.

### 3단계: LoRA를 이용한 미세 조정(fine-tune)

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> 약 3M 학습 가능 / 총 809M
```

그 다음 표준 `Trainer` 루프를 실행합니다. 1000 스텝마다 체크포인트를 저장하세요. 홀드아웃(held-out) 데이터셋에 대해 WER로 평가합니다.

### 4단계: 각 레이어가 무엇을 학습하는지 조사하기 (inspect what each layer learns)

```python
# 디코더가 어디에 어텐션을 기울이는지 확인하기 위해 디코딩 과정 중 크로스 어텐션(cross-attention) 가중치를 가져옵니다.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: 레이어 × 헤드 × 단계 × 소스_길이(src_len)
```

히트맵(heatmap)으로 시각화해 보세요. 디코더 단계가 인코더 프레임을 스캔함에 따라 대각선 정렬이 나타나는 것을 볼 수 있습니다. 그 대각선이 바로 Whisper가 인식하는 단어 타임스탬프(word timestamps)입니다.

## 활용하기 (Use It)

2026년 스택:

| 상황 | 선택 (Pick) |
|-----------|------|
| 일반적인 영어, 오프라인 | `whisperx`를 통한 Large-v3-turbo |
| 모바일 / 에지 (Edge) | 양자화된 Whisper-Tiny (int8) 또는 Moonshine |
| 다국어 장문 (Long-form) | `whisperx` + 화자 분리(diarization)를 통한 Large-v3 |
| 저자원 언어 (Low-resource) | LoRA를 이용한 Medium 또는 Turbo 미세 조정(Fine-tune) |
| 스트리밍 (2초 지연 시간) | Whisper-Streaming 또는 Parakeet-TDT |
| 단어 단위 타임스탬프 | WhisperX (wav2vec 2.0을 통한 강제 정렬) |

`faster-whisper` (CTranslate2 백엔드)는 2026년 기준 가장 빠른 CPU+GPU 추론 런타임입니다. 동일한 결과물을 출력하면서도 기본(vanilla) 모델보다 4배 더 빠릅니다.

## 2026년에도 여전히 발생하는 실수들 (Pitfalls that still ship in 2026)

- **침묵 구간에서의 환각 텍스트(Hallucinated text on silence).** 캡션 데이터로 학습된 Whisper는 "Thanks for watching!", "Subscribe!", 노래 가사 등을 생성할 수 있습니다. 호출 전 항상 VAD(Voice Activity Detection) 게이트를 적용하세요.
- **`condition_on_previous_text` 연쇄 효과(cascade).** 하나의 환각이 이후의 윈도우(window)들을 오염시킵니다. 청크(chunk) 간의 유창성이 반드시 필요한 경우가 아니라면 `False`로 설정하세요.
- **짧은 클립 패딩(Short-clip padding).** 2초짜리 클립을 30초로 패딩하면, 뒤따르는 침묵 구간에서 환각이 발생할 수 있습니다. `pad=False`를 사용하거나 VAD 게이트를 적용하세요.
- **잘못된 멜 스펙트로그램 통계(Wrong mel stats).** Whisper의 방식 대신 `librosa`의 mels를 사용하면 거의 무작위(random)에 가까운 출력이 생성됩니다. `whisper.audio.log_mel_spectrogram`을 사용하세요.

## Ship It (실행해 보세요)

`outputs/skill-whisper-tuner.md`로 저장하세요. 주어진 도메인을 위한 Whisper 미세 조정(fine-tune) 또는 추론(inference) 파이프라인을 설계해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. Whisper 스타일의 프롬프트를 토큰화하고, 디코딩된 형태의 예산(decoded shape budgets)을 계산하며, 10분 분량의 클립에 대한 청크 스케줄(chunk schedule)을 출력합니다.
2. **중간 (Medium).** `faster-whisper`를 설치하고, 10분 길이의 팟캐스트를 전사(transcribe)한 뒤, 인간이 작성한 전사본과 WER(Word Error Rate)을 비교해 보세요. `language="auto"` 설정과 강제 지정된 `language="en"` 설정을 비교해 보세요.
3. **어려움 (Hard).** HF `datasets`를 사용하여 Whisper가 어려움을 겪는 언어(예: 우르두어)를 선택하세요. 2시간 분량의 데이터로 Medium 모델을 LoRA를 사용하여 2 에포크(epochs) 동안 미세 조정(fine-tuning)하고, WER 변화량(delta)을 보고하세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 의미 | 실제 의미 |
|------|-----------------|-----------------------|
| 30-sec window | Whisper의 제한 사항 | 입력 용량의 하드 캡(Hard cap); 긴 오디오는 청크(chunk) 단위로 나누어야 함. |
| SOT | 전사 시작 (Start-of-transcript) | `<\|startoftranscript\|>`가 디코더 프롬프트를 시작함. |
| Timestamps token | 시간적 정렬 (Temporal alignment) | 0.02초 간격의 모든 오프셋은 51k 어휘 사전(vocab) 내의 특수 토큰임. |
| Turbo | 빠른 변형 모델 | 4개의 디코더 레이어 사용, 8배 빠름, WER(단어 오류율) 저하는 1% 미만. |
| WhisperX | 롱폼 래퍼 (Long-form wrapper) | VAD + Whisper + wav2vec 정렬 + 화자 분리(diarization) 결합. |
| LoRA fine-tune | 효율적인 튜닝 | 어텐션(attention)에 저차원 어댑터를 추가; 파라미터의 약 0.3%만 학습. |
| Hallucination | 침묵의 실패 (The silent failure) | Whisper가 노이즈나 무음 구간에서 유창한 영어를 생성해내는 현상. |

## 추가 읽을거리 (Further Reading)

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 오리지널 아키텍처 및 학습 레시피.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4개 레이어 디코더, 8배 속도 향상.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 긴 형식(long-form), 단어 단위 정렬(word-aligned), 화자 분리(diarized).
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 기반, 4배 더 빠름.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 표준적인 LoRA / 전체 미세 조정(full-FT) 가이드.
