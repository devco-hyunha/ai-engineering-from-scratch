# Whisper — 아키텍처 및 미세 조정

> Whisper는 30초 윈도우를 사용하는 트랜스포머 인코더-디코더로, 68만 시간의 다국어 약한 지도 학습 오디오-텍스트 쌍으로 학습되었습니다. 하나의 아키텍처로 여러 작업을 수행하며, 99개 언어에 걸쳐 견고합니다. 2026년 기준의 참고 ASR입니다.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 04강 (ASR), 5단계 · 10강 (어텐션), 7단계 · 05강 (Full Transformer)
**시간:** 약 75분

## 문제점

OpenAI가 2022년 9월에 공개한 Whisper는 상품(commodity)으로서 출시된 최초의 ASR 모델이었습니다. 오디오를 붙여넣으면 텍스트가 나오고, 99개 언어를 지원하며, 잡음에 강하고, 노트북에서 실행됩니다. 2024년까지 OpenAI는 Large-v3 및 Turbo 변형을 출시했으며, 2026년 현재 Whisper는 팟캐스트 전사부터 음성 비서, YouTube 자막까지 모든 것의 기본 기준선입니다.

하지만 Whisper는 영원히 블랙 박스처럼 취급할 수 있는 파이프라인이 아닙니다. 도메인 이동(domain shift)은 이를 무너뜨립니다 — 전문 용어, 화자 억양, 고유 명사, 짧은 클립, 침묵. 다음을 알아야 합니다:

1. 내부적으로 실제로 무엇인지.
2. 청킹(chunked), 스트리밍, 또는 롱폼(long-form) 오디오를 올바르게 제공하는 방법.
3. 미세 조정(fine-tuning)을 해야 하는 시점과 방법.

## 개념

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**아키텍처.** 표준 트랜스포머 인코더-디코더입니다.

- 입력: 30초 로그-멜 스펙트로그램, 80 mel, 10 ms hop → 3000 프레임. 더 짧은 클립은 0으로 패딩하고, 더 긴 클립은 청킹합니다.
- 인코더: conv-downsample (stride 2) + `N` 트랜스포머 블록. Large-v3의 경우: 32 레이어, 1280 차원, 20 헤드입니다.
- 디코더: `N` 트랜스포머 블록으로 인코더 출력에 대한 인과적 셀프 어텐션(causal self-attn) + 교차 어텐션(cross-attn)을 사용합니다. 크기는 인코더와 동일합니다.
- 출력: 51,865 토큰 어휘에 대한 BPE 토큰입니다.

Large-v3는 15.5억 매개변수를 가집니다. Turbo는 디코더를 4 레이어로 줄여 (32에서) 지연 시간을 8배 단축하며 WER 손실은 1% 미만입니다.

**프롬프트 형식.** Whisper는 디코더 프롬프트의 특수 토큰으로 제어되는 멀티태스크 모델입니다:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 언어 태그; 번역 대 전사(transcription) 동작을 강제합니다.
- `<|transcribe|>` 또는 `<|translate|>` — 모든 언어 입력에서 영어 출력으로 번역하거나, 원문 그대로(transcription) 처리합니다.
- `<|notimestamps|>` — 단어 단위 타임스탬프를 건너뛰세요 (더 빠릅니다).

프롬프트는 하나의 모델이 여러 작업을 수행할 수 있게 해줍니다. `<|en|>`를 `<|fr|>`로 변경하면 프랑스어를 전사합니다.

**30초 윈도우.** 모든 것이 30초에 고정되어 있습니다. 더 긴 클립은 청킹이 필요하며, 더 짧은 클립은 패딩 처리됩니다. 윈도우는 네이티브로 스트리밍되지 않습니다 — 이것이 WhisperX, Whisper-Streaming, faster-whisper가 존재하는 이유입니다.

**로그-멜 정규화.** `(log_mel - mean) / std`이며, 통계는 Whisper 자체의 학습 코퍼스에서 나옵니다. Whisper의 전처리 (`whisper.audio.log_mel_spectrogram`)를 *반드시* 사용해야 하며, `librosa.feature.melspectrogram`를 사용해서는 안 됩니다.

### 2026년 변형

| 변형 | 매개변수 | 지연 시간 (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 실시간 1배 | 5.4% |
| Base | 74M | 1배 | 4.1% |
| Small | 244M | 1배 | 3.0% |
| Medium | 769M | 1배 | 2.7% |
| Large-v3 | 1.55B | 2배 | 1.8% |
| Large-v3-turbo | 809M | 8배 | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 스트리밍 | 2.0% |

### 미세 조정

2026년 표준 워크플로우:

1. 정렬된 전사문과 함께 대상 도메인 오디오 10–100시간을 수집하세요.
2. `transformers.Seq2SeqTrainer`를 `generate_with_loss` 콜백과 함께 실행하세요.
3. 매개변수 효율적: 어텐션 레이어의 `q_proj`, `k_proj`, `v_proj`에 LoRA를 적용하면 GPU 메모리를 4배 줄이면서 WER 비용은 <0.3%로 감소합니다.
4. 10시간 미만인 경우 인코더를 동결하세요. 디코더만 미세 조정하세요.
5. Whisper 자체의 토크나이저와 프롬프트 형식을 사용하세요. 토크나이저를 절대 교체하지 마세요.

커뮤니티 결과: 의료 구술 20시간으로 Medium를 미세 조정하면 의료 어휘에서 WER가 12%에서 4.5%로 감소합니다. 아이슬란드어 4시간으로 Turbo를 미세 조정하면 WER가 18%에서 6%로 감소합니다.

```figure
sp-asr-attention
```

## 구현하기

### 1단계: 기본 상태로 Whisper 실행

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 무한 반복을 방지합니다
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

항상 오버라이드해야 하는 주요 기본값: `temperature=0.0` (샘플링이 0.0 → 0.2 → 0.4 … 폴백 체인으로 기본 설정됨), `condition_on_previous_text=False` (연쇄 환각 문제를 방지), `no_speech_threshold=0.6` (침묵 감지).

### 2단계: 청킹된 장문

```python
# whisperx는 단어 단위 타임스탬프가 포함된 장문 오디오의 2026년 표준입니다
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX는 (1) Silero VAD 게이트, (2) wav2vec 2.0를 통한 단어 수준 정렬, (3) `pyannote.audio`를 통한 화자 분리(diarization)를 추가합니다. 2026년 프로덕션 전사(transcription)의 주력 도구입니다.

### 3단계: LoRA로 미세 조정(Fine-tuning)하기

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

이후 표준 Trainer 루프를 진행합니다. 1000 스텝마다 체크포인트(Checkpoint)를 저장하세요. 홀드아웃(held-out) 데이터에서 WER로 평가해 보세요.

### 4단계: 각 레이어가 무엇을 학습하는지inspect하기

```python
# 디코딩(decoding) 중 교차 어텐션(Cross-Attention) 가중치를 가져와 디코더가 무엇을 어텐션(Attention)하는지 확인해 보세요.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

히트맵(heatmap)으로 시각화해 보세요. 디코더 스텝이 인코더 프레임을 스캔하면서 대각선 정렬을 확인할 수 있습니다. 이 대각선은 Whisper의 단어 타임스탬프 개념입니다.

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 일반 영어, 오프라인 | `whisperx`를 통한 Large-v3-turbo |
| 모바일 / 엣지 | Whisper-Tiny 양자화(int8) 또는 Moonshine |
| 다국어 롱폼(long-form) | `whisperx` + 화자 분리(diarization)를 통한 Large-v3 |
| 저자원 언어 | LoRA로 Medium 또는 Turbo 미세 조정(Fine-tuning) |
| 스트리밍(2초 지연) | Whisper-Streaming 또는 Parakeet-TDT |
| 단어 수준 타임스탬프 | WhisperX (wav2vec 2.0를 통한 강제 정렬) |

`faster-whisper` (CTranslate2 백엔드)는 2026년 가장 빠른 CPU+GPU 추론(Inference) 런타임입니다. 동일한 출력으로 바닐라(vanilla)보다 4배 빠릅니다.

## 2026년에도 여전히 발생하는 함정

- **침묵에서의 환각(Hallucination) 텍스트.** Whisper는 자막으로 학습되어 "Thanks for watching!", "Subscribe!", 가사 등이 포함됩니다. 호출하기 전에 항상 VAD 게이트를 적용하세요.
- **`condition_on_previous_text` 캐스케이드.** 하나의 환각(Hallucination)이 후속 윈도우를 오염시킵니다. 청크(chunk) 간 유창성이 필요하지 않은 경우 `False`를 설정하세요.
- **짧은 클립 패딩.** 2초 클립을 30초로 패딩하면 후미 침묵에서 환각(Hallucination)이 발생할 수 있습니다. `pad=False`를 사용하거나 VAD 게이트를 적용하세요.
- **잘못된 mel 통계.** librosa의 mel을 Whisper의 mel 대신 사용하면 거의 랜덤한 출력이 생성됩니다. `whisper.audio.log_mel_spectrogram`를 사용하세요.

## 출시하기

`outputs/skill-whisper-tuner.md`로 저장하세요. 특정 도메인에 대한 Whisper 미세 조정(Fine-tuning) 또는 추론(Inference) 파이프라인을 설계해 보세요.

## 연습 문제

1. **쉬움.** `code/main.py`을 실행해 보세요. Whisper 스타일 프롬프트를 토큰화하고, 디코딩된 형상 예산을 계산하며, 10분 클립의 청크 스케줄을 출력합니다.
2. **중간.** `faster-whisper`을 설치하고, 10분 팟캐스트를 전사한 후 인간 전사본과 WER을 비교해 보세요. `language="auto"`와 강제 `language="en"`를 비교해 보세요.
3. **어려움.** HF `datasets`를 사용하여 Whisper가 어려워하는 언어(예: 우르두어)를 선택하고, 2시간 데이터로 LoRA를 통해 Medium를 2에포크 동안 미세 조정하며 WER 델타를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 30초 윈도우 | Whisper의 한계 | 하드 입력 상한; 더 긴 오디오는 청크로 분할하세요. |
| SOT | 전사 시작 | `<\|startoftranscript\|>`가 디코더 프롬프트를 시작합니다. |
| 타임스탬프 토큰 | 시간 정렬 | 0.02초 오프셋마다 51k 어휘의 특수 토큰입니다. |
| Turbo | 빠른 변형 | 디코더 레이어 4개, 8배 빠름, WER 감소 <1%. |
| WhisperX | 롱폼 래퍼 | VAD + Whisper + wav2vec 정렬 + 화자 분리. |
| LoRA 미세 조정 | 효율적인 튜닝 | 어텐션에 저랭크 어댑터를 추가; 매개변수의 약 0.3%만 학습합니다. |
| 환각(Hallucination) | 조용한 실패 | Whisper는 잡음/침묵에서 유창한 영어를 생성합니다. |

## 추가 읽기

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 원본 아키텍처 및 학습 레시피.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4레이어 디코더, 8배 속도 향상.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 롱폼, 단어 정렬, 화자 분리 지원.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 기반, 4배 빠름.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 표준 LoRA / 전체 미세 조정 가이드.
