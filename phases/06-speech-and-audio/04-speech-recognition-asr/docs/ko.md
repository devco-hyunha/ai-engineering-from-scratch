# 음성 인식 (ASR) — CTC, RNN-T, Attention

> 음성 인식은 매 타임스텝(timestep)마다 수행되는 오디오 분류이며, 영어와 무음(silence)을 이해하는 시퀀스 모델을 통해 하나의 텍스트로 결합됩니다. CTC, RNN-T, 그리고 Attention은 이를 수행하는 세 가지 방법입니다. 각 방식의 차이를 이해하고 상황에 맞는 모델을 선택해 보세요.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 minutes

## 문제 (The Problem)

10초 길이의 16 kHz 오디오 클립이 있습니다. 목표는 "turn on the kitchen lights"라는 문자열을 얻는 것입니다. 여기서 문제는 구조적인 측면입니다. 오디오 프레임은 문자와 일대일로 일치하지 않습니다. "okay"라는 단어는 200ms가 걸릴 수도 있고 1200ms가 걸릴 수도 있습니다. 발화 사이에 침묵(silence)이 포함되기도 하며, 어떤 음소(phoneme)는 다른 음소보다 더 길기도 합니다. 또한 출력될 토큰의 개수를 미리 알 수 없습니다.

이를 해결하기 위한 세 가지 공식화(formulations)는 다음과 같습니다:

1. **CTC (Connectionist Temporal Classification).** 특수 기호인 *blank*를 포함하여 프레임당 토큰 확률을 방출합니다. 디코딩 시 중복된 토큰과 *blank*를 하나로 합칩니다. 비자기회귀(Non-autoregressive) 방식이며 속도가 빠릅니다. wav2vec 2.0, MMS 등에서 사용됩니다.
2. **RNN-T (Recurrent Neural Network Transducer).** 인코더 프레임과 이전 토큰들이 주어졌을 때, 결합 네트워크(joint network)가 다음 토큰을 예측합니다. 스트리밍(Streamable)이 가능합니다. Google의 온디바이스 ASR, NVIDIA Parakeet 등에서 사용됩니다.
3. **Attention encoder-decoder.** 인코더가 오디오를 은닉 상태(hidden states)로 압축하면, 디코더가 크로스 어텐션(cross-attention)을 통해 토큰을 자기회귀(autoregressively) 방식으로 생성합니다. Whisper, SeamlessM4T 등에서 사용됩니다.

2026년 기준, LibriSpeech test-clean 데이터셋에서의 SOTA WER(Word Error Rate)은 1.4% (Parakeet-TDT-1.1B, NVIDIA) 및 1.58% (Whisper-Large-v3-turbo)입니다. 성능 차이는 미미하지만, 배포(deployment) 측면에서의 차이는 매우 큽니다.

## 개념 (The Concept)

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 직관 (CTC intuition).** 인코더가 `V+1`개 토큰(`V`개 문자 + blank)에 대한 `T`개의 프레임 수준 분포를 출력한다고 가정합니다. 길이가 `U < T`인 타겟 문자열 `y`에 대해, `y`로 축약(collapse)되는 모든 프레임 정렬(alignment)은 유효합니다. CTC 손실(loss)은 이러한 모든 정렬에 대해 합산합니다. 추론 시에는 프레임별로 `argmax`를 수행하고, 반복되는 값을 축약하며, blank를 제거합니다.

장점: 비자기회귀적(non-autoregressive), 스트리밍 가능(streamable), 제로 룩어헤드(zero lookahead). 단점: *조건부 독립 가정(conditional independence assumption)* — 각 프레임 예측이 다른 프레임과 독립적이므로 내부 언어 모델(internal language model)이 존재하지 않습니다. 빔 서치(beam search)나 얕은 융합(shallow fusion)을 통해 외부 LM으로 이를 보완할 수 있습니다.

**RNN-T 직관 (RNN-T intuition).** 토큰 이력을 임베딩하는 *예측기(predictor)* 네트워크와, 예측기 상태를 인코더 프레임과 결합하여 `V+1`(`+1`은 null / no-emit을 의미)에 대한 결합 분포를 만드는 *조이너(joiner)*를 추가합니다. CTC가 무시했던 조건부 의존성을 명시적으로 모델링합니다. 각 단계가 과거의 프레임과 과거의 토큰에만 의존하므로 스트리밍이 가능합니다.

장점: 스트리밍 가능 + 내부 LM 보유. 단점: 학습이 더 복잡하고 메모리 소모가 큼(3D loss lattice); RNN-T 손실 커널은 그 자체로 하나의 라이브러리 카테고리를 형성할 정도입니다.

**어텐션 인코더-디코더 (Attention encoder-decoder).** 로그-멜(log-mel) 프레임에 대한 인코더(6-32개 트랜스포머 레이어)를 사용합니다. 디코더(6-32개 트랜스포머 레이어)는 인코더 출력에 크로스 어텐션(cross-attends)하여 토큰을 자기회귀적(autoregressively)으로 생성합니다. 정렬 제약이 없으므로 어텐션이 오디오의 어느 곳이든 참조할 수 있습니다. 어텐션을 제한하지 않는 한 스트리밍은 불가능합니다(예: chunked Whisper-Streaming, 2024).

장점: 오프라인 ASR에서 가장 높은 품질을 제공하며, 표준 seq2seq 도구로 학습하기 쉽습니다. 단점: 자기회귀적 지연 시간(latency)이 출력 길이에 비례하며, 별도의 엔지니어링 없이는 스트리밍할 수 없습니다.

### WER: 단 하나의 지표 (the one number)

**단어 오류율 (Word Error Rate)** = `(S + D + I) / N`이며, 여기서 S=치환(substitutions), D=삭제(deletions), I=삽입(insertions), N=참조 단어 수(reference word count)를 의미합니다. 이는 단어 수준에서의 레벤슈타인 편집 거리(Levenshtein edit distance)와 일치합니다. 수치가 낮을수록 좋습니다. 일반적으로 WER이 20%를 넘으면 사용이 불가능하며, 5% 미만은 낭독된 음성(read speech)에 대해 인간 수준의 성능(human-parity)에 해당합니다. 표준 벤치마크의 2026년 수치는 다음과 같습니다:

| 모델 (Model) | LibriSpeech test-clean | LibriSpeech test-other | 크기 (Size) |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

이 모델들은 모두 인코더-디코더(encoder-decoder) 또는 RNN-T 기반입니다. 순수 CTC 시스템(`wav2vec 2.0`)은 test-clean에서 약 1.8–2.1% 정도의 수치를 기록합니다.

```figure
ctc-collapse
```

## 직접 구현해 보기 (Build It)

### 1단계: greedy CTC 디코딩 (greedy CTC decode)

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 프레임별 확률 벡터의 리스트
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

두 가지 규칙이 있습니다: 연속된 중복 문자를 하나로 합치고, blank(`_`)를 제거합니다. 예시: `a a _ _ a b b _ c` → `a a b c`.

### 2단계: beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

실제 운영 환경에서는 언어 모델(LM) 융합을 포함한 접두사 트리(prefix tree) 빔 서치를 사용하지만, 위 코드는 개념적인 골격(skeleton)입니다.

### 3단계: WER (Word Error Rate)

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 4단계: Whisper를 이용한 추론 (Inference against Whisper)

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026년 기준 가장 강력한 범용 ASR(자동 음성 인식)을 구현하는 한 줄짜리 코드입니다. 24GB GPU에서 약 20배의 실시간 속도로 실행됩니다.

### 5단계: Parakeet 또는 wav2vec 2.0을 이용한 스트리밍(Streaming)

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

스트리밍 ASR(Automatic Speech Recognition)에는 청크 단위의 인코더 어텐션(encoder attention)과 캐리오버 상태(carryover state)가 필요합니다. 이를 지원하는 라이브러리를 사용하세요 (Parakeet의 경우 NeMo, `transformers` 파이프라인의 경우 `chunk_length_s` 옵션 활용).

## 사용 방법 (Use It)

2026년 스택:

| 상황 (Situation) | 선택 (Pick) |
|-----------|------|
| 영어, 오프라인, 최고 품질 | Whisper-large-v3-turbo |
| 다국어, 견고함 (Robust) | SeamlessM4T v2 |
| 스트리밍, 낮은 지연 시간 | Parakeet-TDT-1.1B 또는 Riva |
| 엣지, 모바일, 500ms 미만 지연 시간 | 양자화된 Whisper-Tiny 또는 Moonshine (2024) |
| 긴 형식 (Long-form) | VAD 기반 청킹을 사용하는 Whisper (WhisperX) |
| 도메인 특화 (의료, 법률) | wav2vec 2.0 미세 조정(Fine-tune) + 도메인 LM 융합 |

## 2026년에도 여전히 발생하는 실수들 (Pitfalls that still ship in 2026)

- **VAD 미사용.** 무음 구간에서 Whisper를 실행하면 환각(hallucination) 현상(예: "시청해 주셔서 감사합니다!")이 발생합니다. 항상 VAD(Voice Activity Detection)를 통해 게이트를 설정하세요.
- **문자 vs 단어 vs 서브워드 WER.** 정규화(소문자 변환, 문장 부호 제거)를 *거친 후*의 단어 수준 WER(Word Error Rate)을 보고하세요.
- **언어 식별(Language ID) 드리프트.** Whisper의 자동 LID 기능이 노이즈가 섞인 클립을 일본어나 웨일스어로 잘못 분류할 수 있습니다. 언어를 알고 있는 경우에는 `language="en"`으로 강제 지정하세요.
- **청킹(Chunking) 없는 긴 클립.** Whisper는 30초의 윈도우를 가집니다. 30초보다 긴 클립에는 `chunk_length_s=30, stride=5`를 사용하세요.

## Ship It (실전 적용)

`outputs/skill-asr-picker.md`로 저장하세요. 주어진 배포 대상에 맞춰 모델, 디코딩 전략, 청킹(chunking), 그리고 LM 퓨전(LM fusion)을 선택해 보세요.

## 연습 문제 (Exercises)

1. **쉬움 (Easy).** `code/main.py`를 실행해 보세요. 이 코드는 수동으로 작성된 CTC 출력을 탐욕적 디코딩(greedy decoding)하고, 참조 데이터(reference)에 대한 WER을 계산합니다.
2. **중간 (Medium).** 2단계의 접두사 트리 빔 서치(prefix-tree beam search)를 올바르게 구현해 보세요 (blank 병합 규칙을 고려해야 합니다). 10개의 예시로 구성된 합성 데이터셋에서 탐욕적 디코딩 결과와 비교해 보세요.
3. **어려움 (Hard).** [LibriSpeech test-clean](https://www.openslr.org/12) 데이터셋에 `whisper-large-v3-turbo`를 사용해 보세요. 처음 100개의 발화(utterance)에 대해 WER을 계산하고, 이미 발표된 수치와 비교해 보세요.

## 주요 용어 (Key Terms)

| 용어 | 통용되는 표현 | 실제 의미 |
|------|-----------------|-----------------------|
| CTC | blank-token 손실 (The blank-token loss) | 모든 프레임-토큰 정렬에 대한 한계 확률(Marginal); 비자기회귀(non-AR) 방식. |
| RNN-T | 스트리밍 손실 (The streaming loss) | CTC + 다음 토큰 예측기(next-token predictor); 단어 순서를 처리함. |
| Attention enc-dec | Whisper 스타일 (Whisper-style) | 인코더 + 크로스 어텐션(cross-attending) 디코더; 오프라인 품질이 가장 뛰어남. |
| WER | 보고하는 수치 (The number you report) | 단어 수준에서의 `(S+D+I)/N`. |
| Blank | 공백 (The emptiness) | "이번 프레임에는 방출(emission)이 없음"을 알리는 CTC의 특수 토큰. |
| LM fusion | 외부 언어 모델 (External language model) | 빔 서치(beam search) 중에 가중치가 적용된 LM 로그 확률을 추가함. |
| VAD | 침묵 게이트 (The silence gate) | 음성 활동 감지기(Voice activity detector); 비음성 구간을 제거함. |

## 추가 읽을거리 (Further Reading)

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 논문입니다.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 논문입니다.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022년의 표준적인 논문이며, 2024년에 v3-turbo 확장판이 발표되었습니다.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026년 Open ASR 리더보드 1위 모델입니다.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25개 이상의 모델을 대상으로 하는 실시간 벤치마크입니다.
