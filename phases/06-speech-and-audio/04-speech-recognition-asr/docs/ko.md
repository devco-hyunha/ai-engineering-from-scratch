# 음성 인식 (ASR) — CTC, RNN-T, 어텐션

> 음성 인식은 모든 타임스텝에서 오디오를 분류하는 작업이며, 영어와 침묵을 아는 시퀀스 모델이 이를 연결합니다. CTC, RNN-T, 어텐션은 이를 수행하는 세 가지 방법입니다. 하나를 선택하고 그 이유를 이해해 보세요.

**유형:** Build
**언어:** Python
**선수 요건:** 6단계 · 02강 (스펙트로그램 & 멜), 5단계 · 08강 (텍스트를 위한 CNN 및 RNN), 5단계 · 10강 (어텐션)
**시간:** 약 45분

## 문제점

10초 길이의 16 kHz 클립이 있습니다. "turn on the kitchen lights"라는 문자열을 얻고자 합니다. 도전 과제는 구조적입니다. 오디오 프레임은 문자와 일대일로 정렬되지 않습니다. 단어 "okay"는 200 ms가 걸릴 수도, 1200 ms가 걸릴 수도 있습니다. 침묵이 발화를 구분합니다. 일부 음소는 다른 음소보다 길이가 길 수 있습니다. 출력 토큰의 개수는 사전에 알 수 없습니다.

이 문제를 해결하는 세 가지 공식화가 있습니다:

1. **CTC (Connectionist Temporal Classification).** *공백(blank)*이라는 특수 토큰을 포함하여 프레임별 토큰 확률을 방출합니다. 디코딩 시 반복과 공백을 축약합니다. 비자기회귀적이며 빠릅니다. wav2vec 2.0, MMS에서 사용됩니다.
2. **RNN-T (Recurrent Neural Network Transducer).** 조인트 네트워크는 인코더 프레임과 이전 토큰을 고려하여 다음 토큰을 예측합니다. 스트리밍이 가능합니다. Google의 온디바이스 ASR, NVIDIA Parakeet에서 사용됩니다.
3. **어텐션 인코더-디코더.** 인코더는 오디오를 은닉 상태로 압축하고, 디코더는 교차 어텐션을 통해 토큰을 자기회귀적으로 생성합니다. Whisper, SeamlessM4T에서 사용됩니다.

2026년 기준, LibriSpeech test-clean에서의 SOTA WER는 1.4% (Parakeet-TDT-1.1B, NVIDIA)와 1.58% (Whisper-Large-v3-turbo)입니다. 성능 차이는 매우 작지만, 배포 방식의 차이는 큽니다.

## 개념

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 직관.** 인코더가 `T` 프레임 수준에서 `V+1` 토큰(V 문자 + 공백)에 대한 분포를 출력하도록 합니다. 길이 `U < T`인 타겟 문자열 `y`에 대해, `y`로 축약되는 모든 프레임 정렬이 유효합니다. CTC 손실은 이러한 모든 정렬에 대해 합산합니다. 추론: 프레임별 argmax를 취하고, 반복을 축약하며, 공백을 제거합니다.

장점: 비자기회귀, 스트리밍 가능, 제로 룽어헤드. 단점: *조건부 독립 가정* — 각 프레임 예측이 서로 독립적이므로 내부 언어 모델이 없습니다. 빔 검색이나 얕은 융합을 통해 외부 LM으로 보완하세요.

**RNN-T 직관.** 토큰 히스토리를 임베딩하는 *예측자* 네트워크와 예측자 상태와 인코더 프레임을 결합하여 `V+1`에 대한 결합 분포를 생성하는 *조인*을 추가합니다(`+1`은 null / 무음 방출). CTC가 무시한 조건부 의존성을 명시적으로 모델링합니다. 각 단계가 과거 프레임과 과거 토큰에만 조건을 걸기 때문에 스트리밍이 가능합니다.

장점: 스트리밍 가능 + 내부 LM. 단점: 학습이 더 복잡하고 메모리를 많이 소모합니다(3D 손실 격자); RNN-T 손실 커널은 그 자체로 하나의 라이브러리 범주입니다.

**어텐션 인코더-디코더.** 로그-멜 프레임에 대한 인코더(6-32 트랜스포머 레이어). 디코더(6-32 트랜스포머 레이어)는 인코더 출력에 교차 어텐션하여 토큰을 자기회귀적으로 생성합니다. 정렬 제약이 없습니다 — 어텐션은 오디오의 어디든 볼 수 있습니다. 어텐션을 제한하지 않는 한 스트리밍이 불가능합니다(청크 Whisper-Streaming, 2024).

장점: 오프라인 ASR에서 최고 품질, 표준 seq2seq 도구로 쉽게 학습 가능. 단점: 자기회귀 지연은 출력 길이에 비례합니다; 엔지니어링 없이는 스트리밍할 수 없습니다.

### WER: 하나의 숫자

**단어 오류율(WER)** = `(S + D + I) / N`, 여기서 S=치환, D=삭제, I=삽입, N=참조 단어 수. 단어 수준에서 레벤슈타인 편집 거리와 일치합니다. 낮을수록 좋습니다. WER이 20% 이상이면 일반적으로 사용 불가능하며, 5% 미만은 낭독된 음성에서 인간 수준의 성능입니다. 표준 벤치마크에서의 2026년 수치:

| 모델 | LibriSpeech test-clean | LibriSpeech test-other | 크기 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 파라미터 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

이 모든 모델은 인코더-디코더 또는 RNN-T 기반입니다. 순수 CTC 시스템(wav2vec 2.0)은 test-clean에서 약 1.8–2.1%입니다.

```figure
ctc-collapse
```

## 구현하기

### 1단계: 탐욕 CTC 디코딩

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

두 가지 규칙: 연속 반복을 병합하고, 공백을 제거합니다. 예: `a a _ _ a b b _ c` → `a a b c`.

### 2단계: 빔 검색 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (토큰, log_prob)
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

프로덕션에서는 LM 융합을 사용하는 접두어 트리 빔 검색을 사용하며, 이는 개념적 골격입니다.

### 3단계: WER

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

### 4단계: Whisper를 대상으로 추론

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026년 기준 가장 강력한 범용 ASR을 위한 한 줄 코드입니다. 24 GB GPU에서 약 20배 실시간 속도로 실행됩니다.

### 5단계: Parakeet 또는 wav2vec 2.0을 사용한 스트리밍

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

스트리밍 ASR은 청킹된 인코더 어텐션과 캐리오버 상태가 필요하므로, 이를 지원하는 라이브러리를 사용하세요 (Parakeet의 경우 NeMo, `chunk_length_s`와 `transformers` 파이프라인).

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 영어, 오프라인, 최대 품질 | Whisper-large-v3-turbo |
| 다국어, 견고함 | SeamlessM4T v2 |
| 스트리밍, 낮은 지연 | Parakeet-TDT-1.1B 또는 Riva |
| 엣지, 모바일, <500 ms 지연 | 양자화된 Whisper-Tiny 또는 Moonshine (2024) |
| 롱폼 | VAD 기반 청킹을 사용하는 Whisper (WhisperX) |
| 도메인 특화 (의료, 법률) | wav2vec 2.0 미세 조정 + 도메인 LM 융합 |

## 2026년에도 여전히 배포되는 함정

- **VAD 없음.** Whisper를 무음에 실행하면 환각("Thanks for watching!")이 발생합니다. 항상 VAD로 게이트 처리하세요.
- **문자 vs 단어 vs 서브워드 WER.** 정규화(소문자화, 구두점 제거) *후*에 단어 단위 WER를 보고하세요.
- **언어 식별(LID) 드리프트.** Whisper의 자동 LID는 잡음이 있는 클립을 일본어나 웨일스어로 잘못 라우팅합니다. 알고 있다면 `language="en"`를 강제하세요.
- **청킹 없는 긴 클립.** Whisper는 30초 윈도우를 가집니다. 더 긴 경우 `chunk_length_s=30, stride=5`를 사용하세요.

## 출시하기

`outputs/skill-asr-picker.md`로 저장하세요. 특정 배포 대상에 대해 모델, 디코딩 전략, 청킹 및 LM 융합을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 이는 수작업으로 만든 CTC 출력을 탐욕적으로 디코딩하고 참조에 대해 WER를 계산합니다.
2. **중간 난이도.** 2단계에서 접두어 트리 빔 검색을 올바르게 구현하세요 (공백 병합 규칙을 고려해야 합니다). 10개 예제의 합성 데이터셋에서 탐욕(greedy) 방식과 비교해 보세요.
3. **어려운 난이도.** [LibriSpeech test-clean](https://www.openslr.org/12)에 `whisper-large-v3-turbo`를 사용하세요. 첫 100개 발화에 대해 WER을 계산하세요. 공개된 수치와 비교해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| CTC | 공백 토큰 손실 | 모든 프레임-토큰 정렬에 대한 주변 확률(marginal); 비자기회귀(non-AR). |
| RNN-T | 스트리밍 손실 | CTC + 다음 토큰 예측기; 단어 순서를 처리합니다. |
| 어텐션 인코더-디코더 | Whisper 스타일 | 인코더 + 교차 어텐션 디코더; 오프라인 품질이 가장 좋습니다. |
| WER | 보고하는 수치 | 단어 수준에서의 `(S+D+I)/N`. |
| 공백(Blank) | 공백 상태 | CTC에서 "이 프레임에서는 방출(emission) 없음"을 신호하는 특수 토큰. |
| LM 융합 | 외부 언어 모델 | 빔 검색 중 가중치가 적용된 LM 로그 확률을 추가합니다. |
| VAD | 침묵 게이트 | 음성 활동 감지(Voice Activity Detector); 비음성 부분을 잘라냅니다. |

## 추가 읽기

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 논문.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 논문.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022년 표준 논문; 2024년에 v3-turbo 확장판이 나왔습니다.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026년 Open ASR 리더보드 선두.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25개 이상의 모델을 아우르는 라이브 벤치마크.
