# 하위 단어 토큰화 — BPE, WordPiece, Unigram, SentencePiece

> 단어 토크나이저는 본 적 없는 단어를 처리하지 못합니다. 문자 토크나이저는 시퀀스 길이가 폭발합니다. 하위 단어 토크나이저는 이 둘의 절충안입니다. 모든 최신 LLM은 이 중 하나를 기반으로 출시됩니다.

**유형:** Learn
**언어:** Python
**선수 요건:** 5단계 · 01강 (텍스트 처리), 5단계 · 04강 (GloVe / FastText / 하위 단어)
**시간:** 약 60분

## 문제점

어휘가 50,000개 단어로 구성되어 있습니다. 사용자가 "untokenizable"을 입력합니다. 토크나이저는 `[UNK]`를 반환합니다. 모델은 이제 단어에 대한 신호를 전혀 갖지 못합니다. 더 나쁜 점은, 코퍼스의 90퍼센타일 문서가 40개의 희귀 단어를 포함한다는 것입니다. 이는 문서당 40비트의 정보가 손실됨을 의미합니다.

하위 단어 토큰화는 이 문제를 해결합니다. 흔한 단어는 단일 토큰으로 유지됩니다. 희귀 단어는 의미 있는 조각으로 분해됩니다: `untokenizable` → `un`, `token`, `izable`. 모든 문자열은 궁극적으로 바이트 시퀀스이므로, 학습 데이터가 모든 것을 포괄합니다.

2026년의 모든 최전선 LLM은 세 가지 알고리즘(BPE, Unigram, WordPiece) 중 하나를 세 가지 라이브러리(tiktoken, SentencePiece, HF Tokenizers) 중 하나로 감싸서 출시됩니다. 언어 모델을 출시하려면 이 중 하나를 선택해야 합니다.

## 개념

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (바이트 쌍 인코딩)(Byte-Pair Encoding).** 문자 수준 어휘로 시작합니다. 모든 인접 쌍을 세어 가장 빈번한 쌍을 새로운 토큰으로 병합합니다. 목표 어휘 크기에 도달할 때까지 반복합니다. 지배적인 알고리즘입니다: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**바이트 수준 BPE.** 유니코드 문자 대신 원시 바이트(256개 기본 토큰)에 대해 동일한 알고리즘을 적용합니다. `[UNK]` 토큰이 0개임을 보장합니다 — 모든 바이트 시퀀스가 인코딩됩니다. GPT-2는 50,257개 토큰(256개 바이트 + 50,000개 병합 + 1개 특수 토큰)을 사용합니다.

**Unigram.** 거대한 어휘로 시작합니다. 각 토큰에 유니그램 확률을 할당합니다. 코퍼스 로그 우도(log-likelihood)를 가장 적게 증가시키는 토큰을 반복적으로 제거합니다. 추론 시 확률적입니다: 토큰화를 샘플링할 수 있습니다(하위 단어 정규화를 통한 데이터 증강에 유용). T5, mBART, ALBERT, XLNet, Gemma가 사용합니다.

**WordPiece.** 원시 빈도 대신 학습 코퍼스의 우도를 극대화하는 쌍을 병합합니다. BERT, DistilBERT, ELECTRA가 사용합니다.

**SentencePiece vs tiktoken.** SentencePiece는 원시 유니코드 텍스트에 직접 어휘를 *훈련*(BPE 또는 Unigram)하는 라이브러리로, 공백을 `▁`로 인코딩합니다. tiktoken은 OpenAI의 사전 구축된 어휘에 대한 빠른 *인코더*이며, 훈련을 수행하지 않습니다.

경험칙:

- **새로운 어휘 훈련:** SentencePiece (다국어 지원, 사전 토큰화 없음) 또는 HF Tokenizers를 사용하세요.
- **GPT 어휘에 대한 빠른 추론:** tiktoken (cl100k_base, o200k_base)을 사용하세요.
- **둘 다:** HF Tokenizers — 하나의 라이브러리로 훈련 및 서빙을 수행합니다.

```figure
bpe-merge
```

## 구현하기

### 1단계: 처음부터 BPE 구현

`code/main.py`을 참고하세요. 루프는 다음과 같습니다:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

이 알고리즘이 인코딩하는 세 가지 사실. `</w>`은 단어 끝을 표시하므로 접미사 "low"와 접두사 "lower"가 구별됩니다. 빈도 가중치로 인해 고빈도 쌍이 초기에 우선 병합됩니다. 병합 목록은 순서가 정해져 있으며, 추론 시 훈련 순서대로 병합을 적용합니다.

### 2단계: 학습된 병합으로 인코딩

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

소박한 구현은 O(n·|merges|) 복잡도를 가집니다. 프로덕션 구현(tiktoken, HF Tokenizers)은 병합 순위 조회와 우선순위 큐를 사용하여 거의 선형 시간에 실행됩니다.

### 3단계: SentencePiece 실전 사용

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 또는 "unigram"
    character_coverage=0.9995, # CJK의 경우 낮게 설정 (예: 영어는 0.9995, 일본어는 0.995)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

주의: 사전 토큰화가 필요 없으며, 공백은 `▁`로 인코딩되고, `character_coverage`는 희소 문자를 보존하는 것과 `<unk>`로 매핑하는 것 중 어느 쪽을 더 공격적으로 적용할지 제어합니다.

### 4단계: OpenAI 호환 어휘를 위한 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

인코딩 전용입니다. 빠릅니다 (Rust 백엔드). 바이트 수 계산, 비용 추정, 컨텍스트 윈도우 예산 관리에 GPT-4/5 토큰화와 정확히 일치합니다.

## 2026년에도 여전히 발생하는 함정

- **토크나이저 드리프트.** 어휘 A로 훈련하고 어휘 B로 배포하는 경우. 토큰 ID가 달라져 모델이 엉뚱한 출력을 생성합니다. CI에서 `tokenizer.json` 해시를 확인하세요.
- **공백 모호성.** BPE에서 "hello"와 " hello"는 서로 다른 토큰을 생성합니다. `add_special_tokens`과 `add_prefix_space`을 항상 명시적으로 지정하세요.
- **다국어 학습 부족.** 영어 중심의 코퍼스는 비라틴 문자를 5-10배 더 많은 토큰으로 분할하는 어휘를 생성합니다. GPT-3.5에서 일본어/아랍어는 동일한 프롬프트가 5-10배 더 많은 비용이 듭니다. o200k_base는 이를 부분적으로 해결했습니다.
- **이모지 분할.** 단일 이모지가 5개의 토큰을 차지할 수 있습니다. 컨텍스트 예산을 설정할 때 이모지 처리를 점검해 보세요.

## 사용하기

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 단일 언어 모델을 처음부터 학습 | HF Tokenizers (BPE) |
| 다국어 모델 학습 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| OpenAI 호환 API 서빙 | tiktoken (GPT-4+용 `o200k_base`) |
| 도메인 특화 어휘 (코드, 수학, 단백질) | 도메인 코퍼스에 맞춤형 BPE를 학습하고 기본 어휘와 병합 |
| 엣지 추론, 소형 모델 | Unigram (더 작은 어휘가 더 잘 작동) |

어휘 크기는 상수가 아니라 확장 결정입니다. 대략적인 경험칙: 10억 매개변수 미만은 32k, 10억~100억은 50-100k, 다국어/프론티어는 200k 이상.

## 출시하기

`outputs/skill-bpe-vs-wordpiece.md`로 저장:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 연습 문제

1. **쉬움.** `code/main.py`의 작은 코퍼스에 500 병합 BPE를 학습하세요. 홀드아웃된 세 단어를 인코딩해 보세요. 정확히 1개 토큰이 생성된 경우와 1개 이상 토큰이 생성된 경우는 각각 몇 개입니까?
2. **중간.** `cl100k_base`, `o200k_base`, 그리고 어휘 크기가 32k인 SentencePiece BPE를 학습하여 100개의 영어 위키백과 문장에 대한 토큰 수를 비교해 보세요. 각 방법의 압축 비율을 보고하세요.
3. **어려움.** 동일한 코퍼스를 BPE, Unigram, WordPiece로 학습하세요. 작은 감정 분류기에 각각을 사용했을 때의 다운스트림 정확도를 측정해 보세요. 선택에 따라 F1 점수가 1점 이상 변동합니까?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| BPE | 바이트 쌍 인코딩 (Byte-Pair Encoding) | 목표 어휘 크기에 도달할 때까지 가장 빈번한 문자 쌍을 탐욕적으로 병합합니다. |
| 바이트 수준 BPE | 알 수 없는 토큰이 절대 없음 | 원시 256바이트에 대한 BPE; GPT-2 / Llama가 이를 사용합니다. |
| Unigram | 확률적 토크나이저 | 로그 우도를 사용하여 큰 후보 집합에서 가지치기(pruning)를 수행합니다; T5, Gemma가 사용합니다. |
| SentencePiece | 공백 처리 라이브러리 | 원시 텍스트에 BPE/Unigram을 학습하는 라이브러리; 공백은 `▁`로 인코딩됩니다. |
| tiktoken | 빠른 인코더 | OpenAI의 Rust 기반 BPE 인코더로, 사전 구축된 어휘를 사용합니다. 학습이 필요 없습니다. |
| 병합 목록 | 마법 같은 숫자 | `(a, b) → ab` 병합의 순서 있는 목록; 추론 시 순서대로 적용됩니다. |
| 문자 커버리지 | 얼마나 희소해야 희소일까요? | 토크나이저가 학습 코퍼스의 문자 중 커버해야 하는 비율; 일반적으로 ~0.9995입니다. |

## 추가 읽기

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 논문입니다.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 논문입니다.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 라이브러리입니다.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 간결한 참고 자료입니다.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 쿡북 + 인코딩 목록입니다.
