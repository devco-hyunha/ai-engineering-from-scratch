# 서브워드 토큰화 — BPE, WordPiece, Unigram, SentencePiece (Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece)

> 단어 토크나이저는 미학습 단어(unseen words)에서 멈추고, 문자 토크나이저는 시퀀스 길이를 폭발적으로 늘립니다. 서브워드 토크나이저는 둘 사이를 절충합니다. 모든 현대 LLM은 서브워드 토크나이저를 탑재해 출시됩니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## 문제 (The Problem)

어휘 사전(vocabulary) 크기가 50,000단어라고 가정해 봅시다. 사용자가 "untokenizable"을 입력하면 토크나이저는 `[UNK]`를 반환합니다. 이제 모델은 해당 단어에 대해 아무런 신호도 얻지 못합니다. 더 나쁜 상황은 코퍼스의 상위 90백분위수 문서에 40개의 희귀 단어가 포함되어 있어, 문서당 40비트의 정보가 유실된다는 점입니다.

서브워드 토큰화(Subword tokenization)가 이 문제를 해결합니다. 자주 쓰이는 단어는 단일 토큰으로 유지됩니다. 희귀 단어는 의미 있는 조각으로 분해됩니다: `untokenizable` → `un`, `token`, `izable`. 어떤 문자열이든 궁극적으로 바이트 시퀀스이므로, 학습 데이터가 모든 경우를 커버할 수 있습니다.

2026년 기준 모든 프런티어 LLM은 세 가지 알고리즘(BPE, Unigram, WordPiece) 중 하나를 사용하며, 세 가지 라이브러리(tiktoken, SentencePiece, HF Tokenizers) 중 하나로 래핑되어 배포됩니다. 이 중 하나를 선택하지 않고는 언어 모델을 출시할 수 없습니다.

## 개념 (The Concept)

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** 문자 단위 어휘 사전에서 시작합니다. 모든 인접한 쌍의 빈도를 셉니다. 가장 빈번한 쌍을 새 토큰으로 병합합니다. 목표 어휘 사전 크기에 도달할 때까지 이를 반복합니다. 지배적인 알고리즘: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.** 동일한 알고리즘이지만 유니코드 문자 대신 원시 바이트(256개 기본 토큰)를 대상으로 동작합니다. `[UNK]` 토큰이 전혀 발생하지 않음을 보장하며, 모든 바이트 시퀀스를 인코딩할 수 있습니다. GPT-2는 50,257개 토큰(256개 바이트 + 50,000개 병합 + 1개 특수 토큰)을 사용합니다.

**Unigram.** 거대한 어휘 사전에서 시작합니다. 각 토큰에 유니그램 확률을 할당합니다. 토큰을 제거했을 때 코퍼스의 로그 가능도(log-likelihood) 감소가 가장 적은 토큰들을 반복적으로 프루닝(가지치기)합니다. 추론 시 확률적으로 동작하며, 토큰화를 샘플링할 수 있어(서브워드 정규화를 통한 데이터 증강에 유용) T5, mBART, ALBERT, XLNet, Gemma에서 사용됩니다.

**WordPiece.** 단순 빈도가 아니라 학습 코퍼스의 가능도(likelihood)를 최대화하는 쌍을 병합합니다. BERT, DistilBERT, ELECTRA에서 사용됩니다.

**SentencePiece vs tiktoken.** SentencePiece는 원시 유니코드 텍스트에서 직접 어휘 사전(BPE 또는 Unigram)을 *학습*하는 라이브러리로, 공백을 `▁`로 인코딩합니다. tiktoken은 사전 구축된 어휘 사전을 기반으로 빠르게 *인코딩*하는 OpenAI의 라이브러리이며, 학습 기능은 제공하지 않습니다.

경험 법칙(Rule of thumb):

- **새로운 어휘 사전 학습:** SentencePiece(다국어, 사전 토큰화 불필요) 또는 HF Tokenizers.
- **GPT 어휘 기반 빠른 추론:** tiktoken (`cl100k_base`, `o200k_base`).
- **학습과 추론 모두:** HF Tokenizers — 하나의 라이브러리로 학습과 서빙 모두 지원.

```figure
bpe-merge
```

## 직접 만들기 (Build It)

### 1단계: 밑바닥부터 BPE 구현하기 (BPE from scratch)

`code/main.py`를 살펴보세요. 핵심 루프는 다음과 같습니다:

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

알고리즘이 인코딩하는 세 가지 사실: `</w>`는 단어 끝을 표시하여 "low"(접미사)와 "lower"(접두사)가 서로 구분되도록 합니다. 빈도 가중치 부여는 고빈도 쌍이 초기에 병합되도록 만듭니다. 병합 목록은 순서가 정해져 있어, 추론 시 학습된 순서대로 병합을 적용합니다.

### 2단계: 학습된 병합으로 인코딩하기 (Encode with the learned merges)

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

단순 구현의 시간 복잡도는 O(n·|merges|)입니다. 프로덕션 구현체(tiktoken, HF Tokenizers)는 우선순위 큐를 사용한 병합 순위 룩업을 활용하여 거의 선형 시간에 실행됩니다.

### 3단계: 실무에서의 SentencePiece (SentencePiece in practice)

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 또는 "unigram"
    character_coverage=0.9995, # CJK의 경우 낮춤 (예: 영어 0.9995, 일본어 0.995)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

주목할 점: 사전 토큰화(pre-tokenization)가 필요하지 않고 공백은 `▁`로 인코딩되며, `character_coverage`는 희귀 문자를 보존할지 아니면 `<unk>`로 매핑할지를 얼마나 적극적으로 결정할지 제어합니다.

### 4단계: OpenAI 호환 어휘를 위한 tiktoken (tiktoken for OpenAI-compatible vocabs)

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

인코딩 전용입니다. Rust 백엔드로 빠릅니다. 바이트 계산, 비용 추정, 컨텍스트 윈도우 예산 책정을 위해 GPT-4/5 토큰화와 정확히 일치합니다.

## 2026년에도 여전히 발생하는 함정들 (Pitfalls that still ship in 2026)

- **토크나이저 드리프트(Tokenizer drift).** 어휘 사전 A로 학습하고 어휘 사전 B를 상대로 배포하는 경우입니다. 토큰 ID가 달라져 모델이 쓰레기 출력을 생성합니다. CI에서 `tokenizer.json` 해시를 검사하세요.
- **공백 모호성(Whitespace ambiguity).** BPE에서 "hello"와 " hello"는 서로 다른 토큰을 만듭니다. 항상 `add_special_tokens`와 `add_prefix_space`를 명시적으로 지정하세요.
- **다국어 과소학습(Multilingual undertraining).** 영어 비중이 높은 코퍼스는 비라틴 문자를 5~10배 더 많은 토큰으로 쪼개는 어휘 사전을 만듭니다. GPT-3.5에서 동일한 프롬프트가 일본어/아랍어에서는 5~10배 더 많은 비용을 유발합니다. `o200k_base`가 이를 부분적으로 해결했습니다.
- **이모지 분할(Emoji splits).** 단일 이모지가 5개의 토큰을 차지할 수 있습니다. 컨텍스트 예산을 책정할 때 이모지 처리 방식을 체크하세요.

## 활용하기 (Use It)

2026 스택:

| 상황 | 선택 |
|------|------|
| 밑바닥부터 단일 언어 모델 학습 | HF Tokenizers (BPE) |
| 다국어 모델 학습 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| OpenAI 호환 API 서빙 | tiktoken (GPT-4+용 `o200k_base`) |
| 도메인 특화 어휘 (코드, 수학, 단백질) | 도메인 코퍼스에서 커스텀 BPE 학습 후 기본 어휘와 병합 |
| 엣지 추론, 소형 모델 | Unigram (작은 어휘 사전에서 더 잘 동작함) |

어휘 사전 크기는 상수가 아니라 스케일링 결정 사항입니다. 대략적인 휴리스틱: 매개변수 1B 미만은 32k, 1~10B는 50~100k, 다국어 또는 프런티어 모델은 200k 이상입니다.

## 배포하기 (Ship It)

`outputs/skill-bpe-vs-wordpiece.md`로 저장하세요:

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

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`의 작은 코퍼스로 500회 병합 BPE를 학습시켜 보세요. 홀드아웃(held-out) 단어 3개를 인코딩해 보세요. 정확히 1개 토큰을 생성한 단어와 1개 초과 토큰을 생성한 단어는 각각 몇 개인가요?
2. **보통.** 100개의 영어 위키피디아 문장에서 `cl100k_base`, `o200k_base`, 그리고 vocab=32k로 직접 학습시킨 SentencePiece BPE 간의 토큰 수를 비교해 보세요. 각각의 압축률을 보고하세요.
3. **어려움.** 동일한 코퍼스를 BPE, Unigram, WordPiece로 각각 학습시켜 보세요. 소형 감성 분류기에서 각각을 사용할 때의 다운스트림 정확도를 측정해 보세요. 선택에 따라 F1 점수가 1포인트 이상 달라지나요?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|----------|
| BPE | Byte-Pair Encoding | 목표 어휘 사전 크기에 도달할 때까지 가장 빈번한 문자 쌍을 탐욕적으로 병합. |
| Byte-level BPE | unk 토큰이 절대 없음 | 원시 256바이트를 대상으로 하는 BPE. GPT-2 / Llama가 사용함. |
| Unigram | 확률적 토크나이저 | 로그 가능도를 사용해 대규모 후보 집합에서 프루닝. T5, Gemma가 사용함. |
| SentencePiece | 공백 처리 도구 | 원시 텍스트에서 BPE/Unigram을 학습하는 라이브러리. 공백은 `▁`로 인코딩됨. |
| tiktoken | 빠른 도구 | 사전 구축된 어휘 사전을 위한 OpenAI의 Rust 기반 BPE 인코더. 학습 기능 없음. |
| Merge list | 마법의 숫자들 | `(a, b) → ab` 병합의 정렬된 목록. 추론 시 순서대로 적용됨. |
| Character coverage | 얼마나 희귀해야 너무 희귀한가? | 토크나이저가 커버해야 하는 학습 코퍼스 내 문자의 비율. 일반적으로 ~0.9995. |

## 더 읽을거리 (Further Reading)

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 논문.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 논문.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — SentencePiece 라이브러리 논문.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 간결한 참고 문서.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 쿡북 및 인코딩 목록.
