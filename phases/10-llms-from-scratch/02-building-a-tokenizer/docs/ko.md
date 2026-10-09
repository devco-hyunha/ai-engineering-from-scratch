# 토크나이저를 처음부터 구축하기

> 1강에서는 장난감을 주었습니다. 이 강에서는 무기를 드립니다.

**유형:** Build
**언어:** Python
**선수 요건:** 10단계, 01강 (토크나이저: BPE, WordPiece, SentencePiece)
**시간:** 약 90분

## 학습 목표

- 유니코드, 공백 정규화 및 특수 토큰을 처리하는 프로덕션급 BPE 토크나이저를 구축합니다
- 바이트 수준 폴백을 구현하여 토크나이저가 이모지, CJK 및 코드를 포함하여 모든 입력을 unknown 토큰 없이 인코딩할 수 있도록 합니다
- BPE 병합을 적용하기 전에 텍스트를 단어 경계에서 분할하는 프리토크나이징 정규식 패턴을 추가합니다
- 코퍼pus에 사용자 정의 토크나이저를 학습하고 다국어 텍스트에서 tiktoken 대비 압축률을 평가합니다

## 문제점

1강의 BPE 토크나이저는 영어 텍스트에서 작동합니다. 이제 일본어를 입력해 보세요. 이모지도 입력해 보세요. 탭과 공백이 섞인 Python 코드도 입력해 보세요.

고장납니다.

BPE가 잘못되어서가 아닙니다 -- 구현이 불완전하기 때문입니다. 프로덕션 토크나이저는 모든 인코딩의 원시 바이트를 처리하고, 분할 전에 유니코드를 정규화하며, 절대 병합되지 않는 특수 토큰을 관리하고, 프리토크나이징과 서브워드 분할을 연결하며, 15조 개 토큰을 처리하는 학습 파이프라인의 병목이 발생하지 않을 정도로 충분히 빠르게 이 모든 작업을 수행합니다.

GPT-2의 토크나이저는 50,257개의 토큰을 가집니다. Llama 3은 128,256개입니다. GPT-4는 약 100,000개입니다. 이는 장난감 같은 숫자가 아닙니다. 이러한 어휘 뒤의 병합 테이블은 수백 GB의 텍스트로 학습되었으며, 주변 메커니즘 -- 정규화, 프리토크나이징, 특수 토큰 주입, 채팅 템플릿 포맷팅 --은 "hello world"를 처리하는 토크나이저와 인터넷 전체를 처리하는 토크나이저를 구분하는 요소입니다.

이 메커니즘을 구축할 것입니다.

## 개념

### 전체 파이프라인

프로덕션 토크나이저는 하나의 알고리즘이 아닙니다. 서로 다른 문제를 해결하는 5단계의 파이프라인입니다.

```mermaid
graph LR
    A["원시 텍스트"] --> B["Normalize"]
    B --> C["Pre-Tokenize"]
    C --> D["BPE 병합"]
    D --> E["특수 토큰"]
    E --> F["토큰 ID"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

각 단계는 특정 작업을 수행합니다:

| 단계 | 수행 내용 | 중요성 |
|-------|-------------|----------------|
| 정규화 | NFKC 유니코드, 선택적 소문자화, 선택적 악센트 제거 | "fi" 리거(U+FB01)가 "fi"(두 문자)로 변환됩니다. 이 단계가 없으면 같은 단어가 서로 다른 토큰을 생성합니다. |
| 사전 토큰화 | BPE 전에 텍스트를 청크로 분할 | BPE가 단어 경계를 넘어 병합하는 것을 방지합니다. "the cat"은 절대 "e c"라는 토큰을 생성하지 않아야 합니다. |
| BPE 병합 | 학습된 병합 규칙을 바이트 시퀀스에 적용 | 핵심 압축 단계입니다. 원시 바이트를 하위 단어(subword) 토큰으로 변환합니다. |
| 특수 토큰 | [BOS], [EOS], [PAD], 채팅 템플릿 마커 삽입 | 이 토큰들은 고정된 ID를 가지며, BPE 병합에 참여하지 않습니다. 모델은 구조를 위해 이 토큰들을 필요로 합니다. |
| ID 매핑 | 토큰 문자열을 정수 ID로 변환 | 모델은 문자열이 아닌 정수를 봅니다. |

### 바이트 수준 BPE

1강의 토크나이저는 UTF-8 바이트를 처리했습니다. 이는 올바른 선택이었습니다. 하지만 중요한 한 가지를 간과했습니다: 그 바이트들이 유효한 UTF-8가 아닐 경우 어떻게 되는지입니다.

바이트 수준 BPE는 모든 가능한 바이트 값(0-255)을 유효한 토큰으로 처리하여 이 문제를 해결합니다. 기본 어휘는 정확히 256개 항목입니다. 텍스트, 바이너리, 손상된 파일 등 모든 파일은 알 수 없는 토큰을 생성하지 않고 토큰화할 수 있습니다.

GPT-2는 한 가지 트릭을 추가했습니다: 각 바이트를 인쇄 가능한 유니코드 문자로 매핑하여 어휘가 사람이 읽기 쉽도록 유지합니다. 바이트 0x20(공백)은 해당 매핑에서 문자 "G"가 됩니다. 이는 purely cosmetic(순수하게 외형적인) 것입니다. 알고리즘은 이를 신경 쓰지 않습니다.

진정한 강점: 바이트 수준 BPE는 지구상의 모든 언어를 처리합니다. 중국어 문자는 각각 3바이트의 UTF-8입니다. 일본어는 3-4바이트일 수 있습니다. 아랍어, 데바나가리, 이모지 -- 모두 바이트 시퀀스일 뿐입니다. BPE 알고리즘은 영어 ASCII 바이트에서 패턴을 찾는 것과 정확히 동일한 방식으로 이러한 바이트 시퀀스에서 패턴을 찾습니다.

### 사전 토큰화

BPE가 텍스트를 처리하기 전에, 텍스트를 청크로 분할해야 합니다. 이는 병합 알고리즘이 단어 경계를 넘어 토큰을 생성하는 것을 방지합니다.

GPT-2는 텍스트를 분할하기 위해 정규식 패턴을 사용합니다:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

이 패턴은 축약형("don't"가 "don" + "'t"가 됨), 선행 공백이 있는 단어, 숫자, 구두점, 공백을 기준으로 분할합니다. 선행 공백은 단어에 붙어 유지되므로 "the cat"은 ["the", " ", "cat"]가 아니라 [" the", " cat"]가 됩니다.

Llama는 정규식을 완전히 생략하는 SentencePiece를 사용합니다. 원시 바이트 스트림을 하나의 긴 시퀀스로 취급하고 BPE 알고리즘이 경계를 결정하도록 합니다. 이 방식은 더 단순하지만 BPE가 단어 간 토큰을 생성할 자유도를 높여줍니다.

이 선택은 중요합니다. GPT-2의 정규식은 토크나이저가 한 단어의 끝 "the"와 다음 단어의 시작 "the"가 병합되는 것을 방지합니다. SentencePiece는 이를 허용하며, 이는 때로 더 효율적인 압축을 가능하게 하지만 토큰의 해석 가능성은 낮추는 결과를 낳습니다.

### 특수 토큰

모든 프로덕션 토크나이저는 구조적 마커를 위해 토큰 ID를 예약합니다:

| 토큰 | 용도 | 사용 모델 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 시퀀스 시작 | Llama 3, GPT |
| `[EOS]` / `</s>` | 시퀀스 종료 | 모든 모델 |
| `[PAD]` | 배치 정렬을 위한 패딩 | BERT, T5 |
| `[UNK]` | 알 수 없는 토큰 (바이트 수준 BPE는 이를 제거함) | BERT, WordPiece |
| `<\|im_start\|>` | 채팅 메시지 경계 시작 | ChatGPT, Qwen |
| `<\|im_end\|>` | 채팅 메시지 경계 종료 | ChatGPT, Qwen |
| `<\|user\|>` | 사용자 턴 마커 | Llama 3 |
| `<\|assistant\|>` | 어시스턴트 턴 마커 | Llama 3 |

특수 토큰은 BPE에 의해 분할되지 않습니다. 병합 알고리즘이 실행되기 전에 정확히 매칭되어 고정된 ID로 대체되며, 주변 텍스트는 정상적으로 토큰화됩니다.

### 채팅 템플릿

이 부분이 대부분의 사람들이 혼란을 겪고 많은 구현이 깨지는 지점입니다.

채팅 모델에 메시지를 보낼 때, API는 메시지 목록을 허용합니다:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

모델은 JSON을 보지 않습니다. 평탄한 토큰 시퀀스를 봅니다. 채팅 템플릿은 특수 토큰을 사용하여 메시지를 그 평탄한 시퀀스로 변환합니다. 모든 모델은 이 방식을 다르게 구현합니다:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

템플릿을 잘못 지정하면 모델이 무의미한 결과를 생성합니다. 모델은 하나의 정확한 형식으로 학습되었습니다. 줄바꿈 누락, 토큰 순서 변경, 추가 공백 등 어떤 편차도 입력을 학습 분포 밖으로 밀어냅니다.

### 속도

프로덕션 환경의 토큰화에는 Python이 너무 느립니다.

tiktoken (OpenAI)은 Rust로 작성되었으며 Python 바인딩을 제공합니다. HuggingFace tokenizers도 Rust로 작성되었습니다. SentencePiece는 C++입니다. 이들은 순수 Python 대비 10~100배의 속도 향상을 달성합니다.

참고로: Llama 3 사전 학습을 위해 15조 개의 토큰을 토큰화할 때, 초당 100만 토큰 (빠른 Python) 속도로 처리하면 174일이 걸립니다. 초당 1억 토큰 (Rust) 속도로 처리하면 1.7일이 걸립니다.

알고리즘을 이해하기 위해 Python으로 구축하고 있습니다. 프로덕션 환경에서는 컴파일된 구현을 사용하고 Python 래퍼만 건드리게 됩니다.

```figure
weight-tying
```

## 구현하기

### 1단계: 바이트 수준 인코딩

기반입니다. 모든 문자열을 바이트 시퀀스로 변환하고, 각 바이트를 표시 가능한 문자로 매핑한 후, 이 과정을 역으로 수행합니다.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

다국어 텍스트로 테스트하여 바이트 수를 확인해 보세요:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello"는 5바이트입니다. "你好"는 6바이트 (문자당 3바이트)입니다. 불꽃 이모지는 4바이트입니다. 바이트 수준 토크나이저는 언어에 관계없이 동일하게 동작합니다. 바이트는 바이트일 뿐입니다.

### 2단계: 정규식 기반 프리토크나이저

GPT-2 정규식 패턴을 사용하여 텍스트를 청크로 분할합니다. 각 청크는 BPE에 의해 독립적으로 토큰화됩니다.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 모듈은 유니코드 속성 이스케이프 (`\p{L}`는 문자, `\p{N}`는 숫자)를 지원합니다. 표준 라이브러리 `re` 모듈은 이를 지원하지 않으므로 ASCII 문자 클래스로 대체합니다. 프로덕션 다국어 토크나이저를 위해 `regex`을 설치하세요.

직접 해 보세요:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

선행 공백은 단어가 붙어 있습니다. 축약형은 아포스트로피에서 분할됩니다. 구두점은 자체 청크가 됩니다. BPE는 이러한 경계를 넘어 토큰을 병합하지 않습니다.

### 3단계: 바이트 시퀀스에 대한 BPE

01강의 핵심 알고리즘이지만, 이제 프리토크나이저로 분할된 청크에 대해 독립적으로 동작합니다.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 4단계: 특수 토큰 처리

특수 토큰은 정확한 일치와 고정된 ID가 필요합니다. 이들은 BPE를 완전히 우회합니다.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 5단계: 전체 토크나이저 클래스

모든 것을 연결하세요: 정규화, 특수 토큰으로 분할, 프리토크나이징, BPE 병합, ID 매핑.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 6단계: 다국어 테스트

실제 테스트입니다. 영어, 중국어, 이모지, 코드를 입력해 보세요.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

중국어 문자는 각각 3바이트를 생성합니다. 이모지는 4바이트를 생성합니다. 이 중 어느 것도 토크나이저를 충돌시키지 않습니다. 어느 것도 알 수 없는 토큰을 생성하지 않습니다. 이것이 바이트 수준 BPE의 힘입니다.

## 사용하기

### 실제 토크나이저 비교

Llama 3, GPT-4, Mistral의 실제 토크나이저를 로드하세요. 동일한 다국어 단락에 대해 각각이 어떻게 처리하는지 확인해 보세요.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

동일한 텍스트에 대해 서로 다른 토큰 수를 보게 될 것입니다. 128K 어휘를 가진 Llama 3은 일반적인 패턴 병합에 더 공격적입니다. 100K를 가진 GPT-4는 중간에 위치합니다. 32K를 가진 Mistral은 더 많은 토큰을 생성하지만 임베딩 레이어가 더 작습니다.

트레이드오프는 항상 동일합니다: 더 큰 어휘는 더 짧은 시퀀스를 의미하지만 더 많은 매개변수를 필요로 합니다.

## 출시하기

이 강의는 프로덕션 토크나이저를 구축하고 디버깅하기 위한 프롬프트를 생성합니다. `outputs/prompt-tokenizer-builder.md`를 참조하세요.

## 연습 문제

1. **쉬움:** 임의의 토큰 ID에 대해 원시 바이트를 표시하는 `get_token_bytes(id)` 메서드를 추가하세요. 이를 사용하여 가장 흔한 병합된 토큰이 실제로 무엇을 나타내는지 검사해 보세요.
2. **중간:** 공백과 숫자로 분할하지만 선행 공백은 유지하는 Llama 스타일의 프리토크나이저를 구현하세요. 동일한 코퍼스에 대해 GPT-2의 정규식 접근법과 어휘를 비교하세요.
3. **어려움:** `{"role": ..., "content": ...}` 메시지 목록을 받아 Llama 3 채팅 형식에 대한 올바른 토큰 시퀀스를 생성하는 채팅 템플릿 메서드를 추가하세요. HuggingFace 구현과 비교하여 테스트해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 바이트 수준 BPE | "바이트에서 작동하는 토크나이저" | 256개의 바이트 값으로 기본 어휘를 구성하는 BPE -- 알 수 없는 토큰 없이 모든 입력을 처리 |
| 프리토크나이징 | "BPE 전에 분할" | BPE가 단어 경계를 넘어 병합하는 것을 방지하는 정규식 또는 규칙 기반 분할 |
| NFKC 정규화 | "유니코드 정리" | 표준 분해 후 호환성 결합 -- "fi" 합자 문자가 "fi"가 되고, 전각 "A"가 "A"가 됩니다 |
| 채팅 템플릿 | "메시지가 토큰이 되는 방법" | 역할/내용 메시지 목록을 평탄한 토큰 시퀀스로 변환하는 정확한 형식 -- 모델별이며 학습 형식과 일치해야 합니다 |
| 특수 토큰 | "제어 토큰" | BPE를 우회하는 예약된 토큰 ID -- [BOS], [EOS], [PAD], 채팅 마커 -- 병합 전에 정확히 매칭됩니다 |
| 생식률 | "단어당 토큰 수" | 출력 토큰과 입력 단어의 비율 -- GPT-4의 영어는 1.3, 한국어는 2-3이며, 높을수록 컨텍스트가 낭비됩니다 |
| tiktoken | "OpenAI 토크나이저" | Python 바인딩을 갖춘 Rust BPE 구현 -- 순수 Python보다 10-100배 빠릅니다 |
| 병합 테이블 | "어휘" | 학습 중에 학습된 바이트 쌍 병합의 순서 목록 -- 이것이 토크나이저의 학습된 지식입니다 |

## 추가 읽기

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- GPT-3.5/4에서 사용하는 Rust BPE 구현
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- BPE, WordPiece, Unigram을 지원하는 Rust 토크나이저 라이브러리
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 128K 어휘 및 토크나이저 학습에 대한 상세 정보
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- 언어에 독립적인 토큰화
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 원본 바이트-유니코드 매핑
