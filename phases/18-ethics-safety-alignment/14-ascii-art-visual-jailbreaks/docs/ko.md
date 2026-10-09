# ASCII 아트와 시각적 제일브레이크

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). 유해한 요청에서 안전 관련 토큰을 마스킹하고, 동일한 문자의 ASCII 아트 렌더링으로 대체한 후 은폐된 프롬프트를 전송합니다. GPT-3.5, GPT-4, Gemini, Claude, Llama-2 모두 ASCII 아트 토큰을 강건하게 인식하지 못합니다. 이 공격은 PPL (퍼플렉시티 필터), Paraphrase 방어, Retokenization을 우회합니다. 관련 내용: ViTC 벤치마크는 비의미적 시각적 프롬프트의 인식을 측정합니다. StructuralSleight는 Uncommon Text-Encoded Structures (트리, 그래프, 중첩 JSON)를 인코딩 공격의 한 계열로 일반화합니다.

**유형:** Build
**언어:** Python (stdlib, ArtPrompt 토큰 마스킹 하네스)
**선수 요건:** 18단계 · 12강 (PAIR), 18단계 · 13강 (MSJ)
**시간:** 약 60분

## 학습 목표

- ArtPrompt 공격을 설명해 보세요: 단어 식별 단계, ASCII 아트 대체, 최종 은폐된 프롬프트.
- ArtPrompt에서 표준 방어 기법 (PPL, Paraphrase, Retokenization)이 실패하는 이유를 설명해 보세요.
- ViTC를 정의하고, 이 벤치마크가 무엇을 측정하는지 설명해 보세요.
- StructuralSleight가 임의의 Uncommon Text-Encoded Structures에 대한 일반화임을 설명해 보세요.

## 문제점

패러프레이즈와 롤플레이를 통한 공격 (12강) 및 긴 컨텍스트를 통한 공격 (13강)은 텍스트 수준의 패턴을 조작합니다. ArtPrompt는 인식 수준에서 작동합니다. 모델은 금지된 토큰을 파싱하지 않습니다. 문자로 렌더링된 이미지를 파싱합니다. 안전 필터는 무해한 구두점만 보고, 모델은 단어를 봅니다.

## 개념

### ArtPrompt, 두 단계

1단계. 단어 식별. 유해한 요청이 주어지면, 공격자는 LLM을 사용하여 안전 관련 단어를 식별합니다 (예: "how to make a bomb"에서 "bomb").

2단계. 은폐된 프롬프트 생성. 식별된 각 단어를 ASCII 아트 렌더링 (문자 모양을 형성하는 7x5 또는 7x7 문자 블록)으로 대체합니다. 모델은 충분히 유능한 모델이 단어로 인식할 수 있는 구두점과 공백의 그리드를 받습니다. 안전 필터는 그리드만 봅니다.

결과: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 모두 실패합니다. 벤치마크 하위 집합에서 공격 성공률이 75% 이상입니다.

### 표준 방어책이 실패하는 이유

- **PPL (퍼플렉시티 필터).** ASCII 아트는 높은 퍼플렉시티를 가지지만, 모든 새로운 입력도 마찬가지입니다. ArtPrompt를 차단하는 임계값 선택은 합법적인 구조화된 입력도 차단합니다.
- **패러프레이즈.** 프롬프트를 패러프레이즈하면 ASCII 아트가 파괴됩니다. 실제로는 패러프레이즈 LLM이 아트를 보존하거나 재구성하는 경우가 많습니다.
- **리토큰화.** 토큰을 다르게 분할해도 모델의 비전이 문자 모양을 인식한다는 사실은 변하지 않습니다.

기본적인 문제는 안전 필터가 토큰 또는 의미론적 수준에서 작동하는 반면, ArtPrompt는 시각적 인식 수준에서 작동한다는 점입니다.

### ViTC 벤치마크

비 의미론적 시각적 프롬프트의 인식. 모델이 ASCII 아트, 윙딩 및 기타 비 텍스트 의미론적 시각적 콘텐츠를 읽는 능력을 측정합니다. ArtPrompt의 효과는 ViTC 정확도와 상관관계가 있습니다. 모델이 시각적 텍스트를 잘 읽을수록 ArtPrompt가 더 잘 작동합니다. 이는 능력-안전 트레이드오프입니다.

### StructuralSleight

ArtPrompt를 일반화합니다: 드문 텍스트 인코딩 구조(UTES). 트리, 그래프, 중첩 JSON, JSON 내 CSV, diff 스타일 코드 블록. 구조가 훈련 안전 데이터에서 드물지만 모델이 파싱할 수 있다면, 유해한 콘텐츠를 숨길 수 있습니다.

방어적 함의: 안전성은 모델이 파싱할 수 있는 구조화된 표현 전반에 걸쳐 일반화되어야 합니다. 이 집합은 크고 증가하고 있습니다.

### 이미지 모달리티 유사체

비전 LLM(GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1)은 공격 표면을 확장합니다. 실제 이미지를 사용한 ArtPrompt 스타일 공격은 이미지 인코더가 더 풍부한 신호를 생성하기 때문에 ASCII 아트 유사체보다 더 강합니다.

### 18단계에서의 위치

12-14강은 세 가지 직교 공격 벡터를 설명합니다: 반복적 정제(PAIR), 컨텍스트 길이(MSJ), 인코딩(ArtPrompt/StructuralSleight). 15강은 모델 중심 공격에서 시스템 경계 공격(간접 프롬프트 주입)으로 전환합니다. 16강은 방어적 도구 대응을 설명합니다.

```figure
al-ascii-cloak
```

## 사용하기

`code/main.py`는 장난감 ArtPrompt를 구축합니다. 유해한 쿼리에서 특정 단어를 ASCII 아트 글리프로 은폐하고, 은폐된 문자열이 키워드 필터를 통과하는지 검증하며, (선택적으로) 간단한 인식기를 사용하여 은폐된 문자열을 디코딩할 수 있습니다.

## 출시하기

이 강의는 `outputs/skill-encoding-audit.md`를 생성합니다. jailbreak 방어 보고서를 바탕으로, 다루는 인코딩 공격 계열(ASCII 아트, base64, leet-speak, UTF-8 동형 문자, UTES)과 각각을 포착하는 방어 계층을 나열합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 은폐된 문자열이 간단한 키워드 필터를 통과하는지 검증하세요. 필요한 문자 단위 변경 사항을 보고하세요.

2. 동일한 대상 단어에 대해 두 번째 인코딩(base64)을 구현하세요. ArtPrompt와 비교하여 필터 우회율 및 복원 난이도를 비교하세요.

3. Jiang et al. 2024의 4.3절(5개 모델 결과)을 읽으세요. 동일한 벤치마크에서 Claude의 ArtPrompt 저항성이 Gemini보다 높은 이유를 제안하세요.

4. 프롬프트에서 ASCII 아트 형태의 영역을 감지하는 생성 전(pre-generation) 방어 전략을 설계하세요. 합법적인 코드, 표 및 수학적 표기법에 대한 오탐(false-positive)률을 측정하세요.

5. StructuralSleight는 10가지 인코딩 구조를 나열합니다. 모든 10가지를 처리하는 일반화된 방어 전략을 스케치하고, 방어된 프롬프트당 연산 비용을 추정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| ArtPrompt | "ASCII 아트 공격" | 안전 관련 단어를 ASCII 아트 렌더링으로 마스킹하는 2단계 jailbreak |
| Cloaking | "단어 숨기기" | 금지된 토큰을 모델은 읽지만 필터는 읽지 못하는 시각적 표현으로 대체 |
| UTES | "드문 구조" | Uncommon Text-Encoded Structure — 콘텐츠 밀반입을 위해 사용되는 트리, 그래프, 중첩 JSON 등 |
| ViTC | "시각-텍스트 능력" | 모델이 비의미적 시각 인코딩을 읽는 능력을 측정하는 벤치마크 |
| Perplexity filter | "PPL 방어" | 퍼플렉시티(perplexity)가 높은 프롬프트를 거부; 합법적인 구조화된 입력도 높은 점수를 받기 때문에 실패 |
| Retokenization | "토크나이저 전환 방어" | 다른 토크나이저로 프롬프트를 전처리; 인식은 시각적이기 때문에 실패 |
| 동형 문자 | "유사 문자" | 라틴 문자와 동일해 보이는 유니코드 문자; 부분 문자열 검사 우회 |

## 추가 읽기

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII 아트 제일브레이크 논문
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 일반화
- [Chao et al. — PAIR (12강, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 보완적 반복 공격
- [Anil et al. — Many-shot Jailbreaking (13강)](https://www.anthropic.com/research/many-shot-jailbreaking) — 보완적 길이 공격
