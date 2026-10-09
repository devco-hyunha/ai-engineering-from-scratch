# 캡스톤 17 — 개인 AI 튜터 (적응형, 멀티모달, 메모리 포함)

> Khanmigo (Khan Academy), Duolingo Max, Google LearnLM / Gemini for Education, Quizlet Q-Chat, Synthesis Tutor는 모두 2026년에 대규모 적응형 멀티모달 튜팅을 출시했습니다. 공통적인 형태는 소크라테스식 정책(정답을 단순히 제시하지 않음), 모든 상호작용 후 업데이트되는 학습자 모델(베이즈 지식 추적 스타일), 음성 + 텍스트 + 사진 수학 입력, 커리큘럼 그래프 검색, 간격 반복 스케줄링, 그리고 연령에 적합한 콘텐츠에 대한 엄격한 안전 필터입니다. 이 캡스톤의 목표는 특정 과목(K-12 대수 또는 입문 Python) 튜터를 출시하고, 10명의 학습자를 대상으로 2주간의 효능 연구를 수행하며, 콘텐츠 안전 감사에 통과하는 것입니다.

**유형:** Capstone
**언어:** Python (백엔드, 학습자 모델), TypeScript (웹 앱), SQL (Postgres + Neo4j를 통한 커리큘럼 그래프)
**선수 요건:** 5단계 (NLP), 6단계 (음성), 11단계 (LLM 엔지니어링), 12단계 (멀티모달), 14단계 (에이전트), 17단계 (인프라), 18단계 (안전)

**활용 단계:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**시간:** 30시간

## 문제점

적응형 튜팅은 과거에 교육 기술 연구의 틈새 영역이었습니다. 2026년에는 소비자 제품이 되었습니다. Khanmigo는 대부분의 미국 학교 구역에 배포되었습니다. Duolingo Max는 수천만 명의 월간 활성 사용자(MAU)를 달성했습니다. Google의 LearnLM / Gemini for Education은 Google Classroom의 튜팅을 지원합니다. Quizlet Q-Chat은 플래시카드와 함께 자리 잡았습니다. Synthesis Tutor는 호기심 많은 아이들을 위한 튜터로 바이럴에 성공했습니다. 공통 요소는 다음과 같습니다: 멀티모달 입력(타입, 말하기, 방정식 사진), 소크라테스식 교수법(먼저 질문하고, 나중에 설명), 각 상호작용 후 업데이트되는 학습자 모델, 그리고 엄격한 연령 적합 안전성.

특정 코호트용 튜터 중 하나를 구축할 것입니다. 측정 기준은 실제 효능 연구입니다: 10명의 학습자를 대상으로 2주간의 사전 테스트 및 사후 테스트 점수. 음성 루프는 자연스러워야 합니다(캡스톤 03 하위 스택). 메모리는 프라이버시를 존중해야 합니다. 안전 필터는 K-12용 COPPA 기반 레드 티밍을 통과해야 합니다.

## 개념

네 가지 구성 요소입니다. **튜터 정책**은 소크라테스식 루프입니다. 학습자가 정답을 요청하면 정책이 유도 질문을 던지고, 정답을 맞히면 다음 개념으로 이동하며, 막히면 단계적 힌트를 제공합니다. **학습자 모델**은 각 상호작용 후 커리큘럼 노드별 숙련도 확률을 업데이트하는 베이지안 지식 추적(Bayesian Knowledge Tracing)(또는 단순 변형)입니다. **커리큘럼 그래프**는 선결 조건 간선을 가진 개념들의 Neo4j 그래프이며, 정책은 이 그래프를 순회하여 다음 개념을 선택합니다. **메모리**는 agentmemory 스타일의 일화 기억 + 의미론적 저장소로, 과거 상호작용, 실수, 선호도를 보관합니다.

UX는 멀티모달입니다. 타이핑된 답변을 위한 텍스트 입력, LiveKit + Whisper를 통한 음성 입력(캡스톤 03 재사용), dots.ocr 또는 PaliGemma 2를 통한 수학 문제 사진 입력, Cartesia Sonic-2를 통한 음성 출력. 안전성은 Llama Guard 4와 연령에 적합한 필터(성인 콘텐츠, 폭력, 자해 차단) 및 COPPA 기반 메모리 보존 정책을 사용합니다.

효능 연구가 산출물입니다. 학습자 10명, 사전/사후 테스트, 2주 기간. 학습 향상 델타와 신뢰 구간을 보고합니다. 비적응형 기준선(튜터 정책 없이 동일한 콘텐츠를 선형으로 전달)과 비교합니다.

## 아키텍처

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 스택

- 주제 선택: K-12 대수 또는 입문 Python (깊이 있는 학습을 위해 하나 선택)
- 튜터 정책: LangGraph + Claude Sonnet 4.7 (프롬프트 캐싱 포함)
- 학습자 모델: 베이지안 지식 추적(클래식) 또는 간격을 위한 FSRS
- 커리큘럼 그래프: 개념 + 선결 조건 간선 + OER 콘텐츠의 Neo4j
- 메모리: agentmemory 스타일의 지속 가능한 벡터 + 일화 기억 + 의미론적 저장소
- 음성: LiveKit Agents 1.0 + Cartesia Sonic-2 (캡스톤 03 하위 스택 재사용)
- 사진 수학: 방정식 인식을 위한 dots.ocr 또는 PaliGemma 2
- 안전성: Llama Guard 4 + 맞춤형 연령 적합 필터
- 평가: 블룸 수준 질문 생성, 사전/사후 테스트 하네스, 효능 연구 도구

```figure
cf-tutor-loop
```

## 구현하기

1. **커리큘럼 그래프.** 선결 조건 간선을 가진 50-150개 개념 노드(예: "수직선"부터 "이차 방정식 공식"까지의 K-12 대수)의 Neo4j 그래프를 구축합니다. 각 노드에 OER 콘텐츠(Open Textbook, OpenStax)를 첨부합니다.

2. **학습자 모델.** 베이즈 지식 추적(Bayesian Knowledge Tracing)을 사전 확률: 추측, 실수, 학습률로 초기화합니다. 각 상호작용 후 개념별 숙련도를 업데이트합니다. 학습자별로 저장합니다.

3. **튜터 정책.** LangGraph의 노드: `read_signal` (학습자의 정답/부분 정답/막힘 여부?), `select_concept` (커리큘럼 그래프에서 우선순위가 가장 높은 개념 선택), `scaffold` (소크라테스식 프롬프트), `update_mastery`.

4. **메모리.** 모든 상호작용은 일화 기억(episodic store)에 기록합니다. 실수와 선호도는 의미 기억(semantic memory)으로 승격됩니다. COPPA 준수 보존 정책: 1년 후 자동 삭제, 부모 접근 가능.

5. **음성 경로.** 튜터 정책에 연결된 LiveKit Agents 워커. ASR은 Whisper-v3-turbo를 통해 수행합니다. TTS는 Cartesia Sonic-2를 통해 수행합니다. 바지인(barge-in)이 지원됩니다 (캡스톤 03 메커니즘 재사용).

6. **사진 수학 경로.** 이미지를 업로드하거나 촬영하여 dots.ocr 또는 PaliGemma 2로 방정식을 인식하고, 구조화된 입력으로 튜터에 전달합니다.

7. **안전.** 모든 모델 출력은 Llama Guard 4 + 연령 적절 필터(자해, 성인 콘텐츠, 폭력 차단)를 통과합니다. 메모리 접근은 학습자 ID로 범위가 지정되며, 삭제에 대한 부모 접근 인터페이스가 있습니다.

8. **효능 연구.** 학습자 10명, 사전 테스트(표준화된 30문항 기준선), 2주간의 튜터 상호작용(주 3회 세션), 사후 테스트. 동일한 콘텐츠의 비적응형 기준선 코호트 학습자 10명과 비교합니다.

9. **주간 진행 보고서.** 학습자별로 탐색한 주제, 숙련도 궤적, 권장 다음 단계를 요약한 PDF를 자동 생성합니다.

## 사용하기

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 출시하기

`outputs/skill-ai-tutor.md`는 산출물입니다. 멀티모달 입력, 학습자 모델, 메모리, 안전성 및 측정된 효능을 갖춘 과목별 적응형 튜터입니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 학습 향상 델타 | 10명 학습자 2주 연구의 사전/사후 테스트 델타 |
| 20 | 소크라테스식 충실도 | 전사(transcript) 샘플에 대한 루브릭 점수 |
| 20 | 멀티모달 UX | 음성 + 사진 + 텍스트의 엔드투엔드 일관성 |
| 20 | 안전 + 프라이버시 태세 | Llama Guard 4 통과율 + COPPA 준수 보존 |
| 15 | 커리큘럼 범위 및 그래프 품질 | 개념 커버리지 + 선결 조건 그래프 일관성 |
| **100** | | |

## 연습 문제

1. 적응형 학습자 모델(무작위 개념 순서)을 포함하여 포함하지 않은 상태로 효능 연구를 실행하세요. 델타를 보고하세요. 적응형이 승리할 것으로 예상되지만, 그 크기가 흥미로운 수치입니다.

2. 멀티모달 프로브를 추가하세요: 동일한 개념 질문을 텍스트, 음성, 사진으로 전달합니다. 학습자가 선호하는 모달리티를 사용할 때 더 빠르게 수렴하는지 측정하세요.

3. 부모 대시보드를 구축하세요: 연습한 주제, 숙련도 궤적, 다가오는 개념, 안전 이벤트(가드레일 충돌 포함). COPPA에 정렬되도록 하세요.

4. 언어 전환 모드를 추가하세요: 튜터가 스페인어 입력을 받아 스페인어로 가르칩니다. X-Guard 커버리지를 측정하세요.

5. 메모리 프라이버시를 스트레스 테스트하세요: 음성 클립 재인제스트 공격을 통해 학습자 A가 학습자 B의 데이터를 볼 수 없음을 검증하세요. 시도된 접근을 기록하고 알림을 보내세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 소크라테스식 정책 | "묻고,_dump 하지 마세요" | 튜터가 답을 주지 않고 유도 질문을 합니다 |
| 베이지안 지식 추적 | "BKT" | 개념별 숙련도 확률에 대한 고전적인 학습자 모델 방정식 |
| FSRS | "Free Spaced Repetition Scheduler" | SM-2보다 나은 2024년 간격 반복 스케줄러 |
| 커리큘럼 그래프 | "Concept DAG" | 선결 조건 엣지를 가진 개념의 Neo4j |
| 일화 기억 | "Per-interaction log" | 나중에 검색하기 위해 모든 상호작용을 저장 |
| 의미 기억 | "Learned pattern store" | 일화 기억에서 승격된 압축된 실수와 선호도 |
| COPPA | "Kids privacy law" | 13세 미만 아동의 데이터 수집을 제한하는 미국 법률 |

## 추가 읽기

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 소비자 K-12 튜터 참조
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 언어 학습 튜터 참조
- [Google LearnLM / Gemini for Education](https://blog.google/products-and-platforms/products/education/google-learnlm-gemini-generative-ai/) — 호스팅된 참조 모델
- [Quizlet Q-Chat](https://quizlet.com) — 대체 참조
- [Synthesis Tutor](https://www.synthesis.com) — 스타트업 참조
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — 간격 반복 스케줄러
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 학습자 모델 고전
- [LiveKit Agents](https://github.com/livekit/agents) — 음성 스택
