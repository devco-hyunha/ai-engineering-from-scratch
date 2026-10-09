# 캡스톤 15 — 헌법적 안전 하네스 + 레드팀 레인지

> Anthropic의 Constitutional Classifiers, Meta의 Llama Guard 4, Google의 ShieldGemma-2, NVIDIA의 Nemotron 3 Content Safety, 그리고 다국어 커버리지용 X-Guard가 2026년 안전 분류기 스택을 정의했습니다. garak, PyRIT, NVIDIA Aegis, promptfoo는 표준 적대적 평가 도구가 되었습니다. NeMo Guardrails v0.12는 이를 프로덕션 파이프라인으로 통합합니다. 이 캡스톤은 모든 요소를 연결합니다: 타겟 앱 주변에 계층화된 안전 하네스를 구축하고, 6가지 이상의 공격 계열을 실행하는 자율 레드팀 에이전트를 운영하며, 측정 가능한 무해성 델타를 생성하는 헌법적 자기비판 실행을 수행합니다.

**유형:** Capstone
**언어:** Python (안전 파이프라인, 레드팀), YAML (정책 구성)
**선수 요건:** 10단계 (LLM을 처음부터 구축), 11단계 (LLM 엔지니어링), 13단계 (도구), 14단계 (에이전트), 18단계 (윤리, 안전, 정렬)

**활용 단계:** P10 · P11 · P13 · P14 · P18
**시간:** 25시간

## 문제점

2026년 LLM 안전의 최전선은 분류기가 작동하는지 여부(대략 작동합니다)가 아니라, 과잉 거절이나 명백한 허점을 남기지 않으면서 프로덕션 앱 주변에 이를 올바르게 구성하는 방법입니다. Llama Guard 4는 영어 정책 위반을 처리합니다. X-Guard (132개 언어)는 다국어 제일브레이크를 처리합니다. ShieldGemma-2는 이미지 기반 프롬프트 인젝션을 포착합니다. NVIDIA Nemotron 3 Content Safety는 엔터프라이즈 카테고리를 커버합니다. Anthropic의 Constitutional Classifiers는 서빙 중이 아닌 학습 중에 사용되는 별도의 접근법입니다.

공격 진화도 중요합니다. PAIR와 TAP은 제일브레이크 발견을 자동화합니다. GCG는 기울기 기반 접미사 공격을 실행합니다. 다중 턴 및 코드 전환 공격은 에이전트 메모리를 악용합니다. 배포된 모든 LLM은 레드팀 레인지(garak과 PyRIT가 표준 드라이버입니다)와 문서화된 완화 조치, CVSS 점수가 매겨진 발견 사항이 필요합니다.

타겟 애플리케이션(8B 지시 미세 조정 모델 또는 다른 캡스톤의 RAG 챗봇 중 하나)을 강화하고, 6가지 이상의 공격 계열을 실행하며, 전후 무해성 측정 결과를 생성합니다.

## 개념

안전 파이프라인은 5개의 계층으로 구성됩니다. **입력 정화**: 제로 폭 문자 제거, base64/rot13 디코딩, 유니코드 정규화. **정책 계층**: NeMo Guardrails v0.12 레일 (도메인 이탈, 독성, PII 추출). **분류기 게이트**: 입력에는 Llama Guard 4, 비영어권에는 X-Guard, 이미지 입력에는 ShieldGemma-2. **모델**: 대상 LLM. **출력 필터**: 출력에는 Llama Guard 4, Presidio PII 스크럽, 해당되는 경우 인용 강제. **HITL 계층**: 고위험으로 플래그된 출력은 Slack 큐로 전송됩니다.

레드팀 범위는 스케줄러에서 실행됩니다. PAIR와 TAP은 제일브레이크를 자율적으로 발견합니다. GCG는 기울기 기반 접미사 공격을 실행합니다. ASCII / base64 / rot13 인코딩 공격. 다중 턴 공격 (페르소나 채택, 메모리 활용). 코드 전환 공격 (영어와 스와힐리어 또는 태국어 혼합). 각 실행은 CVSS 점수와 공개 타임라인이 포함된 구조화된 발견 파일 생성합니다.

헌법적 자기 비판 실행은 학습 시간 중개입입니다. 1k개의 유해 시도 프롬프트를 가져와 모델이 응답을 작성하게 하고, 작성된 헌법 (해치지 마세요 규칙)에 대해 비판하게 한 후, 비판 루프에 대해 재학습합니다. 홀드아웃 평가에서 전후 무해성 델타를 측정합니다.

## 아키텍처

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 스택

- 안전 분류기: Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard
- 가드레일 프레임워크: NeMo Guardrails v0.12 + OPA
- 레드팀 드라이버: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- 제일브레이크 에이전트: PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), GCG 접미사
- 헌법적 학습: Anthropic 스타일 자기 비판 루프 + 비판에 대한 SFT
- PII 스크럽: Presidio
- 대상: 8B 지시문 미세 조정 모델 또는 다른 캡스톤의 RAG 챗봇 중 하나

```figure
cf-safety-stack
```

## 구현하기

1. **대상 설정.** vLLM에서 8B 지시문 미세 조정 모델을 구축합니다 (또는 다른 캡스톤의 RAG 챗봇을 재사용합니다). 이것이 테스트 대상 앱입니다.

2. **안전 파이프라인 래핑.** 대상을 둘러싼 5계층 파이프라인을 연결합니다. 각 계층이 개별적으로 관측 가능한지 확인합니다 (Langfuse에서 계층별 스팬).

3. **분류기 커버리지.** Llama Guard 4, X-Guard (다국어), ShieldGemma-2 (이미지)를 로드합니다. 작은 레이블 지정된 세트에서 각각 실행하여 기준선을 설정합니다.

4. **레드팀 스케줄러.** garak, PyRIT, PAIR 에이전트, TAP 에이전트, GCG 실행기, 다중 턴 공격자, 코드 전환(code-switch) 공격자를 각각 별도의 큐에서 실행하도록 스케줄링합니다.

5. **공격 스위트.** 6가지 공격 계열: (1) PAIR 자동화 제일브레이크, (2) TAP 트리 오브 어택(tree-of-attacks), (3) GCG 기울기 접미사, (4) ASCII / base64 / rot13 인코딩, (5) 다중 턴 페르소나, (6) 다국어 코드 전환. 계열별 성공률을 보고합니다.

6. **헌법적 자기 비판.** 1,000개의 유해 시도 프롬프트를 큐레이션합니다. 각 프롬프트에 대해 대상 모델이 응답을 작성합니다. 비평가 LLM이 작성된 헌법("해로움을 끼치지 말 것", "증거를 인용할 것", "불법 요청을 거부할 것")에 따라 점수를 매깁니다. 비평가가 이의를 제기하는 프롬프트는 재작성되며, 대상 모델은 비평이 개선된 쌍으로 미세 조정합니다. 보존된 평가 세트에서 전후 무해성을 측정합니다.

7. **과도한 거부 측정.** 온건한 프롬프트 스위트(예: XSTest)에서 오탐(false-positive)률을 추적합니다. 대상 모델은 온건한 질문에 대해 유용성을 유지해야 합니다.

8. **CVSS 점수 매기기.** 성공한 각 제일브레이크에 대해 CVSS 4.0(공격 벡터, 복잡도, 영향)으로 점수를 매깁니다. 공개 타임라인과 완화 계획을 작성합니다.

9. **레인지 자동화.** 위의 모든 작업은 cron에서 실행되며, 발견 사항은 큐에 기록되고, 과도한 거부 회귀(regression) 알림은 Slack으로 전송됩니다.

## 사용하기

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 출시하기

`outputs/skill-safety-harness.md`는 산출물입니다. 프로덕션급 계층형 안전 파이프라인과 전후 무해성 델타를 포함하는 재현 가능한 레드팀 레인지가 포함됩니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 공격 표면 커버리지 | 6개 이상의 공격 계열 실행, 2개 이상의 언어 |
| 20 | 진양성/오탐 트레이드오프 | 공격 차단률 vs XSTest 온건한 통과율 |
| 20 | 자기 비판 델타 | 보존된 평가 세트에서의 전후 무해성 |
| 20 | 문서화 및 공개 | 타임라인이 포함된 CVSS 점수화된 발견 사항 |
| 15 | 자동화 및 반복성 | 모든 작업이 cron에서 실행되며 알림이 전송됨 |
| **100** | | |

## 연습 문제

1. RAG 챗봇에 garak의 프롬프트 인젝션 플러그인을 실행하고, 출력 필터 레이어가 있을 때와 없을 때의 공격 성공률을 비교해 보세요.

2. 일곱 번째 공격 계열을 추가하세요: 검색된 문서를 통한 간접 프롬프트 주입(Indirect Prompt Injection). 추가적으로 필요한 방어 수준을 측정해 보세요.

3. "거절과 도움(refuse-with-help)" 모드를 구현하세요: 가드레일(Guardrails)이 차단할 경우, 대상 모델은 단순한 거절 대신 더 안전한 관련 답변을 제시합니다. XSTest 델타를 측정해 보세요.

4. 다국어 커버리지 격차: X-Guard가 성능이 떨어지는 언어를 찾아보세요. 해당 언어를 타겟으로 한 미세 조정(Fine-tuning) 데이터셋을 제안하세요.

5. 30B 모델에서 헌법적 자기 비평(constitutional self-critique)을 실행하고 델타가 확장되는지 측정해 보세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 계층적 안전 | "심층 방어(Defense in Depth)" | 입력, 게이트, 출력, 인간 개입 루프 (HITL)(Human-in-the-Loop (HITL))에 여러 가드레일(Guardrails)을 배치 |
| Llama Guard 4 | "Meta의 안전 분류기" | 2026년 기준 입력/출력 콘텐츠 분류기 |
| PAIR | "제일브레이크(Jailbreak) 에이전트" | LLM 기반 제일브레이크(Jailbreak) 발견에 관한 논문 (Chao et al.) |
| TAP | "Tree-of-Attacks" | PAIR의 트리 검색(Tree-search) 변형 |
| GCG | "Greedy coordinate gradient" | 기울기 기반 적대적 접미사(adversarial suffix) 공격 |
| 헌법적 자기 비평 | "Anthropic 스타일 훈련" | 대상 초안 작성 -> 비평가 점수 매기기 -> 재작성 -> 재훈련 |
| XSTest | "양성(probe) 세트" | 과잉 거절(over-refusal) 회귀 테스트를 위한 벤치마크 |
| CVSS 4.0 | "심각도 점수" | 안전 관련 발견 사항에 대한 표준 취약점 점수 체계 |

## 추가 읽기

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 훈련 시점 참고 자료
- [Meta Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 2026년 입력/출력 분류기
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 이미지 및 멀티모달 안전
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 기업용 참고 자료
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132개 언어 다국어 안전
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 레드 티밍(Red Teaming) 도구 키트
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 레드 티밍(Red Teaming) 프레임워크
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail 프레임워크
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 제일브레이크(Jailbreak) 에이전트 논문
