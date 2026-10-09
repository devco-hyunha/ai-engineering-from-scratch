# 왜 멀티 에이전트인가?

> 단일 에이전트는 한계에 부딪힙니다. 현명한 선택은 더 큰 에이전트가 아니라, 더 많은 에이전트입니다.

**유형:** Learn
**언어:** TypeScript
**선수 요건:** 14단계 (에이전트 엔지니어링)
**시간:** 약 60분

## 학습 목표

- 단일 에이전트의 한계(컨텍스트 오버플로, 혼합 전문성, 순차적 병목)를 식별하고, 여러 에이전트로 분할하는 것이 적절한 시점을 설명해 보세요
- 오케스트레이션 패턴(파이프라인, 병렬 팬아웃, 슈퍼바이저, 계층적)을 비교하고, 주어진 작업 구조에 적합한 패턴을 선택해 보세요
- 명확한 역할 경계, 공유 상태, 통신 계약을 갖춘 멀티 에이전트 시스템을 설계해 보세요
- 멀티 에이전트 복잡성(지연, 비용, 디버깅 난이도)과 단일 에이전트 단순성 간의 트레이드오프를 분석해 보세요

## 문제점

14단계에서 단일 에이전트를 구축했습니다. 이 에이전트는 작동합니다. 파일을 읽고, 명령을 실행하고, API를 호출하고, 결과를 추론할 수 있습니다. 그런 다음 실제 코드베이스에 적용해 보세요: 200개 파일, 세 가지 언어, 인프라에 의존하는 테스트, 그리고 코드를 작성하기 전에 외부 API를 조사해야 하는 요구 사항이 있습니다.

에이전트가 막힙니다. LLM가 멍청해서가 아니라, 작업이 단일 에이전트 루프가 처리할 수 있는 범위를 초과하기 때문입니다. 컨텍스트 윈도우가 파일 내용으로 가득 찹니다. 에이전트는 40번의 도구 호출 전에 읽은 내용을 잊어버립니다. 연구자, 코더, 리뷰어를 동시에 하려 하고, 세 가지를 모두 잘하지 못합니다.

이것이 단일 에이전트의 한계입니다. 작업이 다음을 요구할 때마다 이 한계에 부딪힙니다:

- **하나의 윈도우에 담기에는 너무 많은 컨텍스트** - 50개 파일을 읽으면 200k 토큰을 초과합니다
- **단계별로 다른 전문성** - 연구는 코드 생성과 다른 프롬프트가 필요합니다
- **병렬로 수행할 수 있는 작업** - 세 파일을 순차적으로 읽는 것보다 동시에 읽는 것이 왜 더 좋을까요?

## 개념

### 단일 에이전트의 한계

단일 에이전트는 하나의 루프, 하나의 컨텍스트 윈도우, 하나의 시스템 프롬프트입니다. 이를 상상해 보세요:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

세 가지가 무너집니다:

1. **컨텍스트 포화** - 도구 결과가 쌓입니다. 30번째 턴까지 에이전트(Agent)는 파일 내용, 명령 출력, 이전 추론으로 150k 토큰(Token)을 소비했습니다. 5번째 턴의 중요한 세부 사항이 사라집니다.

2. **역할 혼란** - "당신은 연구자, 코더, 리뷰어, 테스터입니다"라고 말하는 시스템 프롬프트(System Prompt)는 반쯤 연구하고, 반쯤 코딩하며, 리뷰를 끝내지 못하는 에이전트(Agent)를 생성합니다.

3. **순차적 병목** - 에이전트(Agent)는 파일 A를 읽고, 파일 B를 읽고, 파일 C를 읽습니다. 3개의 순차적 LLM (대규모 언어 모델)(LLM (Large Language Model)) 호출. 3개의 순차적 도구 실행. 병렬 처리가 없습니다.

### 멀티 에이전트(Multi-Agent) 솔루션

작업을 분할하세요. 각 에이전트(Agent)에 하나의 작업, 하나의 컨텍스트 윈도우(Context Window), 그리고 그 작업에 맞춰 조정된 하나의 시스템 프롬프트(System Prompt)를 부여하세요:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

각 에이전트(Agent)는 다음을 가집니다:
- 집중된 시스템 프롬프트(System Prompt) ("당신은 코드 리뷰어입니다. 당신의 유일한 작업은 버그를 찾는 것입니다.")
- 자체 컨텍스트 윈도우(Context Window) (다른 에이전트(Agent)의 작업으로 오염되지 않음)
- 명확한 입력/출력 계약(Tool Contract) (연구 노트를 받고, 코드를 출력함)

### 이것을 수행하는 실제 시스템

**Claude Code 하위 에이전트(Subagents)** - Claude Code가 `Task`로 하위 에이전트(Subagents)를 생성할 때, 범위가 지정된 작업을 가진 자식 에이전트(Agent)를 만듭니다. 부모는 컨텍스트를 깨끗하게 유지합니다. 자식은 집중된 작업을 수행하고 요약본을 반환합니다.

**Devin** - 계획 에이전트(Agent), 코더 에이전트(Agent), 브라우저 에이전트(Agent)를 실행합니다. 계획 에이전트(Agent)는 작업을 단계로 분해합니다. 코더 에이전트(Agent)는 코드를 작성합니다. 브라우저 에이전트(Agent)는 문서를 조사합니다. 각각이 별도의 컨텍스트를 가집니다.

**멀티 에이전트(Multi-Agent) 코딩 팀 (SWE-bench)** - SWE-bench에서 최고 성능을 내는 시스템은 코드베이스를 읽는 연구자, 수정을 설계하는 계획자, 이를 구현하는 코더를 사용합니다. 단일 에이전트(Agent) 시스템은 점수가 낮습니다.

**ChatGPT Deep Research** - 여러 검색 에이전트(Agent)를 병렬로 생성하여 각각 다른 각도를 탐색한 후, 결과를 종합합니다.

### 스펙트럼

멀티 에이전트(Multi-Agent)는 이분법적이지 않습니다. 이는 스펙트럼입니다:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**단일 에이전트(Agent)** - 하나의 루프, 하나의 프롬프트. 단순한 작업에 적합합니다.

**하위 에이전트(Subagents)** - 부모가 집중된 하위 작업을 위해 자식을 생성합니다. 부모는 계획을 유지합니다. 자식들이 보고합니다. 이것이 Claude Code가 하는 방식입니다.

**파이프라인(Pipeline)** - 에이전트(Agent)가 순차적으로 실행됩니다. 에이전트 A의 출력이 에이전트 B의 입력이 됩니다. 단계별 워크플로우에 적합합니다: 연구 -> 코드 -> 리뷰 -> 테스트.

**팀(Team)** - 에이전트(Agent)가 공유 메시지 버스를 통해 병렬로 실행됩니다. 각 에이전트에는 역할이 있습니다. 오케스트레이터(Orchestrator)가 조정합니다. 동시에 다양한 스킬이 필요할 때 적합합니다.

**스웜(Swarm)** - 공유 상태를 가진 많은 동일하거나 거의 동일한 에이전트(Agent)가 있습니다. 고정된 오케스트레이터(Orchestrator)가 없습니다. 에이전트(Agent)는 큐에서 작업을 가져옵니다. 높은 처리량의 병렬 작업에 적합합니다.

### 네 가지 멀티 에이전트 패턴

#### 패턴 1: 파이프라인(Pipeline)

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

각 에이전트(Agent)가 데이터를 변환하고 다음 단계로 전달합니다. 이해하기 쉽습니다. 한 단계에서 실패하면 나머지가 차단됩니다.

#### 패턴 2: 팬아웃 / 팬인(Fan-out / Fan-in)

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

작업을 병렬 에이전트(Agent)로 분할한 후 결과를 병합합니다. 독립적인 하위 작업으로 분해되는 작업에 적합합니다.

#### 패턴 3: 오케스트레이터-워커(Orchestrator-Worker)

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

스마트한 오케스트레이터(Orchestrator)가 무엇을 할지 결정하고, 워커(Worker)에게 위임하며, 결과를 종합합니다. 오케스트레이터(Orchestrator)는 워커(Worker)를 생성하기 위한 도구를 가진 에이전트(Agent) 그 자체입니다.

#### 패턴 4: 피어 스웜(Peer Swarm)

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

중앙 오케스트레이터(Orchestrator)가 없습니다. 에이전트(Agent)가 피어 투 피어(peer-to-peer)로 통신합니다. 결정은 상호작용에서 발생합니다. 디버깅이 어렵지만, 많은 에이전트(Agent)로 확장 가능합니다.

### 멀티 에이전트(Multi-Agent)를 사용하지 말아야 할 때

멀티 에이전트(Multi-Agent)는 복잡성을 추가합니다. 에이전트(Agent) 간의 모든 메시지는 잠재적인 실패 지점입니다. 디버깅이 "하나의 대화 읽기"에서 "다섯 에이전트(Agent)에 걸친 메시지 추적"으로 바뀝니다.

**단일 에이전트(Single-Agent)로 유지해야 할 때:**
- 작업이 하나의 컨텍스트 윈도우(Context Window)에 fits (작업 데이터 약 100k 토큰(Token) 미만)
- 단계별로 다른 시스템 프롬프트(System Prompt)가 필요하지 않습니다
- 순차적 실행이 충분히 빠릅니다
- 작업이 단순하여 분할하면 가치보다 오버헤드가 더 큽니다

**복잡성 비용:**
- 모든 에이전트(Agent) 경계는 손실 있는 압축 단계입니다: 에이전트 A의 전체 컨텍스트가 에이전트 B를 위한 메시지로 요약됩니다
- 조정 로직(누가, 무엇을, 언제, 어떤 순서로 수행하는지)은 그 자체로 버그의 원천입니다
- 지연이 증가합니다: N개의 에이전트(Agent)는 최소 N개의 직렬 LLM (대규모 언어 모델)(LLM (Large Language Model)) 호출을 의미하며, 상호 통신이 필요하면 더 많아집니다
- 비용이 곱해집니다: 각 에이전트(Agent)가 독립적으로 토큰(Token)을 소모합니다

경험칙: 작업이 20회 미만의 도구 호출(Function Calling)로 완료되고 100k 토큰(Token) 내에 Fits한다면, 단일 에이전트(Agent)로 유지해 보세요.

```figure
swarm-messages
```

## 구현하기

### 1단계: 과부하된 단일 에이전트(Agent)

모든 작업을 수행하려는 단일 에이전트(Agent)가 있습니다. 거대한 시스템 프롬프트(System Prompt)와 연구, 코드, 리뷰를 담고 있는 하나의 컨텍스트 윈도우(Context Window)를 가지고 있습니다:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

이 접근 방식의 문제점:
- 컨텍스트 윈도우(Context Window)가 모든 단계에서 증가합니다. 리뷰 단계에서는 연구 노트와 코드, 그리고 이전 추론(Inference)이 모두 포함됩니다.
- 시스템 프롬프트(System Prompt)가 일반적입니다. 각 단계에 맞춰 조정할 수 없습니다.
- 병렬로 실행되는 것이 없습니다.

### 2단계: 전문가 에이전트(Agent)

이제 분리해 보세요. 각 에이전트(Agent)는 하나의 작업만 수행합니다:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

각 전문가 에이전트(Agent)는 집중된 프롬프트를 가집니다. 각 에이전트(Agent)는 필요한 입력만 담은 깨끗한 컨텍스트 윈도우(Context Window)를 받습니다.

### 3단계: 메시지 전달로 조정하기

명시적인 메시지 전달로 전문가 에이전트(Agent)들을 연결합니다:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

각 에이전트(Agent)는 자신에게 보내진 메시지만 받습니다. 컨텍스트 오염이 없습니다. 연구자의 50k 토큰(Token) 문서 읽기 내용은 리뷰어의 컨텍스트에 들어가지 않습니다.

### 4단계: 비교하기

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

멀티 에이전트(Agent) 버전은 더 많은 총 토큰(Token)을 사용합니다(세 개의 에이전트, 세 개의 별도 LLM (대규모 언어 모델) 호출)하지만, 각 에이전트(Agent)의 컨텍스트는 깨끗하게 유지됩니다. 시스템 프롬프트(System Prompt)가 전문화되어 각 단계의 품질이 향상됩니다.

## 사용하기

이 강의는 멀티 에이전트(Agent)로 전환할 시기를 결정하기 위한 재사용 가능한 프롬프트를 생성합니다. `outputs/prompt-multi-agent-decision.md`를 참고하세요.

## 연습 문제

1. 네 번째 전문가를 추가하세요: "테스터" 에이전트(Agent)는 코더로부터 코드를 받고 리뷰어로부터 리뷰 피드백을 받은 후 테스트를 작성합니다
2. 리뷰어가 수정 루프(최대 2라운드)를 위해 피드백을 코더에게 보낼 수 있도록 파이프라인을 수정하세요
3. 순차적 파이프라인을 팬아웃(fan-out)으로 변환하세요: 연구자와 "요구사항 분석가" 에이전트(Agent)를 병렬로 실행한 후, 코더에게 전달하기 전에 출력물을 병합하세요

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 스웜(Swarm) | "AI 에이전트의 군집 지능" | 공유된 상태를 가지며 고정된 리더가 없는 동등한 에이전트 집합. 행동은 지역적 상호작용에서 발생합니다. |
| 오케스트레이터(Orchestrator) | "상사 에이전트" | 다른 에이전트를 생성하고 관리하는 도구를 가진 에이전트. 계획을 수립하고 위임하지만 실제 작업을 수행하지는 않을 수 있습니다. |
| 코디네이터(Coordinator) | "교통 경찰" | 규칙에 기반하여 에이전트 간 메시지를 라우팅하는 비에이전트 구성 요소(종종 LLM이 아닌 단순 코드). |
| 합의(Consensus) | "에이전트들이 동의함" | 진행하기 전에 여러 에이전트가 합의에 도달해야 하는 프로토콜. 충돌하는 출력의 해결이 필요할 때 사용됩니다. |
| 창발적 행동(Emergent behavior) | "에이전트들이 스스로 알아냄" | 에이전트 상호작용에서 발생하지만 명시적으로 프로그래밍되지 않은 시스템 수준의 패턴. 유용하거나 해로울 수 있습니다. |
| 팬아웃 / 팬인(Fan-out / fan-in) | "에이전트를 위한 맵리듀스" | 작업을 병렬 에이전트들에게 분할(팬아웃)한 후, 그 결과를 결합(팬인)하는 방식. |
| 메시지 패싱(Message passing) | "에이전트들이 서로 대화함" | 에이전트 간 통신 메커니즘: 한 에이전트에서 다른 에이전트로 전송되는 구조화된 데이터로, 공유 컨텍스트 윈도우를 대체합니다. |

## 추가 읽기

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 다중 에이전트 패턴에 대한 조사
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft의 다중 에이전트 대화 프레임워크
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code가 Task로 위임하는 방법
- [CrewAI documentation](https://docs.crewai.com/) - 역할 기반 다중 에이전트 프레임워크
