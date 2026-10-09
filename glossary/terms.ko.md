# AI 엔지니어링 용어집

강의, 논문, 모델 카드, 코드 리뷰에서 용어가 설명보다 빠르게 등장할 때 이 용어집을 사용하십시오. 정확한 용어 또는 별칭으로 검색하고, 먼저 직접적인 정의를 읽은 후, 실용적인 노트를 활용하여 구축할 수 있는 시스템과 연결하십시오.

각 항목은 하나의 학습 범주에 속합니다. `Related terms`는 고정된 경로를 강제하지 않으면서 다음에 유용한 개념을 제시합니다. 정의는 일반적인 엔지니어링적 의미를 설명하지만, 제공자별 동작은 다를 수 있습니다. API 계약이나 모델 카드가 일반적인 정의와 일치하지 않는 경우, 최신 공식 문서가 우선합니다.

12가지 범주는 다음과 같습니다: 수학 및 훈련; 모델 및 추론; 데이터 및 표현; 검색 및 생성; 프롬프팅 및 컨텍스트; 에이전트 및 도구; 평가 및 안전; AI 네이티브 개발; 인프라 및 서빙; 신뢰성 및 운영; 보안 및 거버넌스; 멀티모달 시스템.

## A

### Activation Checkpointing - 활성화 체크포인팅
- **분류:** 수학 및 훈련
- **실제 의미:** 선택된 순방향 전파 활성값만 저장하고, 생략된 활성값은 역전파 중에 재계산하는 훈련 메모리 기법입니다.
- **왜 중요한가:** 추가 연산을 대가로 활성값 저장 공간을 줄여, 고정된 메모리 예산 내에서 더 큰 모델이나 시퀀스를 훈련할 수 있게 합니다.
- **실무에서는:** 메모리 사용량이 많은 트랜스포머 블록을 체크포인팅하고, 추가 스텝 시간을 측정하며, 복구 체크포인팅을 활성값 재계산 설정과 분리하여 유지하십시오.
- **흔한 혼동:** 활성화 체크포인팅은 영구적인 훈련 체크포인팅이 아닙니다. 하나의 순방향 및 역방향 전파가 메모리에 Fits하도록 돕지만, 충돌한 실행을 재개할 수는 없습니다.
- **관련 용어:** Autograd, Backpropagation, Checkpoint, Mixed Precision
- **출처:** [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)

### Activation Function - 활성화 함수
- **분류:** 수학 및 훈련
- **흔히 하는 말:** 레이어 사이의 비선형 연산.
- **실제 의미:** 선형 또는 아핀 레이어 이후에 적용되어 비선형성을 도입하는 함수입니다. 이 함수가 없으면 가중치와 편향을 가진 레이어들의 조합은 하나의 아핀 변환으로 축소됩니다. ReLU, GELU, SiLU가 일반적인 선택지입니다. 이 선택은 학습 중 그래디언트 흐름에 직접적인 영향을 미칩니다.
- **배울 곳:** [Activation Functions](../phases/03-deep-learning-core/04-activation-functions/)
- **관련 용어:** ReLU, Gradient, Backpropagation

### Adam (Optimizer) - Adam 옵티마이저
- **분류:** 수학 및 학습
- **흔히 하는 말:** 고민 없이 사용하는 옵티마이저입니다.
- **실제 의미:** Adaptive Moment Estimation입니다. 그래디언트의 지수 평균과 그래디언트 제곱의 지수 평균을 결합하고, 편향 보정을 적용하며, 매개변수별로 업데이트 스케일을 적응적으로 조정합니다. 유용한 기준선(baseline)이지만, 여전히 적절한 학습률과 스케줄이 필요합니다.
- **흔한 혼동:** Adam은 강력한 기준선일 뿐, 만능 최적의 옵티마이저가 아닙니다.
- **출처:** [Adam paper](https://arxiv.org/abs/1412.6980)
- **관련 용어:** AdamW, Optimizer, Learning Rate

### AdamW
- **분류:** 수학 및 학습
- **흔히 하는 말:** 가중치 감쇠(weight decay)가 수정된 Adam입니다.
- **실제 의미:** 가중치 감쇠를 그래디언트 기반 매개변수 업데이트와 분리(decouple)한 Adam 변형입니다. Adam의 적응적으로 스케일링된 그래디언트 내부에 L2 페널티를 추가하는 것보다 축소(shrinkage) 동작을 더 쉽게 이해할 수 있게 합니다.
- **흔한 혼동:** 분리된 가중치 감쇠(decoupled weight decay)가 AdamW를 만능 최적의 옵티마이저로 만들지는 않습니다. 모델, 데이터, 학습 규모가 여전히 최적의 옵티마이저와 스케줄을 결정합니다.
- **출처:** [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- **관련 용어:** Adam (Optimizer), Weight Decay, Optimizer

### Admission Control - 수용 제어
- **분류:** 신뢰성 및 운영
- **실제 의미:** 시스템의 현재 용량, 우선순위, 정책에 따라 요청이 제한된 큐나 서비스로 진입할 수 있는지 결정하는 사전 수용 게이트입니다.
- **왜 중요한가:** 통제된 경계에서 초과 작업을 거부하면 수용된 요청이 큐 증가, 타임아웃 연쇄, 자원 고갈로부터 보호됩니다.
- **실무에서는:** 요청의 비용을 추정하고, 테넌트 및 시스템 용량을 확인하며, 필요한 예산을 원자적으로 예약하고, 거부 시 과부하 범위를 식별합니다. 조건이 일시적이며 호출자의 재시도 예산이 추가 시도를 허용하는 경우에만 재시도 지침을 제공합니다.
- **흔한 혼동:** 어드미션 컨트롤은 수락 전에 작동합니다. 로드 셰딩은 진입점, 큐, 의존성, 또는 기타 과부하 경계에서 작업을 거부하거나 제거할 수 있습니다.
- **관련 용어:** Load Shedding, Backpressure, Rate Limit, Saturation
- **출처:** [Google SRE: Handling Overload](https://sre.google/sre-book/handling-overload/)

### Agent - 에이전트
- **분류:** 에이전트 및 도구
- **흔히 하는 말:** 단독으로 사고하고 행동하는 자율 모델.
- **실제 의미:** 모델이 목표를 향해 행동을 선택하고, 도구 또는 환경의 결과를 관찰하며, 오케스트레이션 정책 하에 계속 진행하도록 하는 소프트웨어 시스템입니다. 에이전트는 루프, 상태 머신, 워크플로우 엔진, 또는 인간 승인을 사용할 수 있습니다. 모델은 전체 시스템이 아니라 하나의 구성 요소입니다.
- **왜 중요한가:** 신뢰성은 모델 주변의 하네스, 도구 계약, 상태, 권한, 및 검증에서 나옵니다.
- **실무에서는:** 코딩 에이전트는 저장소 컨텍스트를 읽고, 패치를 제안하며, 샌드박스에서 테스트를 실행하고, 배포 전에 승인을 위해 멈춥니다.
- **흔한 혼동:** 자율성은 위임된 권한의 정도이며, 모든 에이전트의 필수 속성이 아닙니다.
- **배울 곳:** [The Agent Loop](../phases/14-agent-engineering/01-the-agent-loop/)
- **관련 용어:** Agent Harness, Agent State, Tool Contract, Human-in-the-Loop (HITL)

### Agent Harness - 에이전트 하네스
- **분류:** 에이전트 및 도구
- **실제 의미:** 컨텍스트를 조립하고, 도구를 노출하며, 상태를 관리하고, 제한을 강제하고, 트레이스를 기록하며, 에이전트가 계속 진행, 재시도, 질문, 또는 중단해야 할 시점을 결정하는 모델 주변의 런타임입니다.
- **왜 중요한가:** 동일한 모델을 사용하는 두 시스템은 하네스가 제공하는 컨텍스트, 도구, 피드백, 및 안전 경계가 다르기 때문에 성능이 매우 다를 수 있습니다.
- **실무에서는:** 하네스는 에이전트를 5번의 도구 호출로 제한하고, 각 승인된 패치 후 체크포인트를 저장하며, 완료 전에 통과된 테스트 명령을 요구할 수 있습니다.
- **흔한 혼동:** 하네스는 프롬프트 템플릿보다 범위가 넓지만, 완전한 제품보다는 범위가 좁습니다.
- **배울 곳:** [Minimal Agent Workbench](../phases/14-agent-engineering/32-minimal-agent-workbench/)
- **관련 용어:** Agent, Tool Contract, Agent State, Verification Gate, Sandbox

### Agent Memory - 에이전트 메모리
- **분류:** Agents & tools
- **실제 의미:** 모델 외부에 저장되며 이후 에이전트 단계에서 사용하도록 선택된 정보로, 이전 결정 사항, 사용자 선호도, 작업 에피소드, 검증된 사실 등이 포함됩니다.
- **왜 중요한가:** 하나의 컨텍스트 윈도우를 넘어서는 연속성을 에이전트에 제공하며, 모든 과거 이벤트를 모든 프롬프트에 강제로 포함시키지 않습니다.
- **실무에서는:** 출처가 포함된 간결한 작업 결과를 저장하고, 관련이 있을 때만 검색하며, 사용자가 영구적인 개인 정보를 검토하거나 수정할 수 있도록 허용합니다.
- **흔한 혼동:** 에이전트 메모리는 에이전트 상태와 동일하지 않습니다. 상태는 현재 실행을 추적하는 반면, 메모리는 가능한 미래 실행을 위해 선택된 정보를 보존합니다.
- **관련 용어:** Agent State, Context Engineering, Checkpoint, Semantic Cache
- **출처:** [Generative Agents](https://arxiv.org/abs/2304.03442)

### Agent State - 에이전트 상태
- **분류:** Agents & tools
- **실제 의미:** 현재 목표, 완료된 작업, 도구 결과, 미해결 질문, 예산, 승인, 아티팩트 참조 등 에이전트가 단계 간에 전달하는 명시적 데이터입니다.
- **왜 중요한가:** 명시적 상태는 긴 작업을 재개 가능하고, 검사 가능하며, 모델이 트랜스크립트로부터 진행 상황을 재구성하는 것에 대한 의존도를 낮추게 합니다.
- **실무에서는:** 선택된 이슈, 변경된 파일, 최신 테스트 결과, 남은 검사를 각 작업 후 업데이트되는 타입 지정된 객체에 저장합니다.
- **흔한 혼동:** 상태는 대화 기록과 동일하지 않습니다. 트랜스크립트는 증거인 반면, 상태는 다음에 일어날 일을 결정하기 위해 사용되는 간결한 운영 기록입니다.
- **배울 곳:** [Repository Memory and State](../phases/14-agent-engineering/34-repo-memory-and-state/)
- **관련 용어:** Checkpoint, Durable Execution, Context Engineering, Handoff

### Agent Skill - 에이전트 스킬
- **분류:** Agents & tools
- **실제 의미:** 진입점이 `SKILL.md`인 절차적 지침의 발견 가능한 디렉토리이며, 호환 런타임이 단계별로 로드할 수 있는 선택적 참조, 스크립트 및 자산이 포함됩니다.
- **왜 중요한가:** 대화와 분리된 재사용 가능한 작업 지식을 패키징하며, 더 깊은 컨텍스트와 결정론적 헬퍼를 필요할 때 사용할 수 있도록 유지합니다.
- **실무에서는:** 간결한 이름과 라우팅 설명을 게시하고, 활성화 후에만 워크플로우를 로드하며, 작업이 해당 지점에 도달했을 때 분기별 참조를 읽습니다.
- **흔한 혼동:** 스킬을 활성화하면 컨텍스트가 제공됩니다. 이는 도구를 노출하거나, 권한을 부여하거나, 샌드박스를 생성하거나, 결과 작업이 정확하다는 것을 증명하는 것이 아닙니다.
- **배울 곳:** [Agent Skills: Portable Contract and Runtime Boundary](../phases/13-tools-and-protocols/22-skills-and-agent-sdks/)
- **관련 용어:** Skill Bundle, Skill Catalog, Skill Invocation, Progressive Disclosure, MCP (Model Context Protocol)
- **출처:** [Agent Skills specification](https://agentskills.io/specification)

### AI Risk Assessment - AI 위험 평가
- **분류:** 보안 및 거버넌스
- **실제 의미:** 컨텍스트, 위험 요소, 발생 가능성, 영향, 통제, 잔존 위험 및 모니터링 책임을 포함하여 AI 시스템이 사람, 조직 및 환경에 어떻게 영향을 미칠 수 있는지에 대한 문서화된 분석입니다.
- **왜 중요한가:** 모델의 능력만으로는 위험이 결정되지 않습니다. 배포 컨텍스트, 영향을 받는 그룹, 인간의 권한, 데이터 및 시스템 통합은 해악과 필요한 통제 모두를 변화시킵니다.
- **실무에서는:** 의도된 사용과 영향을 받는 당사자를 정의하고, 신뢰할 수 있는 실패 및 오용 시나리오를 식별하며, 통제에 소유자를 지정하고, 잔존 위험을 기록하며, 중대한 변경에 대한 검토 트리거를 설정합니다.
- **흔한 혼동:** 위험 평가는 명시된 가정 하에 결정을 지원합니다. 이는 일회성 안전 인증서나 모든 위험 요소가 발견되었음을 증명하는 것이 아닙니다.
- **관련 용어:** Threat Model, Guardrails, Human-in-the-Loop (HITL), Data Classification
- **출처:** [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

### Alignment - 정렬
- **분류:** 평가 및 안전
- **흔히 하는 말:** AI를 안전하게 만드는 것.
- **실제 의미:** 모델이나 AI 시스템이 예상되는 상황과 적대적인 상황 모두에서 의도된 목표, 제약 조건, 인간의 선호에 부합하는 방식으로 작동하도록 하는 노력입니다.
- **왜 중요한가:** 시스템은 명시된 지표는 최적화하면서 사용자의 실제 의도를 위반할 수 있으므로, 정렬은 모델 훈련뿐만 아니라 평가, 감독, 시스템 제어도 필요합니다.
- **관련 용어:** Guardrails, Evaluation (Eval), Human-in-the-Loop (HITL)

### Approval Gate - 승인 게이트
- **분류:** 에이전트 및 도구
- **실제 의미:** 권한을 가진 사람이나 정책이 허가를 부여할 때까지 중대한 조치를 차단하는 통제 지점입니다.
- **왜 중요한가:** 가역적인 작업에 대한 자동화를 유지하면서 불확실한 모델 결정의 영향 범위(폭발 반경)를 제한합니다.
- **실무에서는:** 에이전트가 데이터베이스 마이그레이션을 작성하고 임시 데이터베이스에서 실행하도록 허용하되, 프로덕션 환경에서의 실행은 소유자의 승인을 요구합니다.
- **흔한 혼동:** 승인 게이트는 조치가 권한을 가졌는지 묻습니다. 검증 게이트는 증거가 조치가 정확함을 보여주는지 묻습니다.
- **배울 곳:** [Verification Gates](../phases/14-agent-engineering/38-verification-gates/)
- **관련 용어:** Human-in-the-Loop (HITL), Verification Gate, Least Privilege

### Approximate Nearest Neighbor (ANN) - 근사 최근접 이웃 (ANN)
- **분류:** 검색 및 생성
- **실제 의미:** 쿼리를 저장된 모든 벡터와 비교하지 않고도 쿼리에 가장 가까운 벡터 중 하나일 가능성이 높은 벡터를 반환하는 검색 방법입니다.
- **왜 중요한가:** 근사화는 대규모 벡터 인덱스를 실용적으로 만들지만, 검색 속도, 메모리, 검색 재현율(recall) 사이에 측정 가능한 트레이드오프를 도입합니다.
- **실무에서는:** 홀드아웃(hold-out) 쿼리 세트에 대해 인덱스 및 검색 파라미터를 튜닝한 후, 모든 진짜 이웃이 발견된다고 가정하지 말고 지연 시간(latency)을 Recall@K와 함께 보고합니다.
- **흔한 혼동:** ANN은 검색 목표와 트레이드오프를 설명하는 반면, HNSW는 이를 구현할 수 있는 특정 인덱스 알고리즘입니다.
- **관련 용어:** Vector Database, HNSW, Cosine Similarity, Recall@K
- **출처:** [Efficient and Robust Approximate Nearest Neighbor Search Using HNSW](https://dl.acm.org/doi/10.1109/TPAMI.2018.2889473)

### Attention - 어텐션
- **분류:** 모델 및 추론
- **흔히 하는 말:** 모델이 중요한 토큰에 집중하는 방식입니다.
- **실제 의미:** 쿼리 벡터와 키 벡터를 비교하여 점수를 정규화하고, 이를 사용하여 값 벡터를 결합함으로써 컨텍스트 표현을 형성하는 메커니즘입니다. 마스크, 위치 규칙, 희소 패턴은 참여하는 위치를 제한할 수 있습니다.
- **왜 중요한가:** 어텐션은 모델이 시퀀스 위치 간에 정보를 라우팅하도록 허용하지만, 모델이 무엇을 이해했는지를 설명하거나 증명하지는 못합니다.
- **흔한 혼동:** 어텐션 가중치는 계산 계수이며, 모델 추론에 대한 충실한 설명이 아닙니다.
- **배울 곳:** [Self-Attention from Scratch](../phases/07-transformers-deep-dive/02-self-attention-from-scratch/)
- **출처:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **관련 용어:** Self-Attention, Transformer, KV Cache

### Audio Token - 오디오 토큰
- **분류:** 멀티모달 시스템
- **실제 의미:** 오디오 신호의 짧은 세그먼트나 특징에 대해 오디오 코덱이나 토크나이저가 생성하는 이산 식별자이며, 여러 코드북에 걸쳐 생성되기도 합니다.
- **왜 중요한가:** 이산 오디오 표현은 시퀀스 모델이 토큰 중심 아키텍처를 사용하여 소리를 처리, 예측, 저장 또는 생성할 수 있게 합니다.
- **실무에서는:** 모델과 함께 코덱을 버전 관리하고, 샘플 레이트 및 코드북 메타데이터를 보존하며, 재구성 품질을 측정하고, 의미론적 오디오 토큰과 파형 압축 토큰을 구분해야 합니다.
- **흔한 혼동:** 오디오 토큰은 고정된 기간, 음소, 단어가 아닙니다. 그 의미와 시간 범위는 토크나이저 및 코드북 설계에 따라 달라집니다.
- **배울 곳:** [Neural Audio Codecs](../phases/06-speech-and-audio/13-neural-audio-codecs/)
- **관련 용어:** Token, Embedding, Automatic Speech Recognition (ASR), Multimodal Model
- **출처:** [SoundStream](https://arxiv.org/abs/2107.03312)

### Audit Log - 감사 로그
- **분류:** 보안 및 거버넌스
- **실제 의미:** 누가 또는 무엇이 행동했는지, 무엇이 변경되었는지, 언제 발생했는지, 그리고 resulting status를 포함하는 보안 또는 책임 관련 이벤트의 지속적이고 접근 제어된 기록입니다.
- **왜 중요한가:** 중대한 에이전트 작업은 성능 디버깅을 넘어 조사, 정책 검토, 책임 소재를 뒷받침하는 증거가 필요합니다.
- **실무에서는:** 도구 권한 부여, 승인 결정, 외부 쓰기, 정책 버전, 아티팩트 식별자를 기록하되, 비밀 정보를 가리고 로그 접근을 제한해야 합니다.
- **흔한 혼동:** 트레이스는 하나의 실행 경로를 진단하는 데 도움을 줍니다. 감사 로그는 실행 간 및 시간에 걸쳐 책임 소재에 필요한 이벤트를 보존합니다.
- **관련 용어:** Trace, Observability, Approval Gate, Provenance Attestation
- **출처:** [NIST SP 800-92](https://csrc.nist.gov/pubs/sp/800/92/final)

### Autograd - 오토그라드
- **분류:** 수학 및 학습
- **흔히 하는 말:** 자동 그레이디언트.
- **실제 의미:** 텐서 연산을 기록하거나 변환하여 미분값을 계산할 수 있는 시스템으로, 보통 역방향 모드 자동 미분(reverse-mode automatic differentiation)을 사용합니다. 순방향 연산을 작성하면 프레임워크가 역전파(backpropagation)에 필요한 그레이디언트를 유도합니다.
- **배울 곳:** [Chain Rule and Automatic Differentiation](../phases/01-math-foundations/05-chain-rule-and-autodiff/)
- **관련 용어:** Backpropagation, Gradient, Tensor

### Automatic Speech Recognition (ASR) - 자동 음성 인식 (ASR)
- **분류:** 멀티모달 시스템
- **실제 의미:** 음성 신호를 전사(transcription)로 매핑하는 작업 및 시스템 파이프라인으로, 선택적으로 토큰이나 세그먼트 타이밍 및 신뢰도 정보를 포함합니다.
- **왜 중요한가:** 음성 인터페이스는 언어 모델링 이상의 요소에 의존합니다. 음향 변형, 세그멘테이션, 디코딩, 어휘, 도메인 조건 등이 최종 전사 결과에 영향을 미칩니다.
- **실무에서는:** 언어, 화자, 잡음, 도메인에 따라 단어 또는 문자 오류를 평가하고, 다운스트림 접지(grounding)가 필요할 경우 타임스탬프를 유지하며, 프로덕션에서 사용된 정확한 오디오 전처리 방식을 테스트합니다.
- **흔한 혼동:** ASR은 발화된 내용을 전사합니다. 누가 발화했는지 결정하려면 화자 분리(diarization)나 화자 인식(speaker recognition)이 필요하며, 번역 및 의도 이해는 별도의 작업입니다.
- **배울 곳:** [Speech Recognition and ASR](../phases/06-speech-and-audio/04-speech-recognition-asr/)
- **관련 용어:** Audio Token, Encoder, Tokenization, Multimodal Model
- **출처:** [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

### Autoregressive - 자기회귀
- **분류:** 모델 및 추론
- **흔히 하는 말:** 모델은 한 번에 한 단어를 생성합니다.
- **실제 의미:** 각 출력 토큰이 preceding 토큰들로부터 예측되는 분해 방식입니다. 생성 중 선택된 토큰은 시퀀스에 추가되어 다음 예측의 컨텍스트의 일부가 됩니다.
- **흔한 혼동:** 단위는 토큰이며, 반드시 단어는 아닙니다. 또한 생성은 항상 가장 높은 확률의 토큰을 선택하는 것 외에도 다른 디코딩 방법을 사용할 수 있습니다.
- **관련 용어:** Token, Temperature, KV Cache

### Autoscaling - 자동 확장
- **분류:** 인프라 및 서빙
- **실제 의미:** 관측된 수요, 리소스 사용량, 또는 애플리케이션 메트릭에 기반하여 서빙 워커의 수나 용량을 설정된 범위 내에서 변경하는 제어 루프입니다.
- **왜 중요한가:** AI 워크로드의 변화는 수동 프로비저닝보다 빠를 수 있지만, 스케일링 결정은 모델 로드 시간, 가속기 가용성, 큐잉, 요청 비용을 고려해야 합니다.
- **실무에서는:** 유용한 작업과 연결된 수요 신호에 따라 스케일링하고, 최소 웜(warm) 용량을 설정하며, 스케일다운의 반복적인 변경(churn)을 제한하고, 트래픽을 받기 전에 새 레플리카가 준비성(readiness) 검사를 통과하는지 확인합니다.
- **흔한 혼동:** Autoscaling은 용량을 추가하거나 제거합니다. 과부하된 의존성을 더 빠르게 만들거나, 충분한 하드웨어가 제 시간에 확보될 것을 보장하지는 않습니다.
- **배울 곳:** [GPU Autoscaling on Kubernetes](../phases/17-infrastructure-and-production/03-gpu-autoscaling-kubernetes/)
- **관련 용어:** Model Serving, Saturation, Readiness Probe, Backpressure
- **출처:** [Kubernetes Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

### Availability - 가용성
- **분류:** 신뢰성 및 운영
- **실제 의미:** 정의된 측정 경계 내에서 사용자가 정의된 허용 가능한 서비스를 얻을 수 있는 자격 있는 서비스 상호작용이나 시간 창(window)의 비율입니다.
- **왜 중요한가:** 서비스가 실행 중이라도 사용자가 유용한 요청을 완료하지 못할 수 있으므로, 가용성은 프로세스 가동 시간(uptime)뿐만 아니라 사용자 가시적 성공과 연결되어야 합니다.
- **실무에서는:** 대상 이벤트와 허용 가능한 결과를 정의하고, 문서화된 사례만 제외하며, 고정된 기간 동안 지표를 계산하고, 총 실패와 장기적인 부분적 성능 저하를 모두 조사합니다.
- **흔한 혼동:** 가용성은 하나의 신뢰성 결과입니다. 지연 시간, 정확성, 안전성, 모든 사용자 세그먼트의 경험을 설명하지 않습니다.
- **관련 용어:** Service Level Indicator (SLI), Service Level Objective (SLO), Error Budget, Incident Response
- **출처:** [Google SRE: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)

## B

### Backpressure - 백프레셔
- **분류:** AI-native development
- **실제 의미:** 하위 구성 요소가 현재 속도로 안전하게 처리할 수 없을 때, 상류 작업의 속도를 늦추거나 거부하는 흐름 제어 메커니즘입니다.
- **왜 중요한가:** 백프레셔가 없으면 큐잉된 에이전트 실행, 도구 호출, 스트리밍 이벤트가 메모리를 고갈시키고, 속도 제한을 초과하며, 재시도를 증폭시킬 수 있습니다.
- **실무에서는:** 평가자 큐가 한계에 도달하면, 무제한 작업을 수용하는 대신 새로운 에이전트 작업을 일시 중지하거나 재시도 가능한 응답을 반환합니다.
- **흔한 혼동:** 백프레셔는 실패 전에 용량을 보호합니다. 서킷 브레이커는 의존성이 건강하지 않음을 보여주는 실패가 발생한 후 호출을 중단합니다.
- **관련 용어:** Rate Limit, Retry with Backoff, Circuit Breaker

### Backpropagation - 역전파
- **분류:** Math & training
- **흔히 하는 말:** 신경망이 학습하는 방법입니다.
- **실제 의미:** 스칼라 손실(loss)에서 계산 그래프를 통해 역방향으로 미분값을 전파하는 연쇄 법칙(chain rule)의 효율적인 적용입니다. 이는 그래디언트(gradient)를 계산하며, 옵티마이저(optimizer)는 이 그래디언트를 사용하여 파라미터를 업데이트합니다.
- **흔한 혼동:** 백프로파게이션은 그래디언트를 계산합니다. 업데이트 규칙이나 학습률을 선택하지는 않습니다.
- **이름의 유래:** 미분 정보가 손실(loss)로부터 이전 연산 방향으로 역방향으로 이동하기 때문입니다.
- **배울 곳:** [Backpropagation from Scratch](../phases/03-deep-learning-core/03-backpropagation/)
- **관련 용어:** Autograd, Gradient, Optimizer

### Batch Size - 배치 크기
- **분류:** Math & training
- **흔히 하는 말:** 한 번에 처리되는 예제의 수입니다.
- **실제 의미:** 옵티마이저 업데이트 전에 하나의 그래디언트 추정에 손실(loss)이 기여하는 예제(examples)의 수입니다. 더 큰 배치(batch)는 하드웨어 활용도를 높이고 그래디언트 노이즈를 줄일 수 있지만, 더 많은 메모리를 필요로 하며 다른 학습률이나 스케줄링 선택이 필요할 수 있습니다.
- **흔한 혼동:** 모든 배치 크기 증가가 동일한 학습률 증가를 산출해야 한다고 말하는 보편적인 배치 크기 범위나 규칙은 없습니다.
- **관련 용어:** Learning Rate, Gradient, Optimizer

### Benchmark Contamination - 벤치마크 오염
- **분류:** 평가 및 안전
- **실제 의미:** 평가 예제와 평가 대상 시스템을 사전 훈련, 튜닝, 프롬프트, 선택 또는 기타 방식으로 개선하는 데 사용된 데이터 간의 중첩이나 정보 유출입니다.
- **왜 중요한가:** 오염은 벤치마크 점수가 unseen task에 대한 일반화 능력보다는 사전 노출을 반영하게 만들 수 있습니다.
- **실무에서는:** 데이터셋의 출처(provenance)를 추적하고, 훈련 소스에서 정확하고 유사한 중복을 검색하며, 비공개 테스트 케이스를 보관하고, 새로 작성된 예제로 공개 평가(evals)를 갱신합니다.
- **흔한 혼동:** 오염은 정확한 복사보다 더 광범위합니다. 패러프레이즈(paraphrases), 정답 키, 벤치마크 메타데이터, 반복적인 프롬프트 튜닝도 평가 정보를 유출할 수 있습니다.
- **관련 용어:** Data Leakage, Data Deduplication, Eval Set, Exact Match (EM)
- **출처:** [Investigating Data Contamination in Modern Benchmarks for Large Language Models](https://arxiv.org/abs/2311.09783)

### BM25
- **분류:** 검색 및 생성
- **실제 의미:** 용어의 희소성, 반복 출현, 문서 길이를 고려하면서 쿼리 용어(query-term) 매칭으로 문서를 점수화하는 어휘 기반 순위 함수입니다.
- **왜 중요한가:** 강력한 정확한 용어(exact-term) 검색 기준선(baseline)이며, 식별자, 희귀 단어, 도메인 특화 구문에서 dense retrieval을 보완합니다.
- **실무에서는:** BM25와 dense search로 후보를 검색하고, 그들의 순위를 결합한 후, 더 비싼 reranker를 추가하기 전에 병합된 결과를 평가합니다.
- **흔한 혼동:** BM25는 semantic similarity를 직접 이해하지 못하며, 그 점수는 다른 쿼리나 인덱스 구성에 걸쳐 보편적인 의미를 가지지 않습니다.
- **관련 용어:** Hybrid Retrieval, Dense Retrieval, Reranker, RAG (Retrieval-Augmented Generation)
- **출처:** [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019)

### Byte Pair Encoding (BPE) - 바이트 쌍 인코딩 (BPE)
- **분류:** 데이터 및 표현
- **실제 의미:** 학습 텍스트에서 자주 나타나는 인접 단위들을 반복적으로 병합하여 고정된 어휘를 구성하는 하위 단어(subword) 토큰화 방법입니다.
- **왜 중요한가:** 어휘 크기와 희소하거나 처음 보는 단어를 더 작은 단위로 표현하는 능력 사이의 균형을 유지합니다.
- **실무에서는:** 승인된 코퍼스 분할(corpus splits)에서만 토크나이저를 학습하고, 병합 규칙을 모델과 함께 버전 관리하며, 코드, 다국어 텍스트 및 공백을 어떻게 분할하는지 점검합니다.
- **흔한 혼동:** BPE는 토크나이저 계열 중 하나일 뿐, 모든 모델이 토큰을 생성하는 방식을 설명하는 보편적인 개념이 아닙니다.
- **관련 용어:** Tokenization, Vocabulary, Token, Embedding
- **출처:** [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

## C

### Calibration - 보정
- **분류:** 평가 및 안전
- **실제 의미:** 시스템이 표명한 신뢰도와, 해당 신뢰도에서의 예측이 실제로 맞은 빈도 간의 일치 정도입니다.
- **왜 중요한가:** 시스템이 평균적으로 정확하더라도, 사람들이 점수를 의존하는 사례에서 위험할 정도로 과신할 수 있습니다.
- **실무에서는:** 신뢰도에 따라 예측을 버킷화하고, 신뢰도와 경험적 정확도를 비교하며, 격차가 허용 불가능할 경우 재보정(recalibrate)하거나 예측을 보류(abstain)합니다.
- **흔한 혼동:** Calibration은 신뢰도의 신뢰성을 측정하는 것이지, 전체 정확도, 사실성(factuality) 또는 추론 품질을 측정하는 것이 아닙니다.
- **관련 용어:** Softmax, Evaluation (Eval), Precision & Recall, Logits
- **출처:** [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)

### Canary Release - 카나리 릴리스
- **분류:** 신뢰성 및 운영
- **실제 의미:** 롤아웃(rollout)을 확장하기 전에 트래픽이나 인프라의 제한된 일부에 새로운 버전을 노출하는 배포 전략입니다.
- **왜 중요한가:** 결함의 영향을 제한하며, 새로운 모델, 프롬프트, 에이전트 또는 서비스가 모든 사용자에게 도달하기 전에 프로덕션 환경에서의 증거를 확보할 수 있게 해줍니다.
- **실무에서는:** 소규모의 대상 코호트를 릴리스에 라우팅하고, 품질 및 운영 지표는 통제 그룹과 비교하며, 사전에 정의된 실패 발생 시 중단하거나 롤백합니다.
- **흔한 혼동:** 카나리 릴리스는 노출 범위를 제한할 뿐이며, 배포 전 테스트, 승인, 롤백 준비를 대체하지 않습니다.
- **관련 용어:** Evaluation (Eval), Observability, Rollback, Verification Gate
- **출처:** [Kubernetes Deployments: Canary Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#canary-deployment)

### Chain of Thought (CoT) - 사고의 연쇄 (CoT)
- **분류:** 프롬팅 및 컨텍스트
- **흔히 하는 말:** 모델이 사고의 모든 단계를 보여주도록 요청하는 것.
- **실제 의미:** 답변을 생성하기 전에 작업을 분해하기 위해 사용되는 중간 추론입니다. 프롬프트는 가시적인 근거를 요청할 수 있으며, 일부 시스템은 사용자에게 반환되지 않는 내부 추론을 사용합니다.
- **왜 중요한가:** 분해는 다단계 작업에 도움이 될 수 있지만, 유창한 근거는 답변이 정확하다는 증거가 아니며, 텍스트가 모델의 내부 연산을 충실히 반영한다는 증거도 아닙니다.
- **실무에서는:** 간결한 계획을 요청하고, 결과를 독립적으로 검증하며, 긴 추론 기록에 의존하는 대신 검증 가능한 계산이나 인용을 요청합니다.
- **흔한 혼동:** Chain of Thought는 도구, 테스트, 외부 검증을 대체하지 않습니다.
- **배울 곳:** [Few-Shot and Chain of Thought](../phases/11-llm-engineering/02-few-shot-cot/)
- **관련 용어:** Prompt Engineering, Verification Gate, Evaluation (Eval)

### Checkpoint - 체크포인트
- **분류:** 에이전트 및 도구
- **실제 의미:** 알려진 경계 지점에서 재개하기 위해 사용되는 영구 스냅샷입니다. 워크플로우에서는 운영 상태와 아티팩트 참조를 저장합니다. 모델 훈련에서는 파라미터, 옵티마이저 상태, 스케줄러 상태, 훈련 위치를 저장할 수 있습니다.
- **왜 중요한가:** 장시간 실행되는 워크플로우와 훈련 실행은 완료된 작업을 다시 실행하거나 비싼 진행 상황을 잃지 않고 중단에서 복구할 수 있습니다.
- **실무에서는:** 검증된 단계 후 에이전트의 승인된 패치와 테스트 증거를 저장하거나, 종료 전에 훈련 실행의 가중치, 옵티마이저 상태, 랜덤 상태, 데이터 위치를 저장합니다.
- **흔한 혼동:** 워크플로 체크포인트와 모델 학습 체크포인트는 동일한 복구 목적을 가지지만, 서로 다른 상태를 보존합니다. 둘 다 단순한 트랜스크립트나 재개 메타데이터가 없는 가중치 파일이 아닙니다.
- **배울 곳:** [Checkpoint Save and Resume](../phases/19-capstone-projects/47-checkpoint-save-resume/); [Repository Memory and State](../phases/14-agent-engineering/34-repo-memory-and-state/)
- **관련 용어:** Agent State, Durable Execution, Parameter, Optimizer

### Chunked Prefill - 청크 프리필
- **분류:** 인프라 및 서빙
- **실제 의미:** 긴 프롬프트의 프리필 작업을 더 작은 스케줄링 가능한 단위로 나누어, 프롬프트 처리가 다른 요청의 디코딩 작업과 교대로 수행될 수 있도록 하는 서빙 기법입니다.
- **왜 중요한가:** 하나의 긴 프롬프트가 가속기를 독점하여 활성 생성을 지연시킬 수 있으며, 총 처리량이 건강해 보일지라도 꼬리 지연(tail latency)이 나빠질 수 있습니다.
- **실무에서는:** 측정된 워크로드에 기반하여 청크 정책을 선택하고, 스케줄링 오버헤드를 고려하며, 혼합된 프롬프트 길이에서 프리필 완료, 디코딩 지연 및 goodput을 비교하십시오.
- **흔한 혼동:** 청크드 프리필은 프롬프트 연산의 스케줄링 방식을 변경합니다. 사용자의 컨텍스트를 독립적인 의미 단위 청크로 나누거나 모델의 컨텍스트 윈도우를 변경하지는 않습니다.
- **배울 곳:** [vLLM Serving Internals](../phases/17-infrastructure-and-production/04-vllm-serving-internals/)
- **관련 용어:** Prefill, Decode Phase, Dynamic Batching, Tail Latency
- **출처:** [Sarathi-Serve](https://arxiv.org/abs/2403.02310)

### Chunking - 청킹
- **분류:** 검색 및 생성
- **흔히 하는 말:** 문서를 조각으로 나누는 것.
- **실제 의미:** 인덱싱 전에 소스 자료를 검색 가능한 단위로 나누는 것. 청크 경계, 겹침(overlap), 메타데이터 및 문서 구조는 프롬프트를 과도하게 채우지 않으면서 충분한 컨텍스트를 검색 결과가 반환하는지 결정합니다.
- **왜 중요한가:** 올바른 청킹 전략은 문서 형태, 쿼리 유형, 임베딩 모델 및 평가 결과에 따라 달라집니다. 보편적인 토큰 크기나 겹침 비율은 존재하지 않습니다.
- **실무에서는:** 헤딩과 코드 블록을 온전히 유지하고, 소스 메타데이터를 첨부한 후, 크기를 조정하기 전에 실제 질문으로 검색 품질을 측정하십시오.
- **관련 용어:** RAG (Retrieval-Augmented Generation), Reranker, Grounding

### Circuit Breaker - 서킷 브레이커
- **분류:** AI 네이티브 개발
- **실제 의미:** 의존성 장애가 임계값을 넘으면 의존성 호출을 일시적으로 중단하고, 이후 의존성이 복구되었는지 탐지하는 신뢰성 제어입니다.
- **왜 중요한가:** 반복적인 모델 또는 도구 장애가 시스템 전체의 지연 시간, 예산, 용량을 소모하는 것을 방지합니다.
- **실무에서는:** 반복적인 제공자 시간 초과 후 브레이커를 열고, 페일오버를 수행하거나 제어된 응답을 반환한 후, 쿨다운 기간이 지나면 제한된 헬스 체크를 허용합니다.
- **흔한 혼동:** 서킷 브레이커는 의존성 상태에 반응합니다. 속도 제한은 허용된 요청 볼륨을 제어합니다.
- **관련 용어:** Retry with Backoff, Rate Limit, Model Router, Backpressure

### CNN (Convolutional Neural Network) - CNN (합성곱 신경망)
- **분류:** 모델 및 추론
- **흔히 하는 말:** 이미지를 위한 신경망입니다.
- **실제 의미:** 입력 위에 슬라이딩 필터를 적용하는 컨볼루션 연산을 사용하여 지역 패턴을 감지하는 신경망입니다. 컨볼루션을 쌓아 올리면 가장자리, 텍스처, 객체 등 점점 더 복잡한 특징을 감지합니다.
- **흔한 혼동:** 컨볼루션은 오디오, 시계열 및 기타 그리드형 데이터에도 적용됩니다.
- **관련 용어:** Feature, Inductive Bias, Activation Function

### Coding Agent - 코딩 에이전트
- **분류:** AI 네이티브 개발
- **실제 의미:** 저장소를 검사하고, 파일을 편집하며, 개발 도구를 실행하고, 그 출력 결과를 활용하여 범위가 정해진 엔지니어링 작업을 진행할 수 있는 소프트웨어 작업 특화 에이전트입니다.
- **왜 중요한가:** 그 가치는 코드 생성 품질뿐만 아니라 저장소 컨텍스트, 도구 권한, 리뷰 경계 및 검증에 의존합니다.
- **실무에서는:** 에이전트에게 이슈, 범위 계약, 저장소 지침 및 테스트 명령을 제공하십시오. 승인하기 전에 결과 패치와 증거를 검토하십시오.
- **흔한 혼동:** 텍스트만 제안하는 코딩 어시스턴트는 반드시 에이전트라고 할 수 없습니다. 에이전트는 도구를 통해 행동하고 결과를 관찰합니다.
- **배울 곳:** [Skill Discovery and Progressive Disclosure](../phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/)
- **관련 용어:** Agent Harness, Repository Map, Patch, Scope Contract, Reviewer Agent

### Compensating Action - 보상 조치
- **분류:** 에이전트 및 도구
- **실제 의미:** 원본 작업을 원자적으로 롤백할 수 없을 때, 완료된 사이드 효과를 의미적으로 상쇄하기 위해 의도적으로 수행하는 작업입니다.
- **왜 중요한가:** 다단계 에이전트 워크플로우는 데이터베이스와 외부 서비스를 넘나들며, 이후의 실패가 단일 트랜잭션으로 이전의 쓰기 작업을 되돌릴 수 없는 경우가 많습니다.
- **실무에서는:** 예약 워크플로우가 카드 결제를 완료했으나 예약이 실패한 경우, 이력을 삭제하지 말고 추적 가능한 환불을 발행하고 두 이벤트를 모두 보존해야 합니다.
- **흔한 혼동:** 보상은 새로운 비즈니스 작업이며, 시간 여행이 아닙니다. 보상은 실패할 수 있으므로 멱등성, 모니터링 및 에스컬레이션이 필요합니다.
- **관련 용어:** Durable Execution, Idempotency, Checkpoint, Approval Gate
- **출처:** [Sagas](https://dl.acm.org/doi/10.1145/38713.38742)

### Content Provenance - 콘텐츠 출처
- **분류:** 보안 및 거버넌스
- **실제 의미:** 미디어나 기타 디지털 콘텐츠의 출처 및 편집 이력에 대한 검증 가능한 정보로, 관련 행위자, 도구, 변환 및 첨부된 주장이 포함됩니다.
- **왜 중요한가:** 생성형 시스템은 외관만으로는 출처 주장을 추론하기 어렵기 때문에, 소비자와 플랫폼은 콘텐츠가 어떻게 제작되었는지에 대한 검증 가능한 증거가 필요합니다.
- **실무에서는:** 출처 주장을 콘텐츠에 바인딩하고, 통제된 신원으로 서명하며, 변환 이력을 보존하고, 증거가 없거나 검증할 수 없는 경우 이를 명확히 표시해야 합니다.
- **흔한 혼동:** 출처는 누가 이력을 주장했는지와 기록이 변경되었는지를 확립할 수 있습니다. 그러나 묘사된 사건이 참이라는 사실이나 콘텐츠가 무해하다는 사실을 증명하지는 못합니다.
- **배울 곳:** [Watermarking, SynthID, Stable Signature, and C2PA](../phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa/)
- **관련 용어:** Data Provenance, Provenance Attestation, Audit Log, Grounding
- **출처:** [C2PA Technical Specification](https://c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html)

### Context Compression - 컨텍스트 압축
- **분류:** 프롬팅 및 컨텍스트
- **실제 의미:** 이후 모델의 결정에 필요한 정보를 보존하려고 시도하면서, 원본 자료의 토큰 사용량을 줄이는 것입니다.
- **왜 중요한가:** 압축은 긴 작업을 예산 내로 맞추는 데 도움이 될 수 있지만, 생략된 모든 세부 사항은 모델이 증거, 제약 조건, 또는 미해결 상태를 잃어버릴 위험을 만듭니다.
- **실무에서는:** 권위 있는 사실과 식별자는 원문 그대로 보존하고, 중복된 이력은 요약하며, 출처 포인터를 첨부하고, 대표 작업에서 압축된 컨텍스트를 테스트하십시오.
- **흔한 혼동:** 원본 전체를 보존하지 않는 한 압축은 손실적입니다. 더 짧은 요약이 자동으로 동등한 컨텍스트가 되는 것은 아닙니다.
- **관련 용어:** Token Budget, Context Engineering, Progressive Disclosure, Handoff
- **출처:** [LLMLingua](https://arxiv.org/abs/2310.05736)

### Context Engineering - 컨텍스트 엔지니어링
- **분류:** 프롬팅 및 컨텍스트
- **실제 의미:** 각 단계에서 모델에 제공되는 전체 정보 환경을 설계하는 것. 지시문, 선택된 파일, 검색된 증거, 도구 결과, 예시, 상태, 출력 제약 조건 등이 포함됩니다.
- **왜 중요한가:** 모델 성능은 관련 증거가 누락되었거나, 오래되었거나, 순서가 잘못되었거나, 잡음에 압도당했을 때 종종 실패합니다.
- **실무에서는:** 목표, 저장소 규칙, 관련 인터페이스, 최근 도구 출력, 미해결 결정을 포함하는 간결한 작업 패킷을 구축하고, 상태가 변경됨에 따라 이를 업데이트하십시오.
- **흔한 혼동:** 프롬프트 엔지니어링은 지시문 표현에 집중합니다. 컨텍스트 엔지니어링은 모델의 작업 컨텍스트에 어떤 증거와 상태가 들어갈지도 결정합니다.
- **배울 곳:** [Context Engineering](../phases/11-llm-engineering/05-context-engineering/)
- **관련 용어:** Context Window, Progressive Disclosure, Agent State, Repository Map

### Context Window - 컨텍스트 윈도우
- **분류:** 프롬팅 및 컨텍스트
- **흔히 하는 말:** 모델이 얼마나 기억하는가.
- **실제 의미:** 특정 모델 및 API 계약 하에서 하나의 모델 추론에 사용 가능한 최대 토큰 용량입니다. 이 용량에는 시스템 지시문, 메시지, 검색된 콘텐츠, 도구 교환, 생성된 출력이 포함될 수 있으며, 제공자별 회계 및 출력 제한이 적용됩니다.
- **왜 중요한가:** 대화 이력은 애플리케이션이 이를 전송하거나 재구성할 때만 이용 가능합니다. 큰 윈도우는 포함된 모든 세부 사항이 신뢰할 수 있게 사용된다는 보장을 제공하지 않습니다.
- **흔한 혼동:** 컨텍스트는 추론에 대한 임시 입력입니다. 영구적인 메모리는 모델 외부에 저장되며, 이후 컨텍스트에 다시 선택되어 포함됩니다.
- **배울 곳:** [Context Engineering](../phases/11-llm-engineering/05-context-engineering/)
- **관련 용어:** Token Budget, Context Engineering, Prompt Cache, Agent State

### Continuous Batching - 연속 배치
- **분류:** 인프라 및 서빙
- **실제 의미:** 고정된 배치 내의 모든 요청이 완료될 때까지 기다리지 않고, 반복 경계(iteration boundaries)에서 생성 요청을 추가하고 제거하는 서빙 스케줄러입니다.
- **왜 중요한가:** 자기회귀(Autoregressive) 요청은 출력 길이가 서로 다르므로, continuous batching은 짧은 요청이 가장 긴 요청을 기다리도록 강제하지 않으면서 가속기를 계속 활용 상태를 유지할 수 있습니다.
- **실무에서는:** 용량이 확보되면 새로운 요청을 수용하고, 요청별 지연 시간을 추적하며, 활성 배치나 KV-cache 예산이 가득 찼을 때 백압(backpressure)을 적용합니다.
- **흔한 혼동:** Continuous batching은 추론 스케줄링 정책이며, 그래디언트 누적이나 훈련 배치 크기 기법이 아닙니다.
- **관련 용어:** Dynamic Batching, Decode Phase, Backpressure, Rate Limit
- **출처:** [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)

### Contrastive Learning - 대조 학습
- **분류:** 수학 및 훈련
- **흔히 하는 말:** 비교를 통한 학습.
- **실제 의미:** 임베딩 공간에서 유사한 쌍은 가깝게 당기고, 비유사한 쌍은 멀리 밀어내는 방식으로 훈련하는 것입니다. CLIP은 이를 사용합니다: 이미지-텍스트 쌍을 매칭하는 것과 매칭되지 않는 것을 비교합니다.
- **관련 용어:** Embedding, Cosine Similarity, Loss Function

### Cosine Similarity - 코사인 유사도
- **분류:** 데이터 및 표현
- **흔히 하는 말:** 두 벡터가 얼마나 유사한지.
- **실제 의미:** 두 벡터의 정규화된 내적입니다. 크기가 아닌 방향을 비교하며, 실수 값 벡터의 경우 -1에서 1까지의 범위를 가집니다.
- **흔한 혼동:** 높은 코사인 유사성은 임베딩 모델과 데이터 분포에 상대적인 의미만 가집니다. 사실적이나 의미적 동등성을 증명하지는 않습니다.
- **관련 용어:** Embedding, Semantic Search, Reranker

### Cost per Successful Task - 성공 작업당 비용
- **분류:** AI 네이티브 개발
- **실제 의미:** 정의된 성공 기준을 충족하는 작업 수로 총 시스템 비용을 나눈 값입니다. 여기에는 재시도, 실패한 실행, 도구 사용 및 평가 오버헤드가 포함됩니다.
- **왜 중요한가:** 값싼 모델 호출이 자주 실패하거나 반복적인 인간 수정이 필요할 경우, 비싼 워크플로우를 생성할 수 있습니다.
- **실무에서는:** 100개 저장소 작업에 걸쳐 제공자 요금과 인프라 비용을 측정한 후, 테스트 및 리뷰를 통과한 패치를 가진 작업 수로 나눕니다.
- **흔한 혼동:** 토큰당 비용은 사용량을 측정합니다. 성공한 작업당 비용은 유용한 결과를 측정합니다.
- **관련 용어:** Evaluation (Eval), Retry with Backoff, Model Router, Verification Gate

### Cross-Attention - 교차 어텐션
- **분류:** 멀티모달 시스템
- **실제 의미:** 쿼리 표현은 하나의 시퀀스나 표현에서 나오지만, 키와 값은 다른 시퀀스나 표현에서 나오는 어텐션입니다.
- **왜 중요한가:** 언어 토큰이 시각적 특징에 어텐션하는 것과 같이, 하나의 스트림이 다른 스트림에서 정보를 검색하는 학습 가능한 방법을 제공합니다.
- **실무에서는:** 쿼리, 키, 값을 제공하는 스트림을 명시하고, 누락되거나 유효하지 않은 위치에 대해 마스크를 적용하며, 한 모달리티가 제거(ablated)되었을 때 모델이 여전히 성능을 발휘하는지 확인합니다.
- **흔한 혼동:** 교차 어텐션은 본질적으로 멀티모달이 아닙니다. 두 텍스트 시퀀스나 다른 표현을 연결할 수 있으며, 자기 어텐션(self-attention)은 쿼리, 키, 값을 동일한 시퀀스 표현에서 파생합니다.
- **관련 용어:** Attention, Self-Attention, Vision-Language Model (VLM), Multimodal Fusion
- **출처:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Cross-Entropy - 교차 엔트로피
- **분류:** 수학 및 학습
- **흔히 하는 말:** 분류 손실.
- **실제 의미:** 대상 결과에 할당된 음의 로그 확률에 기반한 손실입니다. 다음 토큰 학습에서는 관측된 다음 토큰에 낮은 확률을 할당할 때 모델에 페널티를 부과합니다.
- **흔한 혼동:** 퍼플렉시티(perplexity)는 평균화와 로그 밑(base)이 일관되게 정의될 때만 교차 엔트로피의 지수화된 평균입니다.
- **관련 용어:** Loss Function, Softmax, Perplexity

### CUDA
- **분류:** 모델 및 추론
- **흔히 하는 말:** GPU 프로그래밍.
- **실제 의미:** 호환되는 GPU에서 범용 연산을 수행하기 위한 NVIDIA의 플랫폼 및 프로그래밍 모델입니다. 딥러닝 프레임워크는 CUDA 라이브러리와 커널을 사용하여 많은 텐서 연산을 병렬로 실행합니다.
- **흔한 혼동:** GPU 가속은 CUDA와 동의어가 아닙니다. 다른 하드웨어 및 소프트웨어 스택도 존재합니다.
- **관련 용어:** Tensor, Mixed Precision, JAX

## D

### Data Augmentation - 데이터 증강
- **분류:** 수학 및 학습
- **흔히 하는 말:** 더 많은 학습 데이터를 만드는 것.
- **실제 의미:** 변환된 이미지, 변형된 오디오, 패러프레이즈된 텍스트 등 수정된 예제를 만들어, 완전히 새로운 원본 데이터를 수집하지 않고도 학습 다양성을 높이는 것입니다. 변환이 작업 신호를 보존할 경우 과적합을 줄일 수 있습니다.
- **흔한 혼동:** 증강은 모델이 학습해야 하는 목표 레이블이나 동작을 보존해야 합니다.
- **관련 용어:** Overfitting, Epoch, Eval Set

### Data Classification - 데이터 분류
- **분류:** 보안 및 거버넌스
- **실제 의미:** 데이터를 문서화된 민감도 또는 영향 클래스에 할당하여, 공개 또는 손실의 결과에 따라 처리, 접근, 보존, 공유 및 사고 규칙이 따르도록 하는 것입니다.
- **왜 중요한가:** AI 파이프라인은 원본 문서, 프롬프트, 추적 및 생성된 아티팩트가 동일한 민감도로 취급될 경우, 비례적인 통제를 적용할 수 없습니다.
- **실무에서는:** 데이터 수집 시 분류를 수행하고, 파생된 아티팩트에도 레이블을 유지하며, 클래스에 따라 도구와 목적지를 제한하고, 변환 또는 집계 후 레이블이 어떻게 변경되는지 정의합니다.
- **흔한 혼동:** 데이터 분류는 보호 요구 사항을 설명합니다. 기계 학습 분류 작업이나 데이터가 정확하다는 주장과는 동일하지 않습니다.
- **관련 용어:** Data Minimization, Trust Boundary, Least Privilege, Audit Log
- **출처:** [NIST SP 1800-39 Initial Public Draft: Data Classification Practices](https://www.nccoe.nist.gov/sites/default/files/2026-02/nist-sp-1800-39-ipd.pdf); [NIST FIPS 199: Federal Information and Information System Categorization](https://csrc.nist.gov/pubs/fips/199/final)

### Data Deduplication - 데이터 중복 제거
- **분류:** 데이터 및 표현
- **실제 의미:** 데이터셋 내부 또는 데이터셋 간에 존재하는 완전 중복 및 유사 중복 예제를 탐지하고 제거하는 것입니다.
- **왜 중요한가:** 반복은 학습 분포를 왜곡하고, 암기(memorization)를 증가시키며, 테스트 데이터를 유출할 수 있고, 평가 결과가 실제보다 더 강해 보이게 만들 수 있습니다.
- **실무에서는:** 콘텐츠를 정규화하고, 정확한 해시 및 유사도 방법을 사용하며, 경계 클러스터를 검토하고, 각 예제를 제거한 버전과 규칙을 기록합니다.
- **흔한 혼동:** 중복 제거는 일반적인 데이터 클리닝이 아닙니다. 두 개의 서로 다른 레코드가 텍스트를 공유하는 것은 합법적일 수 있으며, 두 개의 패러프레이즈(paraphrase)가 동일한 유출 정보를 담고 있을 수도 있습니다.
- **관련 용어:** Data Provenance, Benchmark Contamination, Dataset Split, Overfitting
- **출처:** [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)

### Data Exfiltration - 데이터 유출
- **분류:** 보안 및 거버넌스
- **실제 의미:** 보호된 데이터를 시스템 또는 신뢰 구역(trust zone)에서 이를 수신할 권한이 없는 사람, 도구, 서비스, 저장 위치로 무단 전송하는 것입니다.
- **왜 중요한가:** 에이전트는 원본 데이터 저장소가 온전하게 유지되는 경우에도 생성된 텍스트, 도구 인자, URL, 로그, 또는 사이드 이펙트를 통해 비밀을 노출할 수 있습니다.
- **실무에서는:** 가독성 있는 데이터를 최소화하고, 목적지를 허용 목록(allowlist)으로 지정하며, 아웃바운드(outbound) 도구 호출을 검사하고, 민감한 필드를 가리고(redact), 신뢰 경계(trust boundary)를 넘는 비정상적인 전송에 대해 알림을 설정합니다.
- **흔한 혼동:** Exfiltration은 무단 이동이나 공개에 관한 것입니다. 권한이 있는 구성 요소가 데이터를 평범하게 검색하는 것은 exfiltration이 아니지만, 이후의 사용은 exfiltration이 될 수 있습니다.
- **배울 곳:** [EchoLeak and CVEs for AI](../phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai/)
- **관련 용어:** Trust Boundary, Least Privilege, Indirect Prompt Injection, Audit Log
- **출처:** [NIST SP 800-53 Rev. 5: AC-4 Information Flow Enforcement](https://csrc.nist.gov/files/pubs/sp/800/53/r5/upd1/final/docs/sp800-53r5-controls.xlsx)

### Data Leakage - 데이터 누수
- **분류:** 데이터 및 표현
- **실제 의미:** 학습이나 피처 구축 과정에서, 실제 예측 시점에는 이용 불가능하거나 홀드아웃(hold-out) 평가 경계에 속하는 정보를 의도치 않게 사용하는 것입니다.
- **왜 중요한가:** 누수는 시스템이 실제로 본 적 없는 입력을 만나면 붕괴하는 낙관적인 지표를 생성합니다.
- **실무에서는:** 전처리기를 적용하기 전에 데이터를 분할하고, 미래 정보를 역사적 특징에 포함하지 않으며, 테스트 레이블과 벤치마크 답변을 프롬프트 및 튜닝 루프에서 격리합니다.
- **흔한 혼동:** 누수는 중복된 행에 국한되지 않습니다. 전역 정규화 통계, 타임스탬프, 타겟 유래 특징, 반복적인 테스트 기반 프롬프트 편집은 모두 정보를 누수시킬 수 있습니다.
- **관련 용어:** Dataset Split, Benchmark Contamination, Eval Set, Data Provenance
- **출처:** [scikit-learn: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

### Data Lineage - 데이터 계보
- **분류:** 보안 및 거버넌스
- **실제 의미:** 소스, 변환, 조인, 필터, 버전 및 다운스트림 사용을 통해 데이터 아티팩트가 파생된 방식에 대한 기록입니다.
- **왜 중요한가:** 소스가 수정, 취소되거나 안전하지 않은 것으로 발견될 때, 계보(lineage)는 영향을 받을 수 있는 데이터셋, 임베딩, 평가 및 모델 아티팩트를 식별합니다.
- **실무에서는:** 입력과 출력에 안정적인 식별자를 부여하고, 각 변환 및 버전을 기록하며, 부모-자식 관계를 보존하고, 영향을 받은 소스가 모든 파생물까지 추적될 수 있는지 테스트합니다.
- **흔한 혼동:** Data Provenance는 출처와 보관을 광범위하게 설명합니다. Lineage는 데이터 아티팩트 간의 변환 경로와 의존성을 강조합니다.
- **관련 용어:** Data Provenance, Datasheet for Datasets, Audit Log, Content Provenance
- **출처:** [W3C PROV-O](https://www.w3.org/TR/prov-o/)

### Data Minimization - 데이터 최소화
- **분류:** 보안 및 거버넌스
- **실제 의미:** 개인 데이터의 경우, 지정된 목적에 필요한 범위 내에서 수집, 처리, 노출 및 보관을 제한하는 것입니다. 팀은 엔지니어링 통제로서 민감한 비개인 데이터에도 동일한 원칙을 적용할 수 있습니다.
- **왜 중요한가:** 프롬프트, 트레이스, 캐시 또는 도구 호출에 불필요한 필드를 포함하면 프라이버시 노출이 증가하고 오용 또는 침해의 잠재적 영향이 커집니다.
- **실무에서는:** 수집 전에 필수 필드를 정의하고, 가장 이른 경계에서 삭제 또는 집계하며, 보존 기간을 설정하고, 선택적 컨텍스트가 측정된 작업 결과를 개선하는지 확인한 후 이를 유지합니다.
- **흔한 혼동:** 최소화는 데이터를 전혀 보관하지 않는다는 의미가 아닙니다. 각 데이터 요소, 사용, 수신자 및 보존 기간이 명시된 목적에 대해 정당화할 수 있어야 한다는 의미입니다.
- **관련 용어:** Purpose Limitation, Data Classification, Least Privilege, Context Engineering
- **출처:** [General Data Protection Regulation, Article 5(1)(c)](https://eur-lex.europa.eu/eli/reg/2016/679/oj)

### Data Provenance - 데이터 출처
- **분류:** Data & representations
- **실제 의미:** 데이터가 어디서 originate했는지, 누가 또는 무엇이 이를 변환했는지, 어떤 버전이 사용되었는지, 그리고 파생된 산출물이 원본과 어떻게 관련되는지에 대한 추적 가능한 정보입니다.
- **왜 중요한가:** 결과를 재현하고, 사용 제약 조건을 준수하며, 오염을 조사하고, 소스가 변경될 때 영향을 받은 데이터를 제거하기 위해 출처 정보(provenance)가 필요합니다.
- **실무에서는:** 불변의 데이터셋 버전을 할당하고, 변환 작업 및 소스 식별자를 기록하며, 임베딩, 평가 사례 및 모델 산출물에 계보(lineage) 메타데이터를 포함합니다.
- **흔한 혼동:** 소스 URL은 출처 정보의 일부일 뿐이며, 수집 시간, 라이선스, 필터링, 변환 또는 다운스트림 사용을 설명하지 못합니다.
- **관련 용어:** Dataset Split, Data Deduplication, Provenance Attestation, Grounding
- **출처:** [W3C PROV Overview](https://www.w3.org/TR/prov-overview/)

### Dataset Split - 데이터셋 분할
- **분류:** Data & representations
- **실제 의미:** 피팅(fitting), 개발 결정 및 최종 평가를 위해 예제를 별도의 하위 집합으로 문서화된 분할입니다.
- **왜 중요한가:** 분리를 통해 시스템 선택에 사용된 증거가 선택된 시스템이 일반화됨을 독립적으로 증명하는 데에도 사용되지 않도록 방지합니다.
- **실무에서는:** 사용자, 저장소, 조직 또는 시간과 같은 실제 배포 단위로 분할하며, 상관관계가 있는 행을 무작위로 나누지 않습니다.
- **흔한 혼동:** 무작위 분할은 자동으로 독립적이지 않습니다. 유사한 중복 데이터, 미래 관측치 또는 동일 엔티티의 레코드가 경계를 넘나들 수 있습니다.
- **관련 용어:** Eval Set, Overfitting, Data Leakage, Distribution Shift
- **출처:** [Datasheets for Datasets](https://cacm.acm.org/research/datasheets-for-datasets/)

### Datasheet for Datasets - 데이터셋 명세서
- **분류:** 보안 및 거버넌스
- **실제 의미:** 데이터셋의 동기, 구성, 수집 과정, 전처리, 용도, 분포, 유지보수 및 알려진 한계에 대한 구조화된 문서화입니다.
- **왜 중요한가:** 데이터셋은 단순히 이용 가능하다는 이유만으로 안전하거나 적합하지 않습니다. 후속 개발자는 데이터셋이 어떻게 생성되었는지, 그리고 그 가정이 어디에서 깨지는지에 대한 증거가 필요합니다.
- **실무에서는:** 버전이 지정된 데이터셋과 함께 데이터시트를 게시하고, 질문에 답할 수 있는 사람을 식별하며, 제외된 인구 집단과 변환을 기록하고, 데이터셋이 변경될 때 문서를 업데이트합니다.
- **흔한 혼동:** 데이터시트는 증거와 의도된 용도를 문서화합니다. 이는 라이선스, 품질 보증, 또는 배포별 평가의 대체물이 아닙니다.
- **배울 곳:** [Model, System, and Dataset Cards](../phases/18-ethics-safety-alignment/26-model-system-dataset-cards/)
- **관련 용어:** Data Lineage, Data Provenance, Model Card, Dataset Split
- **출처:** [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

### Deadline Propagation - 마감 시간 전파
- **분류:** 신뢰성 및 운영
- **실제 의미:** 남은 엔드투엔드 시간 예산을 후속 호출에 전달하여, 각 의존성이 원본 요청이 유용하게 대기할 수 있는 시간이 얼마나 남았는지 알 수 있도록 하는 것입니다.
- **왜 중요한가:** 독립적인 타임아웃은 사용자의 마감 시간을 초과할 수 있으며, 결과가 더 이상 유용하지 않은 후에도 버려진 작업이 용량을 소비하게 됩니다.
- **실무에서는:** 진입점에서 하나의 요청 마감 시간을 설정하고, 각 후속 호출에 대해 경과 시간을 차감하며, 만료된 작업을 취소하고, 예산을 소진한 경계를 기록합니다.
- **흔한 혼동:** 마감 시간은 절대적이거나 남은 완료 경계입니다. 재시도 지연은 다른 시도가 시작되는 시점을 제어하며, 그 동일한 예산 내에 맞춰야 합니다.
- **관련 용어:** Retry with Backoff, Retry Budget, Tail Latency, Service Level Objective (SLO)
- **출처:** [gRPC Deadlines](https://grpc.io/docs/guides/deadlines/)

### Decode Phase - 디코딩 단계
- **분류:** 인프라 및 서빙
- **실제 의미:** 입력 접두어가 처리된 후, 새로운 토큰을 한 단계씩 생성하는 자기회귀 추론의 반복 단계입니다.
- **왜 중요한가:** 디코딩 작업은 프리필과 다른 연산, 메모리, 스케줄링 특성을 가지므로, 단일 집계 지연 시간 수치로는 실제 서빙 병목 현상을 숨길 수 있습니다.
- **실무에서는:** 토큰 간 지연 시간과 출력 처리량을 별도로 측정하고, KV 캐시 점유율을 고려하며, 활성 디코딩이 새로운 프리필과 용량을 공유하는 혼합 워크로드를 테스트해야 합니다.
- **흔한 혼동:** 디코딩 단계는 인코더-디코더 모델의 디코더 구성 요소를 의미하지 않습니다. 이는 런타임 생성 단계를 지칭합니다.
- **배울 곳:** [Disaggregated Prefill and Decode](../phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/)
- **관련 용어:** Prefill, Autoregressive, KV Cache, Time per Output Token (TPOT)
- **출처:** [DistServe](https://arxiv.org/abs/2401.09670)

### Decoder - 디코더
- **분류:** 모델 및 추론
- **흔히 하는 말:** 모델의 출력 측입니다.
- **실제 의미:** 표현을 출력으로 매핑하는 구성 요소입니다. 인코더-디코더 트랜스포머에서 디코더는 마스킹된 자기 주의와 교차 주의를 사용하여 출력을 생성합니다. 디코더 전용 언어 모델은 단일 인과 스택에서 생성합니다.
- **관련 용어:** Encoder, Transformer, Autoregressive

### Decoding Strategy - 디코딩 전략
- **분류:** 모델 및 추론
- **실제 의미:** 모델의 다음 토큰 점수 시퀀스를 선택된 토큰과 완성된 출력으로 변환하는 알고리즘입니다.
- **왜 중요한가:** 탐욕 선택, 샘플링, 절단, 탐색은 동일한 로짓에서 품질, 다양성, 지연 시간, 재현성이 서로 다른 결과를 만들어낼 수 있습니다.
- **실무에서는:** 결과를 공정하게 비교할 수 있도록 평가 구성에서 작업의 디코딩 설정, 중단 규칙, 시드 동작을 정의해야 합니다.
- **흔한 혼동:** 디코딩은 출력 선택 방식을 변경합니다. 모델의 학습된 파라미터를 변경하거나 지식을 추가하지는 않습니다.
- **관련 용어:** Autoregressive, Temperature, Top-k Sampling, Nucleus Sampling (Top-p)
- **출처:** [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

### Defense in Depth - 심층 방어
- **분류:** 보안 및 거버넌스
- **실제 의미:** 여러 시스템 경계에서 독립적인 예방, 탐지, 시정 통제를 사용하여 하나의 통제가 실패해도 결과가 결정되지 않도록 하는 것.
- **왜 중요한가:** AI 시스템은 확률적 모델, 신뢰할 수 없는 콘텐츠, 도구, 외부 서비스를 결합하므로 단일 필터나 프롬프트는 보안 경계로서 부적합합니다.
- **실무에서는:** 지시 통제를 좁은 권한, 샌드박싱, 스키마 검증, 중대한 작업에 대한 승인, 모니터링, 그리고 테스트된 복구 경로와 함께 조합하여 사용해야 합니다.
- **흔한 혼동:** 통제가 많다고 해서 자동으로 더 좋은 것은 아닙니다. 각 계층은 서로 다른 실패 모드를 다루고, 동일한 가정을 반복하지 않으며 테스트 가능해야 합니다.
- **관련 용어:** Guardrails, Sandbox, Least Privilege, Trust Boundary
- **출처:** [NIST Glossary: Defense in Depth](https://csrc.nist.gov/glossary/term/defense_in_depth)

### Delegation - 위임
- **분류:** 에이전트 및 도구
- **실제 의미:** 필요한 컨텍스트, 권한, 출력 계약, 반환 조건과 함께 제한된 하위 작업을 다른 사람이나 에이전트에게 할당하는 것.
- **왜 중요한가:** 명시적인 위임은 소유권, 범위, 결과 통합 능력을 잃지 않으면서 전문화와 병렬 작업을 가능하게 합니다.
- **실무에서는:** 리뷰어 에이전트에게 정확한 파일, 평가 기준, 증거, 마감 기한을 제공하고, 주요 산출물을 조용히 수정하지 않고 발견 사항을 반환하도록 요구해야 합니다.
- **흔한 혼동:** 다른 에이전트에게 모호한 메시지를 보내는 것은 신뢰할 수 있는 위임이 아닙니다. 수신자는 범위 계약과 정의된 반환 절차가 필요합니다.
- **관련 용어:** Scope Contract, Handoff, Reviewer Agent, Orchestration
- **출처:** [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

### Dense Retrieval - 밀집 검색
- **분류:** 검색 및 생성
- **실제 의미:** 쿼리와 후보를 벡터 표현으로 임베딩하고 유사도 함수에 따라 후보를 순위 매기는 1단계 검색.
- **왜 중요한가:** BM25와 같은 어휘적 방법을 보완하며, 정확한 단어를 거의 공유하지 않는 패러프레이즈와 의미적 일치 항목을 검색할 수 있습니다.
- **실무에서는:** 도메인에 적합한 임베딩 모델을 학습하거나 선택하고, 후보 벡터를 인덱싱하며, 결과를 생성 단계에 연결하기 전에 검색 재현율(retrieval recall)을 평가합니다.
- **흔한 혼동:** Dense retrieval은 리랭커(reranker)가 아닙니다. Dense retrieval은 전체 컬렉션을 검색하는 반면, 리랭커는 더 작은 후보 집합의 점수를 재계산합니다.
- **관련 용어:** Embedding, Semantic Search, BM25, Hybrid Retrieval
- **출처:** [Dense Passage Retrieval](https://aclanthology.org/2020.emnlp-main.550/)

### Diffusion Model - 확산 모델
- **분류:** 모델 및 추론
- **흔히 하는 말:** 노이즈로부터 이미지를 생성하는 모델입니다.
- **실제 의미:** 점진적인 노이즈화 과정과 학습된 역과정(reverse process)을 중심으로 학습된 생성 모델입니다. 샘플링은 보통 노이즈에서 시작하여 반복적인 디노이징(denoising) 단계를 적용하며, 때로는 학습된 잠재 공간(latent space)에서 수행됩니다.
- **흔한 혼동:** Diffusion은 일반적인 생성 프레임워크이며, 이미지 전용 기법이 아닙니다.
- **관련 용어:** Latent Space, VAE (Variational Autoencoder), Inference

### Disaggregated Serving - 분산 서빙
- **분류:** 인프라 및 서빙
- **실제 의미:** Prefill과 decode 작업을 각각 독립적으로 프로비저닝된 워커 풀(worker pool)에서 실행하고, 두 풀 간에 필요한 attention state를 전송하는 서빙 아키텍처입니다.
- **왜 중요한가:** Prefill과 decode는 하드웨어에 서로 다른 부하를 가하므로, 독립적인 풀을 각각의 병목(bottleneck)에 맞춰 크기를 조정하고 스케줄링할 수 있으며, 하나의 큐에서 경쟁하지 않습니다.
- **실무에서는:** 상태 전송(state-transfer) 비용을 측정하고, 호환되는 모델 버전을 통해 요청을 라우팅하며, 각 풀의 자체 수요 신호에 따라 확장하고, 단계 간 장애 복구(failure recovery)를 테스트합니다.
- **흔한 혼동:** Disaggregation은 런타임 단계를 분리합니다. 한 단계 내에서 하나의 모델을 tensor 병렬이나 pipeline 병렬 샤드(shard)로 나누는 것이 아닙니다.
- **배울 곳:** [Disaggregated Prefill and Decode](../phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/)
- **관련 용어:** Prefill, Decode Phase, Model Serving, Goodput
- **출처:** [DistServe](https://arxiv.org/abs/2401.09670)

### Distribution Shift - 분포 이동
- **분류:** 평가 및 안전
- **실제 의미:** 시스템을 구축하거나 평가하는 데 사용된 데이터 분포와 배포 후 encountered되는 분포 간의 차이입니다.
- **왜 중요한가:** 모델이 홀드아웃 테스트를 통과하더라도 사용자, 작업, 언어, 도구, 운영 조건이 변경되면 실패할 수 있습니다.
- **실무에서는:** 예상 배포 슬라이스를 정의하고, 슬라이스별로 성능 및 입력 특성을 모니터링하며, 새로운 실패 사례를 버전 관리되는 평가 세트에 추가합니다.
- **흔한 혼동:** 분포 이동은 항상 모델 드리프트를 의미하지는 않습니다. 모델은 변하지 않았지만 환경이나 사용자 집단이 변할 수 있습니다.
- **관련 용어:** Dataset Split, Eval Set, Overfitting, Model Card
- **출처:** [WILDS](https://proceedings.mlr.press/v139/koh21a.html)

### DPO (Direct Preference Optimization) - DPO (직접 선호 최적화)
- **분류:** 수학 및 학습
- **흔히 하는 말:** 별도의 보상 모델 단계 없이 수행하는 선호 학습.
- **실제 의미:** 참조 정책을 기준으로 선호 응답과 기각 응답 쌍을 사용하여 정책을 직접 학습하는 선호 최적화 목적 함수입니다. 이 단계에서는 명시적인 보상 모델과 강화 학습 루프를 실행하지 않습니다.
- **흔한 혼동:** DPO는 여전히 선호 데이터의 품질과 범위에 의존하며, 평가 및 정렬 위험을 제거하지는 않습니다.
- **배울 곳:** [Direct Preference Optimization](../phases/10-llms-from-scratch/08-dpo/)
- **출처:** [Direct Preference Optimization paper](https://arxiv.org/abs/2305.18290)
- **관련 용어:** RLHF (Reinforcement Learning from Human Feedback), SFT (Supervised Fine-Tuning), Alignment

### Dropout - 드롭아웃
- **분류:** 수학 및 학습
- **흔히 하는 말:** 활성화 값을 무작위로 끄는 것.
- **실제 의미:** 학습 중 활성화 값의 일부 비율을 무작위로 0으로 설정하여 네트워크가 하나의 활성화 경로에 의존하지 않도록 장려합니다. 표준 추론에서는 일반적으로 비활성화되지만, Monte Carlo dropout은 불확실성을 추정하기 위해 의도적으로 이를 활성화 상태로 유지합니다.
- **관련 용어:** Overfitting, Weight Decay, Activation Function

### Durable Execution - 내구성 있는 실행
- **분류:** 에이전트 및 도구
- **실제 의미:** 워크플로우의 상태와 완료된 단계가 프로세스 충돌, 재시작, 긴 대기 시간에도 살아남도록 실행하며, 확인된 사이드 이펙트를 다시 수행하지 않습니다.
- **왜 중요한가:** 에이전트 작업은 모델 호출, 도구, 승인 및 외부 시스템을 포함하는 경우가 많습니다. 일시적인 프로세스는 진행 상황의 유일한 기록이 되어서는 안 됩니다.
- **실무에서는:** 각 워크플로 전환을 영속화하고, 외부 쓰기에는 멱등성 키를 사용하며, 워커가 재시작된 후 최신 체크포인트에서 복원하십시오.
- **흔한 혼동:** 내구성 있는 실행(Durable execution)은 모든 작업을 자동으로 안전하게 만들지 않습니다. 사이드 이펙트에는 여전히 멱등성과 보상 규칙이 필요합니다.
- **관련 용어:** Checkpoint, Agent State, Idempotency, Approval Gate

### Dynamic Batching - 동적 배치
- **분류:** 인프라 및 서빙
- **실제 의미:** 대기열에 있는 요청을 호환되는 형태, 최대 크기, 우선순위 및 허용된 대기열 지연에 따라 추론 배치를 구성하는 런타임 정책입니다.
- **왜 중요한가:** 요청을 그룹화하면 하드웨어 활용도를 개선할 수 있지만, 트래픽이 희소하거나 요청이 크게 다를 경우 배치를 기다리는 것이 지연 시간을 악화시킬 수 있습니다.
- **실무에서는:** 측정된 지연 시간 목표를 기반으로 대기열 지연 및 배치 한도를 설정하고, 호환되지 않는 요청 형태를 분리하며, 현실적인 도착률에서 처리량과 꼬리 지연(tail latency)을 비교하십시오.
- **흔한 혼동:** 동적 배치는 대기열에 있는 작업으로 배치를 조립합니다. 연속 배치는 자기회귀 생성이 이미 실행 중일 때 멤버십을 변경합니다.
- **배울 곳:** [vLLM Serving Internals](../phases/17-infrastructure-and-production/04-vllm-serving-internals/)
- **관련 용어:** Admission Control, Continuous Batching, Saturation, Tail Latency
- **출처:** [NVIDIA Triton: Models and Schedulers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#scheduling-and-batching)

## E

### Early Fusion - 초기 융합
- **분류:** 멀티모달 시스템
- **실제 의미:** 대부분의 작업 특화 모델링이 발생하기 전에 여러 모달리티의 원시 또는 저수준 표현을 결합하는 것입니다.
- **왜 중요한가:** 초기 상호작용은 세밀한 교차 모달리티 관계를 드러낼 수 있지만, 호환되는 표현과 정렬 및 누락된 입력에 대한 주의 깊은 처리가 필요합니다.
- **실무에서는:** 각 모달리티를 선언된 토큰 또는 특징 표현으로 변환하고, 소스 및 위치 마커를 보존하며, 공유 백본(backbone) 전에 융합하고, 단일 모달리티 및 후기 융합(late-fusion) 기준선과 비교하십시오.
- **흔한 혼동:** Early fusion은 아키텍처 내에서 스트림이 결합되는 위치를 설명합니다. 이는 모델이 스트림 간에 유용한 정렬(alignment)을 학습한다는 것을 보장하지 않습니다.
- **배울 곳:** [Chameleon Early-Fusion Tokens](../phases/12-multimodal-ai/11-chameleon-early-fusion-tokens/)
- **관련 용어:** Late Fusion, Multimodal Fusion, Modality Alignment, Token
- **출처:** [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818); [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)

### Eigenvalue - 고유값
- **분류:** 수학 및 학습
- **흔히 하는 말:** PCA에서 사용되는 행렬의 속성입니다.
- **실제 의미:** 선형 변환이 대응하는 비영(zero가 아닌) 고유벡터의 방향을 변경하지 않으면서 스케일링하는 정도를 설명하는 스칼라 값입니다. 공분산 행렬 PCA에서는 더 큰 고유값이 더 큰 분산을 가진 방향에 대응합니다.
- **관련 용어:** Tensor, Feature, Latent Space

### Embedding - 임베딩
- **분류:** 데이터 및 표현
- **흔히 하는 말:** 의미를 나타내는 벡터입니다.
- **실제 의미:** 이산적인 항목(단어, 이미지, 사용자)을 연속 공간의 밀집 벡터로 학습된 매핑으로 변환하는 것을 의미하며, 유사한 항목은 서로 가깝게 위치하게 됩니다.
- **흔한 혼동:** 유사성은 모델, 학습 목표 및 메트릭에 따라 달라집니다. 한 임베딩 공간에서의 거리가 다른 임베딩 공간으로 이어지지 않습니다.
- **이름의 유래:** 항목들이 기하학적 표현 공간에 배치, 즉 '임베딩(embedded)'되기 때문입니다.
- **배울 곳:** [Embeddings](../phases/11-llm-engineering/04-embeddings/)
- **관련 용어:** Cosine Similarity, Semantic Search, Vector Database

### Encoder - 인코더
- **분류:** 모델 및 추론
- **흔히 하는 말:** 모델의 입력 측입니다.
- **실제 의미:** 입력을 표현(representation)으로 변환하는 구성 요소입니다. 트랜스포머 인코더는 일반적으로 masks의 적용을 받는 비-causal self-attention을 사용하므로, 각 위치는 입력 전체의 컨텍스트를 반영할 수 있습니다.
- **흔한 혼동:** Encoder-only 모델은 일반적으로 autoregressive text generation에 사용되지 않지만, task heads를 통해 출력물을 생성할 수 있습니다.
- **관련 용어:** Decoder, Transformer, Embedding

### Epoch - 에포크
- **분류:** 수학 및 학습
- **흔히 하는 말:** 학습 데이터를 한 번 순회하는 것.
- **실제 의미:** 정의된 학습 데이터셋을 한 번 순회하는 것. 분산 학습이나 샘플링 학습에서는 에포크(epoch)의 정확한 구현 방식이 데이터 로더와 샘플링 정책에 따라 달라집니다.
- **흔한 혼동:** 에포크(epoch)를 많이 수행한다고 해서 일반화 성능이 반드시 좋아지지 않습니다. 홀드아웃(held-out) 데이터로 평가해야 합니다.
- **관련 용어:** Batch Size, Overfitting, Eval Set

### Error Budget - 오류 예산
- **분류:** 신뢰성 및 운영
- **실제 의미:** 서비스 수준 목표(SLO)의 측정 기간 동안, 목표가 소진되기 전까지 허용되는 서비스 실패의 양.
- **왜 중요한가:** 신뢰성 및 제품 작업에 공유된 의사결정 경계를 제공합니다. 팀은 잔여 예산을 변경 작업에 투입할 수 있으며, 사용자 가시적 실패가 예산을 소진할 때는 위험을 늦출 수 있습니다.
- **실무에서는:** SLO에서 예산을 도출하고, 원인과 사용자 세그먼트별로 소진율을 추적하며, 소진 전에 릴리스 조치를 정의하고, 사고 발생 후 회계 처리를 리셋하지 않도록 합니다.
- **흔한 혼동:** 에러 버짓(error budget)은 사고를 유발하기 위한 할당량이 아닙니다. 이는 사용자 중심의 신뢰성 목표에서 도출된 운영 정책입니다.
- **관련 용어:** Service Level Objective (SLO), Service Level Indicator (SLI), Availability, Incident Response
- **출처:** [Google SRE Workbook: Error Budget Policy](https://sre.google/workbook/error-budget-policy/)

### Eval Set - 평가 세트
- **분류:** 평가 및 안전
- **다른 이름:** Evaluation set
- **실제 의미:** AI 시스템을 정의된 능력이나 위험에 대해 측정하기 위해 사용되는, 버전 관리된 입력, 기대 속성, 채점 규칙 및 메타데이터의 집합.
- **왜 중요한가:** 반복 가능한 세트는 모호한 품질 주장을 비교 가능한 증거로 전환하며, 프롬프트, 모델, 도구 또는 검색(retrieval)이 변경된 후의 회귀(regression)를 포착합니다.
- **실무에서는:** 대표적인 지원 질문, 적대적 지시문, 기대 인용문 및 실패 레이블을 개발 예제와 분리된 검토된 데이터셋에 유지합니다.
- **흔한 혼동:** 개발용 평가는 반복적인 개선을 안내하며, 최종 홀드아웃 테스트는 선택이 고정된 후의 성능을 추정하고, 표준화된 벤치마크는 공유된 프로토콜 하에서의 비교를 지원합니다. 어떤 홀드아웃 세트에 대해서도 반복적으로 튜닝하면 테스트 정보가 유출되어 결과가 과장됩니다.
- **배울 곳:** [Eval-Driven Agent Development](../phases/14-agent-engineering/30-eval-driven-agent-development/)
- **관련 용어:** Evaluation (Eval), Regression Test, LLM-as-a-Judge, Verification Gate

### Evaluation (Eval) - 평가
- **분류:** 평가 및 안전
- **다른 이름:** Eval
- **실제 의미:** 명시적인 성공 기준, 데이터, 채점자 및 검토 절차를 사용하여 대표 작업에서 모델 또는 시스템의 행동을 측정하는 정의된 프로세스입니다.
- **왜 중요한가:** 성공이 몇몇 데모에서의 주관적인 인상일 뿐이라면 신뢰성을 개선할 수 없습니다.
- **실무에서는:** 검색을 변경하기 전후로 동일한 고객 지원 시나리오를 실행하고, 정확성과 인용 지원 여부를 채점하며, 실패를 범주별로 검토합니다.
- **흔한 혼동:** 벤치마크 점수는 하나의 평가 결과일 뿐, 생산 품질에 대한 완전한 설명이 아닙니다.
- **배울 곳:** [LLM Evaluation](../phases/11-llm-engineering/10-evaluation/)
- **관련 용어:** Eval Set, LLM-as-a-Judge, Cost per Successful Task, Regression Test

### Exact Match (EM) - 정확 일치
- **분류:** 평가 및 안전
- **실제 의미:** 출력의 정규화된 표현이 허용된 참조 답변과 정확히 일치할 때만 정답으로 계산하는 지표입니다.
- **왜 중요한가:** 하나의 표준 답이 있는 작업에서는 결정적이고 감사하기 쉽지만, 부분 점수를 제공하지 않습니다.
- **실무에서는:** 평가 전에 정규화 및 모든 허용된 참조를 정의한 후, 여러 출력이 유효할 수 있는 경우 정확 일치와 작업별 검사를 병행합니다.
- **흔한 혼동:** 낮은 정확 일치 점수는 무해한 형식 차이로 인한 것일 수 있으며, 일치하는 문자열이 문맥상 지원되지 않거나 안전하지 않을 수 있습니다.
- **관련 용어:** ROUGE, Eval Set, Structured Output, Pass@k
- **출처:** [SQuAD](https://aclanthology.org/D16-1264/)

### Expert Parallelism - 전문가 병렬화
- **분류:** 인프라 및 서빙
- **실제 의미:** Mixture-of-Experts 서브네트워크를 여러 장치에 분산 배치하고, 각 토큰의 활성값을 선택된 전문가(expert)가 호스팅하는 장치로 라우팅하는 것.
- **왜 중요한가:** Sparse experts는 모든 토큰에 대해 모든 전문가를 실행하지 않으면서 모델 용량을 증가시키지만, 라우팅은 통신, 부하 균형 및 배치 제약 조건을 도입합니다.
- **실무에서는:** 전문가별 토큰 분포를 측정하고, 통신 대역폭을 확보하며, 오버플로우를 의도적으로 제한하거나 라우팅하고, 트래픽이 불균등한 전문가 수요를 생성할 때 품질을 테스트합니다.
- **흔한 혼동:** Expert parallelism은 라우터가 선택한 전문가를 파티셔닝합니다. Tensor parallelism은 레이어 내부의 텐서 연산을 파티셔닝합니다.
- **배울 곳:** [Mixture of Experts](../phases/07-transformers-deep-dive/11-mixture-of-experts/)
- **관련 용어:** MoE (Mixture of Experts), Tensor Parallelism, Pipeline Parallelism, Model Serving
- **출처:** [GShard](https://arxiv.org/abs/2006.16668)

## F

### Feature - 특징
- **분류:** 데이터 및 표현
- **흔히 하는 말:** 데이터셋의 한 열(column).
- **실제 의미:** 데이터의 개별 측정 가능한 속성입니다. Classical ML에서는 수동으로 features를 엔지니어링합니다. Deep learning에서는 네트워크가 원시 데이터로부터 features를 자동으로 학습합니다.
- **흔한 혼동:** 저장된 열(column)은 여러 유용한 features를 포함할 수 있으며, 학습된 표현(representation)은 단순한 인간 라벨이 없는 features를 포함할 수 있습니다.
- **관련 용어:** Embedding, Latent Space, Inductive Bias

### Few-Shot - 소수 예시
- **분류:** 프롬팅 및 컨텍스트
- **흔히 하는 말:** 프롬프트에 모델에게 몇 가지 예시를 제공합니다.
- **실제 의미:** 목표 입력 전에 소량의 시연(demonstrations)을 포함하는 in-context learning으로, 모델이 원하는 작업, 형식 또는 결정 경계를 추론할 수 있게 합니다.
- **왜 중요한가:** 예시의 품질과 커버리지가 보편적인 예시 개수보다 더 중요합니다. 빈약하거나 모순되는 시연은 신뢰성을 감소시킬 수 있습니다.
- **관련 용어:** Zero-Shot, In-Context Learning, Prompt Engineering, Context Window

### Fine-tuning - 미세 조정
- **분류:** 수학 및 학습
- **흔히 하는 말:** 모델에 데이터를 학습시키는 것.
- **실제 의미:** 사전 학습된 파라미터를 더 좁은 데이터셋이나 목적 함수에 대해 계속 학습하는 것. 방법에 따라 모든 파라미터, 선택된 파라미터, 또는 추가된 어댑터 파라미터를 업데이트할 수 있습니다.
- **왜 중요한가:** 파인 튜닝은 행동, 스타일, 형식, 또는 작업 성능을 적응시킬 수 있지만, 사실이 최신 상태여야 하거나 추적 가능해야 할 때 검색(retrieval)을 대체할 수 있는 신뢰할 수 있는 방법은 아닙니다.
- **흔한 혼동:** 파인 튜닝은 인코딩된 지식에 영향을 미칠 수 있지만, 모델 내부의 검색 가능한 데이터베이스에 레코드를 단순히 추가하는 것은 아닙니다.
- **배울 곳:** [Fine-Tuning and LoRA](../phases/11-llm-engineering/08-fine-tuning-lora/)
- **관련 용어:** SFT (Supervised Fine-Tuning), LoRA (Low-Rank Adaptation), QLoRA, RAG (Retrieval-Augmented Generation)

### Flaky Test - 불안정한 테스트
- **분류:** AI 네이티브 개발
- **실제 의미:** 코드나 의도된 테스트 환경에 관련 변경이 없음에도 동일한 실행에서 통과와 실패를 반복할 수 있는 테스트.
- **왜 중요한가:** 불안정성(flakiness)은 검증 게이트를 약화시키며, 실제 실패를 무시하거나 거짓 통과를 얻을 때까지 재시도하도록 사람이나 에이전트를 훈련시킬 수 있습니다.
- **실무에서는:** 실패한 시드(seed)와 환경을 보존하고, 소유자와 마감 기한이 있는 경우에만 격리(quarantine)하며, 제어되지 않은 시간, 동시성, 네트워크, 순서, 또는 공유 상태 의존성을 수정합니다.
- **흔한 혼동:** 간헐적인 제품 버그를 일관되게 드러내는 테스트는 가치 있는 증거이며, 반드시 불안정한 테스트(flaky test)는 아닙니다.
- **관련 용어:** Regression Test, Test Oracle, Retry with Backoff, Verification Gate
- **출처:** [De-Flake Your Tests](https://conferences.computer.org/icsme/pdfs/ICSME2020-1oOutvkGTwF4GyVvNtr3Mm/561900a736/561900a736.pdf)

### FlashAttention
- **분류:** 인프라 및 서빙
- **실제 의미:** 가속기 메모리 계층 간의 전송을 줄이고 고대역폭 메모리(HBM)에서 전체 어텐션 행렬의 물리적 생성(materialization)을 피하기 위해 연산을 타일링(tiling)하는 정확한 어텐션 알고리즘.
- **왜 중요한가:** 특히 긴 시퀀스의 경우 어텐션은 연산량보다 메모리 이동에 의해 제한될 수 있으므로, IO-aware 커널은 사용 가능한 속도와 메모리 효율성을 개선할 수 있습니다.
- **실무에서는:** 모델의 shape, mask, dtype, 하드웨어가 지원하는 커널을 사용하고, 수치 허용 오차를 검증하며, 논문 결과를 고정된 배수로 인용하는 대신 엔드투엔드 지연 시간을 벤치마킹합니다.
- **흔한 혼동:** FlashAttention은 attention을 계산하는 방식을 변경하는 것이지, 목표하는 수학적 attention 결과를 변경하는 것이 아닙니다. 이는 KV 캐싱 및 양자화와 별개입니다.
- **배울 곳:** [KV Cache and Flash Attention](../phases/07-transformers-deep-dive/12-kv-cache-flash-attention/)
- **관련 용어:** Attention, Self-Attention, KV Cache, Mixed Precision
- **출처:** [FlashAttention](https://arxiv.org/abs/2205.14135)

### Function Calling - 함수 호출
- **분류:** Agents & tools
- **흔히 하는 말:** 도구를 사용하는 모델.
- **실제 의미:** 모델이 도구 이름과 인수를 지정하는 구조화된 요청을 생성하는 공급자 또는 애플리케이션 인터페이스입니다. 애플리케이션 코드가 요청을 검증하고, 연산을 수행하며, 결과를 반환하여 모델의 다음 단계로 전달할 수 있습니다.
- **흔한 혼동:** 모델은 함수 호출을 요청합니다. 신뢰할 수 있는 코드가 이를 실행할지 여부와 실행 방식을 결정합니다. Function calling만으로는 완전한 agent가 되지 않습니다.
- **배울 곳:** [Function Calling](../phases/11-llm-engineering/09-function-calling/)
- **관련 용어:** Structured Output, Tool Contract, Agent, MCP (Model Context Protocol)

## G

### GAN (Generative Adversarial Network) - GAN (생성적 적대 신경망)
- **분류:** Models & inference
- **흔히 하는 말:** 학습 중에 경쟁하는 두 개의 신경망.
- **실제 의미:** 생성자 네트워크는 현실적인 데이터를 생성하려고 시도하고, 판별자 네트워크는 진짜와 가짜를 구분하려고 시도합니다. 두 네트워크는 함께 학습됩니다. 생성자는 판별자를 속이는 능력이 향상되고, 판별자는 가짜를 탐지하는 능력이 향상됩니다.
- **관련 용어:** Loss Function, Latent Space, Diffusion Model

### Goodput - 순수 처리량
- **분류:** Infrastructure & serving
- **실제 의미:** 명시된 워크로드 하에서 time-to-first-token 및 per-token 지연 시간 목표와 같은 정의된 서비스 제약 조건을 충족하는 완료된 요청의 비율입니다.
- **왜 중요한가:** Raw throughput는 증가할 수 있지만, 사용자는 더 많은 지연된 요청을 경험할 수 있습니다. Goodput은 서비스 계약 조건을 충족하는 작업만 계산합니다.
- **실무에서는:** 요청 분포와 지연 임계값을 선언하고, 규정 준수 완료 건만 집계하며, 총 집계율 옆에 백분위수를 보고하고, 서로 다른 목표 하에서 시스템을 비교하지 마십시오.
- **흔한 혼동:** Goodput은 모든 완료된 처리량이 아니며, 모델의 보편적인 속성이 아닙니다. 이는 워크로드와 성공 임계값에 따라 달라집니다.
- **배울 곳:** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **관련 용어:** Service Level Objective (SLO), Time to First Token (TTFT), Time per Output Token (TPOT), Cost per Successful Task
- **출처:** [DistServe](https://arxiv.org/abs/2401.09670)

### GPT
- **분류:** 모델 및 추론
- **흔히 하는 말:** 모든 채팅봇의 일반적 명칭.
- **실제 의미:** Generative Pre-trained Transformer는 시퀀스 예측 목적 함수로 사전 학습된 생성형 트랜스포머 모델 계열을 지칭하는 레이블입니다. 제품명과 모델 아키텍처는 상호 교환 가능한 것으로 취급되어서는 안 됩니다.
- **이름의 유래:** Generative는 출력 생성을, pre-trained는 초기 광범위한 학습 단계를, transformer는 아키텍처 계열을 각각 설명합니다.
- **관련 용어:** Transformer, Autoregressive, LLM (Large Language Model)

### Graceful Degradation - 우아한 저하
- **분류:** 신뢰성 및 운영
- **실제 의미:** 모든 요청이 실패하는 대신 선택적 품질, 기능, 신선도, 워크로드를 줄임으로써 용량이나 의존성이 손상될 때 제한된 핵심 서비스를 유지하는 것.
- **왜 중요한가:** AI 시스템은 종종 여러 느리거나 불안정한 구성 요소에 의존하므로, 명시적인 축소 모드는 부분적 장애 발생 시 필수적인 사용자 결과를 보호할 수 있습니다.
- **실무에서는:** 비활성화할 수 있는 기능을 사전에 정의하고, 축소 모드를 운영자에게 가시적으로 표시하며, 안전성 검사를 보호하고, 의존성 장애 하에서 폴백을 테스트하며, 전체 서비스를 의도적으로 복원하십시오. 정확성, 안전성, 신선도, 또는 약속된 계약이 실질적으로 변경될 경우 사용자에게 알리십시오.
- **흔한 혼동:** 우아한 저하(graceful degradation)는 아무 일도 일어나지 않은 것처럼 더 나쁜 답변을 조용히 반환하는 것이 아닙니다. 운영자는 항상 가시성을 확보해야 하며, 사용자가 축소된 모드에서 결과나 서비스 계약이 실질적으로 변경될 경우 고지를 받아야 합니다.
- **배울 곳:** [Production LLM Application](../phases/11-llm-engineering/13-production-app/)
- **관련 용어:** Circuit Breaker, Load Shedding, Model Router, Availability
- **출처:** [Google SRE: Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)

### Gradient - 기울기
- **분류:** 수학 및 학습
- **흔히 하는 말:** 손실의 기울기.
- **실제 의미:** 가장 급격한 증가 방향을 가리키는 부분 미분들의 벡터입니다. ML에서는 손실을 최소화하기 위해 기울기의 반대 방향으로 이동합니다(gradient descent).
- **흔한 혼동:** 옵티마이저는 단순한 음의 기울기 단계 대신 기울기를 변환, 평균화, 클리핑(clipping)하거나 적응적으로 처리할 수 있습니다.
- **관련 용어:** Backpropagation, Gradient Descent, Optimizer

### Gradient Accumulation - 기울기 누적
- **분류:** 수학 및 학습
- **실제 의미:** 여러 마이크로배치(microbatch)의 기울기를 합산하거나 평균한 후 단일 옵티마이저 업데이트를 수행하는 것입니다.
- **왜 중요한가:** 하나의 장치에서 모든 예제와 활성값을 동시에 처리할 수 없을 때, 더 큰 유효 배치(effective batch)를 근사할 수 있게 해줍니다.
- **실무에서는:** 손실을 일관되게 스케일링하고, 선택한 마이크로배치 수 이후에만 옵티마이저를 호출하며, 정규화나 분산 동기화가 동작을 변경하는지 측정합니다.
- **흔한 혼동:** Gradient accumulation은 단계별 활성 메모리를 줄여주지만, 전체 배치를 동시에 처리하는 모든 특성을 재현하지는 못합니다.
- **관련 용어:** Batch Size, Mixed Precision, Optimizer, Backpropagation
- **출처:** [PyTorch AMP examples: Gradient accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)

### Gradient Clipping - 기울기 클리핑
- **분류:** 수학 및 학습
- **실제 의미:** 옵티마이저 업데이트 전에 선택한 임계값을 초과하는 경우, 기울기 값이나 그 결합된 노름(norm)을 제한하는 것입니다.
- **왜 중요한가:** 비정상적으로 큰 기울기가 학습 단계를 불안정하게 만들고 비유한(non-finite) 값을 생성하는 것을 방지할 수 있습니다.
- **실무에서는:** 클리핑되지 않은 노름(norm)을 기록하고, 혼합 정밀도(mixed-precision) 그래디언트의 스케일링을 되돌린 후 클리핑을 수행하며, 불안정성 진단의 대체 수단으로 클리핑을 취급하지 말고 반복적인 클리핑을 조사하십시오.
- **흔한 혼동:** 클리핑은 업데이트의 크기를 제어합니다. 이는 잘못된 데이터, 손상된 손실(loss), 또는 일관되게 부적절한 학습률을 복구하지 못합니다.
- **관련 용어:** Gradient, NaN (Not a Number), Mixed Precision, Learning Rate
- **출처:** [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063)

### Gradient Descent - 경사 하강법
- **분류:** 수학 및 학습
- **흔히 하는 말:** 손실 표면(loss surface)을 따라 내려가는 것.
- **실제 의미:** 목적 함수의 음의 그래디언트를 사용하여 파라미터를 이동시키는 최적화 업데이트의 계열로, 일반적으로 전체 데이터셋이 아닌 배치(batch)로부터 추정됩니다.
- **관련 용어:** Gradient, Learning Rate, Optimizer

### Grounding - 그라운딩
- **분류:** 검색 및 생성
- **실제 의미:** 생성된 답변이나 행동을 시스템이 식별하고 확인할 수 있는 증거, 상태, 또는 관찰과 연결하는 것.
- **왜 중요한가:** Grounding은 시스템에 무제약 생성(unconstrained generation) 이상의 근거를 제공하며, 근거 없는 주장의 탐지를 용이하게 합니다.
- **실무에서는:** 정책 섹션을 검색하고, 답변이 이를 인용하도록 요구하며, 인용된 구절이 지지하지 않는 주장을 거부하십시오.
- **흔한 혼동:** 프롬프트에 문서를 추가하는 것은 Grounding의 기회를 만듭니다. 이는 모델이 이를 올바르게 사용할 것을 보장하지는 않습니다.
- **배울 곳:** [Retrieval-Augmented Generation](../phases/11-llm-engineering/06-rag/)
- **관련 용어:** RAG (Retrieval-Augmented Generation), Hallucination, Verification Gate, Reranker

### Guardrails - 가드레일
- **분류:** 평가 및 안전
- **흔히 하는 말:** 모델 주변의 안전 필터.
- **실제 의미:** 입력, 도구 사용, 출력, 권한 및 에스컬레이션(escalation)을 제한하는 시스템 제어입니다. 스키마, 정책 검사, 분류기, 허용 목록(allowlist), 샌드박싱, 승인 및 사후 행동 검증이 포함될 수 있습니다.
- **왜 중요한가:** 단일 필터가 모든 실패 모드(failure modes)를 커버하지 못하므로, 위험도에 따라 제어 계층을 구성해야 합니다.
- **흔한 혼동:** 가드레일(guardrails)은 위험을 줄여 주지만, AI 시스템이 안전하다는 것을 증명하지는 않습니다.
- **배울 곳:** [Guardrails](../phases/11-llm-engineering/12-guardrails/)
- **관련 용어:** Least Privilege, Approval Gate, Sandbox, Evaluation (Eval)

## H

### Hallucination - 환각
- **분류:** 평가 및 안전
- **흔히 하는 말:** 모델이 거짓말을 하고 있다.
- **실제 의미:** 생성된 콘텐츠가 거짓이거나, 이용 가능한 증거에 의해 뒷받침되지 않거나, 작업의 진실 출처(source of truth)와 일치하지 않는 경우를 의미합니다. 출력 내용이 유창하고 모델이 속이려는 의도가 없는 경우에도 발생할 수 있습니다.
- **왜 중요한가:** 문장이 학습 데이터에 존재했는지 검사하는 것은 일반적으로 불가능하므로, 프로덕션 환경에서의 검사는 지원(support), 정확성, 추적 가능성에 초점을 맞춰야 합니다.
- **실무에서는:** 사실적 답변에 인용된 증거를 요구하며, 각 인용이 관련 주장에 실제로 뒷받침되는지 평가해야 합니다.
- **흔한 혼동:** 환각은 출력 품질의 실패이지, 모델의 의도에 대한 진단이 아닙니다.
- **관련 용어:** Grounding, RAG (Retrieval-Augmented Generation), Verification Gate

### Handoff - 핸드오프
- **분류:** AI 네이티브 개발
- **실제 의미:** 목표, 현재 상태, 증거, 결정 사항, 제약 조건 및 남은 작업을 보존하면서 사람이나 에이전트 간에 작업을 구조적으로 전달하는 것을 의미합니다.
- **왜 중요한가:** 좋은 인수인계는 다음 작업자가 긴 대화 기록(transcript)에서 전체 작업을 재구성하거나 완료된 작업을 반복하는 것을 방지합니다.
- **실무에서는:** 승인된 계획, 변경된 파일, 테스트 명령 및 결과, 미해결된 위험, 그리고 정확한 다음 조치를 간결한 작업 패킷으로 전달해야 합니다.
- **흔한 혼동:** 요약은 '무슨 일이 일어났는지'를 말합니다. 인수인계는 '어떤 상태가 권위 있는 상태(authoritative)인지'와 '다음에 무슨 일이 일어나야 하는지'도 포함합니다.
- **배울 곳:** [Multi-Session Handoff](../phases/14-agent-engineering/40-multi-session-handoff/)
- **관련 용어:** Agent State, Checkpoint, Scope Contract, Progressive Disclosure

### HNSW
- **분류:** 검색 및 생성
- **다른 이름:** Hierarchical Navigable Small World
- **실제 의미:** 벡터를 계층형 근접 그래프로 조직화하고, 거친 상위 계층에서 상세한 하위 계층으로 검색하는 근사 최근접 이웃 인덱스입니다.
- **왜 중요한가:** 완전한 비교가 너무 느린 대규모 환경에서 높은 재현율의 벡터 검색을 실용적으로 만드는 일반적인 방법입니다.
- **실무에서는:** 지연 시간, 메모리, Recall@K 목표를 기준으로 구축 및 쿼리 매개변수를 조정하고, 임베딩 버전이 변경되면 인덱스를 재구축합니다.
- **흔한 혼동:** HNSW는 인덱스 알고리즘이며, 유사도 메트릭, 임베딩 모델, 완전한 벡터 데이터베이스가 아닙니다.
- **관련 용어:** Approximate Nearest Neighbor (ANN), Vector Database, Embedding, Recall@K
- **출처:** [Efficient and Robust Approximate Nearest Neighbor Search Using HNSW](https://dl.acm.org/doi/10.1109/TPAMI.2018.2889473)

### Human-in-the-Loop (HITL) - 인간 개입 루프 (HITL)
- **분류:** 에이전트 및 도구
- **다른 이름:** 인간 감독, 인간 검토
- **실제 의미:** AI 주도 프로세스의 정의된 지점에서 사람이 판단, 수정, 승인 또는 에스컬레이션을 제공하는 워크플로 설계입니다.
- **왜 중요한가:** 인간 개입은 모든 단계 후의 정의되지 않은 폴백이 아니라, 고영향, 모호하거나 되돌릴 수 없는 경계에서 가장 유용합니다.
- **실무에서는:** 에이전트가 일상적인 요청을 자동으로 분류하도록 하되, 불확실하거나 고가치인 케이스는 증거와 제안된 조치와 함께 검토자에게 라우팅합니다.
- **흔한 혼동:** HITL은 시스템을 자동으로 안전하게 만들지 않습니다. 검토자에게는 시간, 맥락, 권한, 명확한 결정 기준이 필요합니다.
- **관련 용어:** Approval Gate, Verification Gate, Agent, Guardrails

### Hybrid Retrieval - 하이브리드 검색
- **분류:** 검색 및 생성
- **실제 의미:** 결과를 병합하거나 재순위화하기 전에 어휘 매칭과 밀집 벡터 유사도 등 다양한 방법의 신호를 결합하는 검색입니다.
- **왜 중요한가:** 정확한 식별자, 희소한 용어, 의미적 패러프레이즈는 서로 다른 방식으로 동작하므로, 하나의 검색 신호만으로는 유용한 증거를 놓칠 수 있습니다.
- **실무에서는:** BM25 스타일의 키워드 검색과 임베딩을 모두 사용하여 후보를 검색하고, 순위를 병합한 후 사용자 쿼리에 대해 결합된 집합을 재순위화합니다.
- **흔한 혼동:** 하이브리드 검색은 후보 신호를 결합합니다. 리랭커는 이미 검색된 후보에 두 번째 관련성 모델을 적용합니다.
- **배울 곳:** [Advanced RAG](../phases/11-llm-engineering/07-advanced-rag/)
- **관련 용어:** Semantic Search, Reranker, RAG (Retrieval-Augmented Generation), Embedding

### Hyperparameter - 하이퍼파라미터
- **분류:** 수학 및 학습
- **흔히 하는 말:** 조정하는 설정입니다.
- **실제 의미:** 모델 구조, 최적화, 데이터 처리 또는 추론을 형성하는 구성 선택이며, 일반적인 모델 파라미터로서 학습되는 것이 아닙니다. 예로는 학습률, 배치 크기, 레이어 수 및 디코딩 설정이 있습니다.
- **흔한 혼동:** 일부 하이퍼파라미터는 학습 전에 선택되지만, 다른 것들은 스케줄 중이나 추론 시점에 변경할 수 있습니다.
- **관련 용어:** Parameter, Learning Rate, Batch Size, Temperature

## I

### Idempotency - 멱등성
- **분류:** AI 네이티브 개발
- **실제 의미:** 동일한 식별자로 동일한 작업을 반복하더라도, 첫 번째 성공적인 적용 이후에는 추가적인 부작용이 발생하지 않는 속성입니다.
- **왜 중요한가:** 분산 에이전트 시스템에서는 재시도가 일반적입니다. 멱등성이 없으면, 불확실한 응답 하나로 인해 결제, 댓글, 배포 또는 레코드가 중복될 수 있습니다.
- **실무에서는:** 도구 요청에 멱등성 키를 첨부하고 완료된 결과를 저장하여, 재시도 시 다시 실행하지 않고 그 결과를 반환하도록 합니다.
- **흔한 혼동:** 멱등성은 모든 응답이 바이트 단위로 동일하다는 것을 의미하지 않습니다. 의도된 상태 변경이 중복되지 않는다는 것을 의미합니다.
- **출처:** [HTTP Semantics: idempotent methods](https://www.rfc-editor.org/rfc/rfc9110.html#name-idempotent-methods)
- **관련 용어:** Retry with Backoff, Durable Execution, Checkpoint

### Image Token - 이미지 토큰
- **분류:** 멀티모달 시스템
- **실제 의미:** 모델 고유의 시각적 단위로, 벡터나 이산 코드로 표현되며, 일반적으로 이미지 패치, 영역 또는 학습된 시각적 코드북 항목에서 파생됩니다.
- **왜 중요한가:** 시각 입력을 시퀀스로 변환하면 트랜스포머 스타일 구성 요소가 이미지를 텍스트나 다른 토큰화된 모달리티와 함께 처리할 수 있습니다.
- **실무에서는:** 토큰이 연속적인 패치인지 이산적인 코드인지 문서화하고, 공간 위치를 보존하며, 해상도와 종횡비 변경을 테스트하고, 모델의 입력 예산 내에서 시각 토큰을 계산하십시오.
- **흔한 혼동:** 이미지 토큰은 반드시 하나의 픽셀, 하나의 객체, 또는 하나의 고정된 물리적 영역을 의미하지는 않습니다. 그 범위는 시각 인코더나 토크나이저에 따라 결정됩니다.
- **배울 곳:** [Vision-Language Models](../phases/04-computer-vision/25-vision-language-models/)
- **관련 용어:** Patch Embedding, Token, VAE (Variational Autoencoder), Vision Transformer (ViT)
- **출처:** [Vision Transformer](https://arxiv.org/abs/2010.11929); [VQ-VAE](https://arxiv.org/abs/1711.00937)

### In-Context Learning - 인컨텍스트 학습
- **분류:** 프롬팅 및 컨텍스트
- **실제 의미:** 모델이 일반적인 파라미터 업데이트 없이 현재 입력에 제공된 지침, 예시, 패턴으로부터 행동을 적응시키는 것.
- **왜 중요한가:** 하나의 사전 학습된 모델이 가중치를 변경하지 않으면서 컨텍스트를 통해 새로운 작업을 수행할 수 있는 방식을 설명합니다.
- **실무에서는:** 타겟 입력 앞에 대표적인 시연(demonstration)을 배치하고, 순서 및 형식 변형을 테스트하며, 평가 예제를 시연과 분리하여 유지하십시오.
- **흔한 혼동:** In-context learning은 일시적인 조건부 설정(temporary conditioning)이며, 파인 튜닝(fine-tuning), 영구적인 기억, 모델이 의도된 규칙을 추론했다는 증거가 아닙니다.
- **관련 용어:** Few-Shot, Zero-Shot, Context Window, Prompt Engineering
- **출처:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

### Incident Response - 인시던트 대응
- **분류:** 신뢰성 및 운영
- **실제 의미:** 서비스, 데이터, 안전, 보안에 위협이 되는 사건을 감지, 분석, 격리(containing), 복구, 소통, 학습하는 조정된 프로세스.
- **왜 중요한가:** 사건 발생 시, 모델의 행동과 분산된 의존성으로 인해 실패 경계가 모호해지기 때문에, 즉흥적인 영웅적 행동보다 명확한 역할과 증거가 더 중요합니다.
- **실무에서는:** 심각도와 명령 역할을 정의하고, 추적 기록과 감사 기록을 보존하며, 유해한 조치를 중단하고, 영향을 전달하며, 복구 여부를 검증하고, 시정 조치를 완료까지 추적합니다.
- **흔한 혼동:** 사고 대응은 사건과 그 결과를 관리합니다. 근본 원인 분석과 장기적인 예방은 즉각적인 서비스가 복구된 이후에 계속됩니다.
- **배울 곳:** [SRE for AI](../phases/17-infrastructure-and-production/23-sre-for-ai/)
- **관련 용어:** Observability, Audit Log, Postmortem, Availability
- **출처:** [Google SRE: Managing Incidents](https://sre.google/sre-book/managing-incidents/); [NIST SP 800-61 Rev. 3](https://csrc.nist.gov/pubs/sp/800/61/r3/final)

### Indirect Prompt Injection - 간접 프롬프트 주입
- **분류:** 보안 및 거버넌스
- **실제 의미:** 웹 페이지, 문서, 이메일, 이미지 텍스트, 도구 결과 등 시스템이 검색하거나 관찰하는 콘텐츠를 통해 전달되는 프롬프트 주입 공격으로, 사용자의 지시를 통해 직접 전달되는 것이 아닙니다.
- **왜 중요한가:** 에이전트가 승인된 작업을 수행하는 동안 공격자가 제어하는 지시를 접하고, 해당 콘텐츠를 권위를 가진 지침으로 착각할 수 있습니다.
- **실무에서는:** 외부 콘텐츠를 신뢰할 수 없는 데이터로 라벨링하고, 지시와 분리하며, 도구 권한을 최소화하고, 중대한 조치에 대해 승인을 요구하며, 악성 검색 콘텐츠를 회귀 테스트에 포함합니다.
- **흔한 혼동:** '간접'은 전달 경로를 설명하는 것이지, 더 약한 공격을 의미하지 않습니다. 검색된 콘텐츠에 숨겨진 지시는 직접적인 사용자 프롬프트만큼 중대한 결과를 초래할 수 있습니다.
- **배울 곳:** [Indirect Prompt Injection](../phases/18-ethics-safety-alignment/15-indirect-prompt-injection/)
- **관련 용어:** Prompt Injection, Instruction Hierarchy, Trust Boundary, Data Exfiltration
- **출처:** [Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173)

### Inductive Bias - 귀납적 편향
- **분류:** 모델 및 추론
- **흔히 하는 말:** 학습 시스템에 내장된 가정.
- **실제 의미:** 일부 함수나 표현을 다른 것보다 선호하는 구조적 또는 통계적 가정입니다. 컨볼루션은 지역성과 공유 필터를 선호하며, 인과적 마스킹은 선행 위치로부터의 예측을 선호합니다.
- **흔한 혼동:** 트랜스포머는 토큰화, 위치 처리, 마스킹, 아키텍처, 데이터 및 목적 함수를 통해 귀납적 편향을 여전히 가지고 있습니다.
- **관련 용어:** CNN (Convolutional Neural Network), Transformer, Feature

### Inference - 추론
- **분류:** 모델 및 추론
- **흔히 하는 말:** 학습된 모델을 실행하는 것.
- **실제 의미:** 파라미터에 일반적인 학습 업데이트를 수행하지 않으면서, 학습된 모델을 실행하여 예측, 점수, 임베딩 또는 생성된 토큰을 산출하는 것.
- **흔한 혼동:** 모델 가중치는 변하지 않더라도, 추론 중에 애플리케이션이 캐시, 대화 상태 또는 외부 메모리를 업데이트할 수 있습니다.
- **관련 용어:** Autoregressive, Streaming, KV Cache

### Instruction Following - 지시문 따르기
- **분류:** 프롬팅 및 컨텍스트
- **실제 의미:** 자연어 지시와 제공된 컨텍스트를 명시된 작업 및 제약 조건을 충족하는 행동으로 매핑하는 모델의 능력.
- **왜 중요한가:** 언어 생성은 사용자가 요청한 작업, 형식, 경계 또는 우선순위를 따르지 않더라도 유창할 수 있습니다.
- **실무에서는:** 충돌하는 제약 조건, 형식 요구 사항, 관련 없는 컨텍스트 및 거절 사례를 사용하여 답변 품질과 별도로 지시 준수 여부를 평가합니다.
- **흔한 혼동:** 지시 따르기는 사실적 정확성, 정렬(alignment) 또는 지시처럼 보이는 모든 문자열에 대한 복종이 아닙니다.
- **관련 용어:** SFT (Supervised Fine-Tuning), Prompt Engineering, Instruction Hierarchy, Alignment
- **출처:** [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)

### Instruction Hierarchy - 지시문 계층
- **분류:** 프롬팅 및 컨텍스트
- **실제 의미:** 애플리케이션 정책, 사용자 및 신뢰할 수 없는 검색된 콘텐츠 등 서로 다른 권위를 가진 출처의 지시들 사이의 충돌을 해결하기 위한 규칙 집합.
- **왜 중요한가:** 에이전트 시스템은 신뢰할 수 있는 목표와 외부 텍스트를 혼합하므로, 낮은 권위의 콘텐츠가 높은 권위의 제약 조건과 충돌할 때 모델과 하네스가 정의된 응답을 해야 합니다.
- **실무에서는:** 신뢰할 수 없는 도구 출력은 데이터로 라벨링하고, 해당 콘텐츠 외부에 더 높은 우선순위의 제약 조건을 유지하며, 직접 및 간접 충돌 사례를 테스트합니다.
- **흔한 혼동:** 지시 계층은 행동을 개선할 수 있지만 보안 경계는 아닙니다. 최소 권한 원칙과 승인 제어는 여전히 결과를 제한합니다.
- **관련 용어:** System Prompt, Prompt Injection, Least Privilege, Tool Contract
- **출처:** [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208)

### Inter-Token Latency (ITL) - 토큰 간 지연 (ITL)
- **분류:** 인프라 및 서빙
- **실제 의미:** 하나의 요청에 대해 두 연속적인 출력 토큰 도착 이벤트 사이의 경과 시간으로, 첫 번째 이후의 출력 토큰에 대해 `t_i - t_(i-1)`로 계산됩니다.
- **왜 중요한가:** 개별 간격은 배치, 선점(preemption) 또는 혼합 워크로드 하에서 요청 단위 평균이 숨길 수 있는 디코딩 스톨(stall) 및 스트리밍 지터(jitter)를 드러냅니다.
- **실무에서는:** 각 요청의 첫 번째 토큰 이후 간격과 토큰 위치를 기록한 후, 요청 경계를 통합하지 않고 워크로드, 출력 길이 및 동시성별로 분포를 보고합니다.
- **흔한 혼동:** ITL은 연속적인 토큰 사이의 하나의 간격입니다. 출력 토큰당 시간(TPOT)은 이러한 간격에 대한 요청 단위 평균이며, 첫 번째 토큰까지의 시간(TTFT)은 스트리밍이 시작되기 전의 대기 시간을 포함합니다.
- **배울 곳:** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **관련 용어:** Time per Output Token (TPOT), Time to First Token (TTFT), Decode Phase, Tail Latency
- **출처:** [DistServe](https://arxiv.org/abs/2401.09670)

## J

### Jailbreak - 제일브레이크
- **분류:** 보안 및 거버넌스
- **실제 의미:** 모델이 훈련 또는 애플리케이션 제어에 의해 방지되도록 설계된 행동을 생성하도록 의도된 적대적 입력 또는 상호작용 전략입니다.
- **왜 중요한가:** 성공적인 Jailbreak는 명시된 정책과 실제 행동 사이의 격차를 드러내며, 모델이 도구 또는 보호된 데이터를 제어할 때 더 중대한 결과가 될 수 있습니다.
- **실무에서는:** 금지된 행동에서 테스트 계열을 파생하고, 형식 및 상호작용 길이를 다양화하며, 거절과 유해한 완성을 모두 측정하고, 확인된 실패를 버전 관리된 적대적 평가로 변환합니다.
- **흔한 혼동:** jailbreak는 모델 또는 시스템의 행동 제한을 겨냥합니다. prompt injection은 지시 따르기를 재지향하며, 종종 공격자의 목표를 향합니다. 하나의 상호작용에서 두 가지가 모두 포함될 수 있습니다.
- **배울 곳:** [Jailbreak Taxonomy](../phases/19-capstone-projects/82-jailbreak-taxonomy/)
- **관련 용어:** Prompt Injection, Red Teaming, Guardrails, Eval Set
- **출처:** [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)

### JAX
- **분류:** 수학 및 학습
- **흔히 하는 말:** 가속화된 기계 학습을 위한 NumPy 유사 시스템입니다.
- **실제 의미:** 자동 미분, 컴파일, 벡터화, 가속기 간 병렬 실행을 통해 수치 함수를 변환하는 Python 라이브러리입니다. 그 변환은 명시적 상태와 함수형 스타일 코드와 가장 잘 작동합니다.
- **흔한 혼동:** JAX는 모든 상태 유지 프로그래밍을 금지하지는 않지만, 변환된 함수 내부의 숨겨진 변경(mutation)은 부정확하거나 지원되지 않는 동작을 생성할 수 있습니다.
- **배울 곳:** [Introduction to JAX](../phases/03-deep-learning-core/12-intro-to-jax/)
- **출처:** [JAX documentation](https://docs.jax.dev/en/latest/)
- **관련 용어:** Autograd, Tensor, CUDA

## K

### Knowledge Distillation - 지식 증류
- **분류:** 수학 및 학습
- **실제 의미:** 더 유능한 teacher로부터 선택된 행동이나 출력 분포를 재현하도록 student 모델을 학습하는 것이며, 종종 일반적인 target label과 함께 수행됩니다.
- **왜 중요한가:** teacher를 직접 서빙하는 것이 비실용적일 때, 유용한 행동을 더 작거나 저렴한 모델로 전달할 수 있습니다.
- **실무에서는:** teacher 출력, temperature, student loss, 그리고 유지된 eval set을 정의한 후, student를 teacher와 label-only baseline 모두와 비교합니다.
- **흔한 혼동:** distillation은 training distribution에 대한 행동을 전달합니다. teacher의 모든 능력, 사실, 또는 안전 속성을 복사하지는 않습니다.
- **관련 용어:** Fine-tuning, Loss Function, Logits, Quantization
- **출처:** [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

### KV Cache - KV 캐시
- **분류:** 모델 및 추론
- **흔히 하는 말:** token 생성을 더 빠르게 만드는 캐시입니다.
- **실제 의미:** 자기회귀 생성의 이전 위치에서 저장된 키 및 값 텐서입니다. 이를 재사용하면 매 디코딩 단계에서 변경되지 않은 접두어에 대한 어텐션 투영을 다시 계산하지 않아도 됩니다.
- **왜 중요한가:** 반복적인 연산을 줄이지만, 시퀀스 길이, 레이어, 배치 및 모델 구성에 따라 증가하는 메모리를 소비합니다.
- **흔한 혼동:** KV 캐시는 시퀀스에 대한 런타임 어텐션 상태입니다. 접두어 캐싱은 요청 간에 적합한 KV 상태를 재사용하는 반면, 프롬프트 캐싱은 더 넓은 범위의 제공자 또는 애플리케이션 재사용 계약입니다.
- **배울 곳:** [KV Cache and Flash Attention](../phases/07-transformers-deep-dive/12-kv-cache-flash-attention/)
- **관련 용어:** Attention, Autoregressive, Prefix Caching, Prompt Cache

## L

### Late Fusion - 후기 융합
- **분류:** 멀티모달 시스템
- **실제 의미:** 모달리티를 별도의 인코더나 예측기를 통해 처리하고, 작업 출력 근처에서 고수준 표현, 점수 또는 결정을 결합하는 것입니다.
- **왜 중요한가:** 분리된 브랜치는 모달리티별 아키텍처를 사용할 수 있고 입력이 누락된 경우에도 허용되지만, 초기 융합에서 얻을 수 있는 세밀한 상호작용을 놓칠 수 있습니다.
- **실무에서는:** 각 브랜치를 보정하고, 누락된 모달리티가 병합에 미치는 영향을 정의하며, 점수 수준과 특징 수준의 결합을 비교하고, 각 브랜치만 단독으로 평가하여 Ablation을 수행합니다.
- **흔한 혼동:** Late Fusion은 결합의 위치를 설명합니다. 단순한 평균을 의미하지 않으며, 모달리티가 균등하게 기여한다는 것을 보장하지도 않습니다.
- **배울 곳:** [Cross-Attention Fusion](../phases/19-capstone-projects/61-cross-attention-fusion/)
- **관련 용어:** Early Fusion, Multimodal Fusion, Modality, Evaluation (Eval)
- **출처:** [Multimodal Deep Learning](https://ai.stanford.edu/~ang/papers/icml11-MultimodalDeepLearning.pdf); [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)

### Latent Space - 잠재 공간
- **분류:** 데이터 및 표현
- **흔히 하는 말:** 모델의 숨겨진 표현 공간입니다.
- **실제 의미:** 모델에 유용한 요소를 좌표로 인코딩하는 학습된 표현 공간입니다. 입력보다 차원이 낮을 수 있지만, 모든 Latent 표현에 압축이 요구되는 것은 아닙니다.
- **흔한 혼동:** 가까운 포인트들은 모델과 학습 목표가 학습한 내용에 따라 의미 있는 유사성을 가질 뿐입니다.
- **관련 용어:** Embedding, VAE (Variational Autoencoder), Feature

### Learning Rate - 학습률
- **분류:** 수학 및 학습
- **흔히 하는 말:** 각 최적화 단계의 크기입니다.
- **실제 의미:** 옵티마이저가 매개변수 업데이트의 크기를 제어하기 위해 사용하는 스케일 팩터입니다. 값이 너무 크면 학습이 불안정해질 수 있으며, 값이 너무 작으면 유용한 진전이 비현실적으로 느려질 수 있습니다.
- **흔한 혼동:** 효과적인 업데이트는 옵티마이저, 스케줄, 그래디언트 스케일, 배치 및 매개변수 이력에도 의존합니다.
- **관련 용어:** Optimizer, Gradient Descent, Batch Size

### Learning Rate Schedule - 학습률 스케줄
- **분류:** 수학 및 학습
- **실제 의미:** 학습이 진행됨에 따라 단계, 에포크, 지표 또는 사전 정의된 곡선에 따라 옵티마이저의 학습률을 변경하는 정책입니다.
- **왜 중요한가:** 다양한 학습 단계는 서로 다른 업데이트 스케일의 이점을 누릴 수 있으므로, 일정한 학습률은 초기에 불안정하거나 후반부에 낭비적일 수 있습니다.
- **실무에서는:** 옵티마이저 구성과 함께 스케줄을 버전 관리하고, 모든 단계에서 실제 학습률을 기록하며, 동일한 토큰 또는 업데이트 예산 하에서 스케줄들을 비교합니다.
- **흔한 혼동:** 스케줄러는 시간에 따라 학습률을 제어합니다. 옵티마이저 단계가 언제 발생하는지 결정하지 않으며, 수렴을 보장하지도 않습니다.
- **관련 용어:** Learning Rate, Warmup, Optimizer, Epoch
- **출처:** [SGDR](https://arxiv.org/abs/1608.03983); [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Least Privilege - 최소 권한
- **분류:** 평가 및 안전
- **실제 의미:** 모델, 에이전트, 도구 또는 사용자에게 현재 작업에 필요한 권한만, 그리고 그 권한이 필요한 기간 동안만 부여하는 것입니다.
- **왜 중요한가:** 모델은 실수를 하거나 악성 지시를 따를 수 있습니다. 좁은 권한은 단일 실패가 초래할 수 있는 피해를 줄입니다.
- **실무에서는:** 문서화 에이전트에게 소스 파일에 대한 읽기 권한과 하나의 브랜치에 대한 쓰기 권한을 부여하되, 프로덕션 자격 증명이나 병합 권한은 부여하지 않습니다.
- **흔한 혼동:** 인증은 신원을 증명합니다. 최소 권한 원칙은 그 신원이 할 수 있는 일을 제한합니다.
- **관련 용어:** Sandbox, Approval Gate, Prompt Injection, Tool Contract

### LLM (Large Language Model) - LLM (대규모 언어 모델)
- **분류:** 모델 및 추론
- **흔히 하는 말:** AI 애플리케이션의 두뇌입니다.
- **실제 의미:** 프롬프트나 적응을 통해 다양한 언어 작업을 수행할 수 있는 충분한 용량과 광범위한 학습을 갖춘 언어 모델입니다. 대부분의 최신 LLM은 트랜스포머 아키텍처와 시퀀스 예측 목적 함수를 사용하지만, 크기 임계값, 데이터 소스, 학습 레시피는 다양합니다.
- **흔한 혼동:** LLM은 모델 구성 요소입니다. 도구, 검색, 상태, 정책, 제품 로직은 주변 시스템에 존재합니다.
- **관련 용어:** Transformer, Autoregressive, Agent Harness

### LLM-as-a-Judge
- **분류:** 평가 및 안전
- **실제 의미:** 언어 모델을 사용하여 다른 시스템의 출력에 대해 루브릭(rubric)에 따라 점수를 매기거나, 비교하거나, 분류하거나, 비판하는 것입니다.
- **왜 중요한가:** 명확성이나 지시 준수와 같이 정확한 일치 테스트로 표현하기 어려운 품질의 평가를 확장할 수 있습니다.
- **실무에서는:** 평가자 모델에 작업, 후보 답변, 참조 증거, 구조화된 루브릭을 제공하고, 인간이 검토한 예시와 비교하여 점수를 보정합니다.
- **흔한 혼동:** 판정 모델은 정답(ground truth)이 아닙니다. 순서, 장황함, 스타일, 프롬프트 표현, 공유 모델의 실패에 의해 편향될 수 있습니다.
- **배울 곳:** [Eval-Driven Agent Development](../phases/14-agent-engineering/30-eval-driven-agent-development/)
- **관련 용어:** Evaluation (Eval), Eval Set, Verification Gate, Precision & Recall

### Load Shedding - 로드 셰딩
- **분류:** 신뢰성 및 운영
- **실제 의미:** 유용한 결과를 생성할 수 있는 용량을 초과하는 수요가 있을 때, 하나 이상의 과부하 경계에서 선택적으로 작업을 거부, 드롭(drop) 또는 취소하는 것입니다.
- **왜 중요한가:** 과부하 중에 모든 요청을 계속 수용하면 큐잉이 증가하여 거의 모든 요청이 마감 시간을 놓치게 되고, 회복이 더 어려워질 수 있습니다.
- **실무에서는:** 가장 먼저 파악된 경계 지점에서 부하를 제거하고, 가능한 경우 우선순위가 높고 이미 수락된 작업은 보존하며, 과부하된 범위를 식별하고, 조건이 일시적이며 요청이 재시도 예산 내에 남아 있는 경우에만 응답을 재시도 가능하게 표시합니다.
- **흔한 혼동:** 부하 제거(load shedding)는 이미 수락된 작업에만 국한되지 않습니다. 어드미션 컨트롤(admission control)은 특히 수락 전 게이트이며, 속도 제한(rate limiting)은 용량이 남아 있는 경우에도 사용 정책을 강제할 수 있습니다.
- **관련 용어:** Admission Control, Backpressure, Rate Limit, Graceful Degradation
- **출처:** [Google SRE: Handling Overload](https://sre.google/sre-book/handling-overload/)

### Logits - 로짓
- **분류:** 모델 및 추론
- **실제 의미:** 정규화 함수나 디코딩 규칙이 후보 결과를 선택으로 변환하기 전, 모델이 후보 결과에 대해 산출한 비정규화 수치 점수입니다.
- **왜 중요한가:** 온도(Temperature), softmax, top-k, top-p는 logits에 연산하거나 logits에서 파생되므로, logits는 모델 연산과 생성된 토큰을 연결합니다.
- **실무에서는:** API가 logits나 로그 확률을 노출하는 경우 이를 검사하고, 샘플링 전에 마스크를 적용하며, 원시 크기를 보정된 신뢰도로 해석하지 않도록 주의합니다.
- **흔한 혼동:** Logits는 확률이 아니며, 정의된 변환 없이는 관련 없는 위치, 모델, 작업 간에 비교할 수 없습니다.
- **관련 용어:** Softmax, Temperature, Token, Cross-Entropy
- **출처:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### LoRA (Low-Rank Adaptation) - LoRA (저랭크 적응)
- **분류:** 수학 및 학습
- **흔히 하는 말:** 파라미터 효율적인 미세 조정(fine-tuning).
- **실제 의미:** 기본 가중치를 동결하고 선택된 레이어에 대해 저랭크(low-rank) 업데이트 행렬을 학습하는 방법입니다. 학습 가능한 파라미터 수를 줄이며, 전체 파라미터 미세 조정(fine-tuning)에 비해 학습 메모리를 낮출 수 있습니다.
- **흔한 혼동:** 실제 메모리 및 속도 절감은 랭크(rank), 대상 모듈, 옵티마이저 상태, 활성화 메모리, 양자화, 구현 방식에 따라 달라집니다.
- **배울 곳:** [Fine-Tuning and LoRA](../phases/11-llm-engineering/08-fine-tuning-lora/)
- **출처:** [LoRA paper](https://arxiv.org/abs/2106.09685)
- **관련 용어:** Fine-tuning, QLoRA, Parameter

### Loss Function - 손실 함수
- **분류:** 수학 및 학습
- **흔히 하는 말:** 학습 오차를 측정하는 수치입니다.
- **실제 의미:** 예측값과 목표값을 값으로 매핑하며, 때로는 정규화 항을 포함하여 최적화가 줄이려고 하는 값을 생성하는 목적 함수입니다. 손실 함수는 학습이 직접적으로 보상하거나 벌하는 오차를 결정합니다.
- **흔한 혼동:** 낮은 학습 손실은 생산 환경의 작업에서 유용하고 안전하며 일반화 가능한 행동을 보장하지 않습니다.
- **관련 용어:** Cross-Entropy, Gradient, Evaluation (Eval)

### Lost in the Middle
- **분류:** 프롬프팅 및 컨텍스트
- **실제 의미:** 모델 성능이 증거의 위치에 따라 변하며, 관련 정보가 시작과 끝 사이에 위치할 때 성능이 저하될 수 있는 긴 컨텍스트 실패 패턴입니다.
- **왜 중요한가:** 컨텍스트 윈도우 내에 증거를 포함한다고 해서 모델이 모든 위치를 동일한 신뢰도로 사용한다는 보장은 없습니다.
- **실무에서는:** 여러 증거 위치를 테스트하고, 방해 요소를 줄이며, 결정에 중요한 제약 조건이 눈에 잘 띄는 위치에 배치하고, 출처와 대조하여 답변을 검증합니다.
- **흔한 혼동:** 이는 관찰된 행동 패턴이며, 모든 모델, 작업, 위치에서 동일하게 영향을 미치는 고정된 법칙이 아닙니다.
- **관련 용어:** Context Window, Context Engineering, Eval Set, Grounding
- **출처:** [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)

## M

### Maximum Marginal Relevance (MMR) - MMR (최대 주변 관련성)
- **분류:** 검색 및 생성
- **실제 의미:** 쿼리에 대한 관련성과 이미 선택된 항목에 대한 신규성을 균형 있게 고려하는 선택 규칙입니다.
- **왜 중요한가:** 중복된 청크를 줄여 제한된 컨텍스트 예산으로 더 많은 고유한 증거를 커버할 수 있습니다.
- **실무에서는:** 후보 풀을 검색하고, 문서화된 관련성-다양성 가중치를 사용하여 다음 항목을 선택하며, 답변 품질과 출처 커버리지 모두를 평가합니다.
- **흔한 혼동:** MMR은 기존 후보 집합을 다양화합니다. 누락된 증거를 검색하거나 선택된 지문이 정확하다는 것을 증명하지는 않습니다.
- **관련 용어:** Reranker, Chunking, RAG (Retrieval-Augmented Generation), Grounding
- **출처:** [The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries](https://www.cs.cmu.edu/~jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf)

### MCP (Model Context Protocol) - MCP (모델 컨텍스트 프로토콜)
- **분류:** 에이전트 및 도구
- **흔히 하는 말:** AI 애플리케이션이 도구 및 컨텍스트에 연결하는 표준 방식입니다.
- **실제 의미:** 호스트가 정의된 요청, 결과, 발견 및 전송 계약(contract)을 통해 도구, 리소스, 프롬프트 및 확장 기능을 노출하는 서버에 연결하기 위한 개방형 JSON-RPC 프로토콜입니다. 2026-07-28 개정판에서는 초기화 핸드셰이크나 프로토콜 세션에 의존하는 대신, 모든 요청이 프로토콜 버전과 클라이언트 기능을 포함합니다.
- **흔한 혼동:** MCP는 발견 및 교환을 표준화합니다. 어떤 도구가 호출하기에 안전한지 결정하거나, 권한을 부여하거나, 애플리케이션이 명시적 상태 핸들(handles)을 사용하는 것을 금지하지는 않습니다.
- **배울 곳:** [Model Context Protocol](../phases/11-llm-engineering/14-model-context-protocol/)
- **출처:** [MCP 2026-07-28 key changes](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- **관련 용어:** Stateless MCP, Multi Round-Trip Request (MRTR), Function Calling, Tool Contract, Least Privilege

### Membership Inference - 멤버십 추론
- **분류:** 보안 및 거버넌스
- **실제 의미:** 모델 출력이나 기타 접근 가능한 신호를 관찰하여 특정 레코드나 예제가 모델의 학습 데이터에 포함되었는지 추정하는 공격입니다.
- **왜 중요한가:** 모델이 레코드를 그대로 복제하지 않더라도, 구별되는 행동은 민감한 데이터셋의 참여에 대한 정보를 드러낼 수 있습니다.
- **실무에서는:** 실제 쿼리 인터페이스 하에서 대표 구성원과 비구성원을 테스트하고, 불필요한 신뢰도 신호를 제한하며, 데이터 노출을 줄이고, 유틸리티 요구 사항에 대해 프라이버시 방어 기능을 평가합니다.
- **흔한 혼동:** Membership Inference는 레코드가 학습에 참여했는지 묻습니다. Model Extraction은 모델 행동을 복제하려고 시도하며, 직접 기억(memorization) 테스트는 콘텐츠를 복원할 수 있는지 확인합니다.
- **배울 곳:** [Differential Privacy for LLMs](../phases/18-ethics-safety-alignment/22-differential-privacy-for-llms/)
- **관련 용어:** Data Leakage, Data Minimization, Eval Set, Data Classification
- **출처:** [Membership Inference Attacks Against Machine Learning Models](https://doi.org/10.1109/SP.2017.41)

### Mixed Precision - 혼합 정밀도
- **분류:** 수학 및 학습
- **흔히 하는 말:** 속도와 메모리 절약을 위해 낮은 정밀도의 연산을 사용하는 것입니다.
- **실제 의미:** 연산에 따라 서로 다른 데이터 타입을 사용하는 수치 전략으로, 많은 행렬 연산에는 낮은 정밀도를, 더 넓은 범위나 안정성이 필요한 값에는 높은 정밀도를 사용하는 경우가 많습니다.
- **흔한 혼동:** 속도, 메모리, 정확도에 미치는 영향은 하드웨어, 데이터 타입, 스케일링 방법, 커널, 모델에 따라 달라집니다. 고정된 배수 효과가 아닙니다.
- **관련 용어:** Tensor, CUDA, NaN (Not a Number), Quantization

### Modality - 모달리티
- **분류:** 멀티모달 시스템
- **실제 의미:** 텍스트, 이미지, 오디오, 비디오, 깊이, 센서 측정 등 자체적인 구조와 획득 과정을 가진 정보의 형태입니다.
- **왜 중요한가:** 모달리티마다 샘플링 속도, 잡음, 공간적 또는 시간적 구조, 결측 데이터 처리 방식이 다르므로, 하나의 전처리 가정이 모든 모달리티에 적용되는 경우는 드뭅니다.
- **실무에서는:** 정렬(alignment)이나 융합(fusion)을 설계하기 전에 각 모달리티의 출처, 단위, 해상도, 타이밍, 전처리, 결측값 처리 정책을 문서화해야 합니다.
- **흔한 혼동:** 모달리티는 단순히 파일 확장자나 기능(feature) 컬럼이 아닙니다. 하나의 모달리티를 여러 인코딩 방식으로 표현할 수 있으며, 하나의 샘플이 여러 모달리티를 포함할 수 있습니다.
- **배울 곳:** [MIO Any-to-Any Streaming](../phases/12-multimodal-ai/16-mio-any-to-any-streaming/)
- **관련 용어:** Multimodal Model, Token, Tensor, Embedding
- **출처:** [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665); [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)

### Modality Alignment - 모달리티 정렬
- **분류:** 멀티모달 시스템
- **실제 의미:** 서로 다른 모달리티의 표현(representation) 간에 의미적 또는 시간적으로 관련된 항목을 매칭할 수 있도록 대응 관계를 학습하거나 설정하는 것입니다.
- **왜 중요한가:** 시스템이 구조가 다른 입력 간에 동일한 이벤트, 객체, 개념을 연결할 수 없다면 융합(fusion) 및 크로스 모달리티 검색(cross-modal retrieval)이 실패합니다.
- **실무에서는:** 양수 쌍(positive pairs)과 음수 쌍(negative pairs)을 정의하고, 시간 또는 공간 메타데이터를 보존하며, 불일치(mismatched) 사례를 평가하고, 정렬을 다운스트림 작업 정확도와 별도로 측정해야 합니다.
- **흔한 혼동:** 정렬은 표현을 비교 가능하거나 대응되게 만듭니다. 표현이 동일해지거나 모달리티 특유의 정보가 사라질 필요는 없습니다.
- **배울 곳:** [Projection Layer Modality Alignment](../phases/19-capstone-projects/60-projection-layer-modality-align/)
- **관련 용어:** Shared Embedding Space, Contrastive Learning, Grounding, Multimodal Fusion
- **출처:** [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html)

### Model Card - 모델 카드
- **분류:** 평가 및 안전
- **실제 의미:** 모델의 의도된 사용 목적, 평가 조건, 성능 특성, 한계점, 그리고 관련 윤리 및 안전 고려사항을 설명하는 구조화된 보고서입니다.
- **왜 중요한가:** 보고된 증거가 사용자의 환경 및 배포 조건에 적용되는지 판단하기 위해 개발자에게 필요한 맥락을 제공합니다.
- **실무에서는:** 모델 버전, 학습 및 평가 범위, 하위 그룹별 결과, 알려진 실패 모드, 금지된 사용 사례, 그리고 각 주장의 작성 날짜를 문서화합니다.
- **흔한 혼동:** 모델 카드는 증거와 한계점을 전달하는 것이지, 인증서, 보증서, 시스템 위협 모델, 또는 배포별 평가의 대체물이 아닙니다.
- **관련 용어:** Eval Set, Dataset Split, Distribution Shift, Alignment
- **출처:** [Model Cards for Model Reporting](https://dl.acm.org/doi/10.1145/3287560.3287596)

### Model Router - 모델 라우터
- **분류:** AI 네이티브 개발
- **실제 의미:** 기능, 지연 시간, 비용, 컨텍스트 크기, 정책, 현재 가용성 등의 요구사항을 기반으로 요청에 대한 모델이나 제공자를 선택하는 구성 요소입니다.
- **왜 중요한가:** 다양한 작업과 실패 조건은 서로 다른 모델을 요구하며, 라우팅은 모든 요청을 가장 큰 모델로 보내지 않고도 결과의 품질을 향상시킬 수 있습니다.
- **실무에서는:** 저위험 추출 작업은 빠른 모델로 보내고, 복잡한 코드 리뷰는 더 강력한 모델로 보내며, 동일한 데이터 정책을 충족하는 제공자에서만 페일오버(failover)를 수행합니다.
- **흔한 혼동:** 라우팅은 정책 결정입니다. 랜덤 로드 밸런싱은 단순히 트래픽을 분배할 뿐입니다.
- **관련 용어:** Evaluation (Eval), Circuit Breaker, Rate Limit, Cost per Successful Task

### Model Serving - 모델 서빙
- **분류:** 인프라 및 서빙
- **실제 의미:** 버전화된 모델 아티팩트를 로드하고, 추론 요청을 받아 실행을 스케줄링하며, 자원을 관리하고, 운영 계약에 따라 결과를 반환하는 런타임 및 API 계층입니다.
- **왜 중요한가:** 능력이 뛰어난 모델이라도 큐잉, 배치 처리, 배치 위치 결정, 버전 관리, 취소, 응답 경계를 명시적으로 설계하지 않으면 신뢰할 수 없는 제품을 생성할 수 있습니다.
- **실무에서는:** 모델 및 토크나이저 버전을 고정하고, 요청 한도를 검증하며, 준비 상태 및 지연 신호를 노출하고, 동시성을 제어하며, 프로덕션 트래픽 라우팅 전에 롤백을 테스트합니다.
- **흔한 혼동:** 모델 서빙은 추론을 한 번 호출하는 것보다 범위가 넓지만, 검색, 도구, 정책, 사용자 상태를 포함할 수 있는 전체 애플리케이션보다는 범위가 좁습니다.
- **배울 곳:** [Self-Hosted Serving Selection](../phases/17-infrastructure-and-production/28-self-hosted-serving-selection/)
- **관련 용어:** Inference, Model Router, Autoscaling, Observability
- **출처:** [Clipper](https://arxiv.org/abs/1612.03079)

### MoE (Mixture of Experts) - MoE (혼합 전문가)
- **분류:** 모델 및 추론
- **흔히 하는 말:** 각 토큰마다 파라미터의 일부만 활성화하는 대규모 모델.
- **실제 의미:** 여러 전문가 서브 네트워크와 각 입력 단위(종종 각 토큰)에 대해 하위 집합을 선택하는 학습된 라우터를 포함하는 아키텍처입니다. 희소 활성화는 모든 순전파에서 모든 전문가를 사용하지 않으면서 총 파라미터 용량을 증가시킬 수 있습니다.
- **왜 중요한가:** 연산, 메모리, 통신, 라우팅 균형 및 품질은 특정 아키텍처와 서빙 시스템에 따라 달라집니다.
- **흔한 혼동:** 모델 개발자가 공개하지 않는 한, 제품 이름은 MoE 아키텍처를 증명하지 않습니다.
- **배울 곳:** [Mixture of Experts](../phases/07-transformers-deep-dive/11-mixture-of-experts/)
- **관련 용어:** Transformer, Model Router, Parameter

### Multimodal Fusion - 멀티모달 융합
- **분류:** 멀티모달 시스템
- **실제 의미:** 두 개 이상의 모달리티에서 증거나 학습된 표현을 결합하여 공동 표현, 예측 또는 생성된 출력을 산출하는 것.
- **왜 중요한가:** 모달리티는 상호 보완적인 증거를 제공할 수 있지만, 단순한 결합은 잡음, 타이밍 오류 또는 하나의 지배적인 스트림을 증폭시킬 수 있습니다.
- **실무에서는:** 단일 모달리티 기준선을 설정하고, 융합 지점과 마스크를 지정하며, 누락된 입력 및 모순되는 입력을 테스트하고, 평가된 각 슬라이스를 어떤 모달리티가 주도하는지 보고합니다.
- **흔한 혼동:** Fusion은 결합 연산입니다. Alignment는 대응 관계를 설정하며, 두 모달리티를 하나의 요청에 배치하는 것만으로는 성공적인 Fusion이나 Alignment가 발생했음을 증명하지 못합니다.
- **배울 곳:** [Cross-Attention Fusion](../phases/19-capstone-projects/61-cross-attention-fusion/)
- **관련 용어:** Early Fusion, Late Fusion, Cross-Attention, Modality Alignment
- **출처:** [Multimodal Deep Learning](https://ai.stanford.edu/~ang/papers/icml11-MultimodalDeepLearning.pdf); [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)

### Multimodal Model - 멀티모달 모델
- **분류:** 멀티모달 시스템
- **실제 의미:** 표현(representation), 정렬(alignment), 융합(fusion), 번역(translation), 또는 조정된 예측(coordinated prediction)을 통해 하나 이상의 모달리티로부터 학습하거나, 모달리티를 연관시키거나, 생성하는 모델입니다.
- **왜 중요한가:** 멀티모달 기능은 여러 입력 유형을 수용하는 것뿐만 아니라 모달리티가 어떻게 상호작용하는지에 따라 결정되며, 각 표현 경계에서 실패가 발생할 수 있습니다.
- **실무에서는:** 지원되는 입력 및 출력 조합을 문서화하고, 각 모달리티를 단독으로 그리고 함께 평가하며, 누락되거나 충돌하는 입력을 테스트하고, 모델과 함께 전처리 버전을 추적합니다.
- **흔한 혼동:** 이미지 모델과 텍스트 모델이 분리된 파이프라인은 시스템 수준에서 멀티모달이지만, 반드시 하나의 공동 학습된 멀티모달 모델은 아닙니다.
- **배울 곳:** [MIO Any-to-Any Streaming](../phases/12-multimodal-ai/16-mio-any-to-any-streaming/)
- **관련 용어:** Modality, Vision-Language Model (VLM), Multimodal Fusion, Transformer
- **출처:** [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198); [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)

### Multi Round-Trip Request (MRTR) - MRTR (다중 왕복 요청)
- **분류:** 에이전트 및 도구
- **다른 이름:** MRTR
- **실제 의미:** 연산이 `resultType: input_required`와 하나 이상의 `inputRequests`를 반환한 후, 클라이언트가 `inputResponses`와 정확히 반환된 `requestState`를 사용하여 원본 메서드를 재시도하는 MCP 요청 패턴입니다.
- **왜 중요한가:** 서버가 서버 주도 JSON-RPC 교환을 열거나 프로토콜 세션 상태를 저장하지 않고도 사용자, 모델, 또는 루트 입력을 요청할 수 있게 합니다.
- **실무에서는:** `tools/call`에서 입력 요청을 반환하고, 호스트에서 승인된 응답을 수집하며, 새로운 JSON-RPC id로 동일한 도구 호출을 재시도합니다.
- **흔한 혼동:** `requestState`은 신뢰할 수 없는 왕복(round-trip) 데이터입니다. 권한 부여나 비즈니스 결정에 사용하기 전에 무결성을 보호해야 하며, 서버 측 세션 식별자로 취급하지 마십시오.
- **배울 곳:** [MCP Roots and Elicitation](../phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/)
- **관련 용어:** Stateless MCP, MCP (Model Context Protocol), Human-in-the-Loop (HITL), Tool Contract
- **출처:** [MCP Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)

## N

### NaN (Not a Number) - NaN (숫자가 아님)
- **분류:** 수학 및 학습
- **흔히 하는 말:** 수치 연산이 실패했다는 신호입니다.
- **실제 의미:** 정의되지 않았거나 표현할 수 없는 수치 결과를 나타내는 부동 소수점 값입니다. 학습 중에는 NaN이 잘못된 연산, 오버플로, 불안정한 정규화, 과도한 업데이트, 또는 이전의 손상된 값에서 발생할 수 있습니다.
- **실무에서는:** 첫 번째 비유한(non-finite) 텐서를 찾아 입력을 검사하고, 해당 연산 근처에 어서션(assertion)이나 이상 탐지 로직을 추가하십시오.
- **관련 용어:** Mixed Precision, Learning Rate, Gradient

### Normalization - 정규화
- **분류:** 수학 및 학습
- **흔히 하는 말:** 데이터를 표준 범위로 스케일링하는 것입니다.
- **실제 의미:** 정의된 통계 값을 사용하여 입력, 활성화 값, 또는 특징을 재스케일링하거나 재중심화하는 변환의 집합입니다. Batch normalization과 layer normalization은 서로 다른 축을 사용하며, 학습과 추론 단계에서 거동 방식이 다릅니다.
- **흔한 혼동:** 정규화는 최적화 안정성을 개선할 수 있지만, 항상 더 큰 학습률을 허용하거나 모든 아키텍처를 개선하는 것은 아닙니다.
- **관련 용어:** Tensor, Activation Function, Mixed Precision

### Nucleus Sampling (Top-p) - 핵 샘플링 (Top-p)
- **분류:** 모델 및 추론
- **다른 이름:** Top-p sampling
- **실제 의미:** 누적 확률이 선택된 임계값에 도달하는 가장 작은 다음 토큰 후보 집합에서 샘플링하는 디코딩 방법입니다.
- **왜 중요한가:** 후보 집합의 크기가 분포에 따라 적응합니다. 불확실성이 넓을 때는 더 많은 옵션을 유지하고, 확률이 집중될 때는 더 적은 옵션을 유지합니다.
- **실무에서는:** 온도와 스톱 설정을 고정하여 임계값을 평가하고, 모든 결과와 함께 완전한 디코딩 구성을 기록합니다.
- **흔한 혼동:** Top-p는 확률 질량 임계값인 반면, top-k는 항상 고정된 최대 수의 후보를 유지합니다.
- **관련 용어:** Top-k Sampling, Temperature, Decoding Strategy, Softmax
- **출처:** [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

## O

### Observability - 관측 가능성
- **분류:** AI 네이티브 개발
- **실제 의미:** 기록된 입력, 출력, 상태 전환, 도구 호출, 시간, 비용, 오류 및 평가 신호로부터 AI 시스템의 동작을 이해하는 능력입니다.
- **왜 중요한가:** AI 실패는 종종 모델, 검색, 도구 및 오케스트레이션에 걸쳐 발생합니다. 실패 경계를 위치하기 위해 상관된 증거가 필요합니다.
- **실무에서는:** 검색, 모델 호출, 도구 실행, 승인 및 최종 점수에 걸쳐 추적 ID를 기록하면서 마스킹 및 접근 제어 적용을 수행합니다.
- **흔한 혼동:** 로깅은 이벤트를 수집합니다. 관측성은 운영 질문에 답할 수 있을 정도로 이벤트를 구조화하고 연결합니다.
- **배울 곳:** [Agent Observability Platforms](../phases/14-agent-engineering/24-agent-observability-platforms/)
- **관련 용어:** Trace, Evaluation (Eval), Agent State, Time to First Token (TTFT)

### Optimizer - 옵티마이저
- **분류:** 수학 및 학습
- **흔히 하는 말:** 가중치를 업데이트하는 알고리즘.
- **실제 의미:** 그래디언트를 파라미터 업데이트로 변환하는 알고리즘입니다. 순수 확률적 경사 하강법은 단순한 기준선이며, 모멘텀, Adam 및 기타 최적화기는 이력이나 적응형 스케일링을 사용하여 업데이트를 변경합니다. 각 선택은 메모리, 안정성 및 튜닝 동작이 다릅니다.
- **흔한 혼동:** 최적화기는 그래디언트를 소비합니다. 역전파는 그래디언트를 계산합니다.
- **관련 용어:** Adam (Optimizer), AdamW, Gradient, Learning Rate

### Orchestration - 오케스트레이션
- **분류:** 에이전트 및 도구
- **실제 의미:** 모델 및 도구 단계에 걸쳐 작업을 순서화, 분기, 위임, 재시도, 일시 정지, 재개 및 종료하는 제어 로직입니다.
- **왜 중요한가:** 신뢰할 수 있는 에이전트 동작은 모델 외부의 명시적인 워크플로 결정에 의존합니다. 특히 작업에 의존성이 있거나 중대한 부작용이 있는 경우입니다.
- **실무에서는:** 안정적인 단계를 워크플로 또는 상태 기계로 인코딩하고, 모델에 제한된 결정을 노출하며, 외부 쓰기 전에 상태 전이를 영속화합니다.
- **흔한 혼동:** 오케스트레이션은 자율성이나 멀티 에이전트 시스템과 동의어가 아닙니다. 단일 에이전트도 결정론적 워크플로를 통해 오케스트레이션될 수 있습니다.
- **관련 용어:** Agent Harness, Planning, Delegation, Durable Execution
- **출처:** [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

### Overfitting - 과적합
- **분류:** 수학 및 학습
- **흔히 하는 말:** 모델이 학습 데이터를 암기했습니다.
- **실제 의미:** 학습 데이터에서의 성능이 대표적인 unseen 데이터에서의 성능보다 현저히 좋은 일반화 격차입니다. 암기가 기여할 수 있지만, 운영상의 증상은 일반화 실패입니다.
- **실무에서는:** 학습 지표와 홀드아웃(held-out) 지표를 비교하고, 하위 그룹 실패를 검사하며, 데이터 품질, 정규화, 조기 종료, 모델 용량 등의 변경을 테스트합니다.
- **관련 용어:** Underfitting, Dropout, Weight Decay, Eval Set

## P

### Paged KV Cache - 페이지드 KV 캐시
- **분류:** 인프라 및 서빙
- **실제 의미:** 어텐션 상태를 고정 크기 블록에 저장하고, 시퀀스별 단일 연속 할당 대신 논리적 시퀀스 위치를 물리적 블록에 매핑하는 KV 캐시 메모리 관리자입니다.
- **왜 중요한가:** 가변 시퀀스 길이는 단편화와 예측 불가능한 성장을 유발하므로, 블록 기반 할당은 사용 가능한 메모리를 개선하고 유연한 공유를 가능하게 합니다.
- **실무에서는:** 워크로드 측정값에서 블록 크기를 선택하고, 할당 및 퇴거(eviction)를 추적하며, 요청 간 상태를 격리하고, 메모리 압력 하에서 취소 및 접두어(prefix) 공유를 테스트합니다.
- **흔한 혼동:** Paged KV cache는 런타임 어텐션 상태 메모리를 관리합니다. 모델 파라미터를 디스크로 이동하거나 모델의 학습된 컨텍스트 한계를 확장하지 않습니다.
- **배울 곳:** [vLLM Serving Internals](../phases/17-infrastructure-and-production/04-vllm-serving-internals/)
- **관련 용어:** KV Cache, Prefix Caching, Context Window, Model Serving
- **출처:** [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### Parameter - 매개변수
- **분류:** 모델 및 추론
- **흔히 하는 말:** 모델 크기를 설명하는 숫자입니다.
- **실제 의미:** 학습 중에 학습된 값으로, 일반적으로 가중치, 편향, 임베딩 요소, 정규화 파라미터입니다. 파라미터 개수는 모델 용량의 측정 지표 중 하나이지만, 품질, 메모리, 서빙 비용을 직접적으로 결정하지는 않습니다.
- **흔한 혼동:** 파라미터당 메모리 사용량은 수치 형식, 양자화 메타데이터, 샤딩, 옵티마이저 상태, 활성값, 런타임 오버헤드에 따라 달라집니다.
- **관련 용어:** Weight, MoE (Mixture of Experts), Quantization

### Pass@k
- **분류:** 평가 및 안전
- **실제 의미:** 작업 세트 전체에서, k개의 샘플링된 후보 중 적어도 하나가 정의된 정확성 테스트를 통과한 작업의 비율입니다.
- **왜 중요한가:** 자동 검증기가 각 후보를 확인할 수 있는 코드 생성과 같은 작업에서 여러 번 시도하는 것의 가치를 측정합니다.
- **실무에서는:** 고정된 구성 하에 후보를 독립적으로 생성하고, 각 후보에 동일한 격리된 테스트를 실행하며, 샘플링 및 추정기 세부 사항과 함께 k를 보고합니다.
- **흔한 혼동:** Pass@k는 단일 시도 정확도가 아니며, 더 높은 점수는 더 나은 첫 번째 답변이 아니라 더 큰 시도 예산을 반영할 수 있습니다.
- **관련 용어:** Coding Agent, Regression Test, Eval Set, Test Oracle
- **출처:** [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)

### Patch - 패치
- **분류:** AI 네이티브 개발
- **실제 의미:** 하나 이상의 파일에 대한 변경 사항의 검토 가능한 표현으로, 일반적으로 알려진 기본 리비전에 대한 추가 및 삭제 형태로 표현됩니다.
- **왜 중요한가:** 패치에는 전체 작업 디렉터리를 수용하지 않고도 사람과 에이전트가 검사, 테스트, 적용 또는 거부할 수 있는 좁은 산출물이 제공됩니다.
- **실무에서는:** 코딩 에이전트에게 통합 diff를 반환하도록 요청한 후, 허용된 파일만 건드리며 예상 커밋에 깨끗하게 적용되는지 확인합니다.
- **흔한 혼동:** 패치는 파일 변경을 포착하는 것이지, 이를 배포하기 위해 필요한 추론, 테스트 증거, 승인 등을 포착하는 것이 아닙니다.
- **배울 곳:** [Workbench for Real Repositories](../phases/14-agent-engineering/41-workbench-for-real-repos/)
- **관련 용어:** Coding Agent, Worktree, Scope Contract, Regression Test

### Patch Embedding - 패치 임베딩
- **분류:** 멀티모달 시스템
- **실제 의미:** 이미지 패치를 고정된 너비의 벡터로 변환하여 트랜스포머 입력 시퀀스의 한 요소로 사용하는 학습된 투영입니다.
- **왜 중요한가:** 공간 이미지 그리드와 시퀀스 모델 간의 인터페이스를 생성하며, 패치 크기는 토큰 수와 보존되는 지역적 세부 정보를 제어합니다.
- **실무에서는:** 패치 및 이미지 차원을 기록하고, 패딩이나 리사이징을 명시적으로 처리하며, 위치 정보를 추가하고, 해상도 변화가 정확도와 토큰 비용에 미치는 영향을 측정합니다.
- **흔한 혼동:** 패치 임베딩은 패치의 벡터 표현일 뿐, 의미론적 객체 감지기가 아니며 패치 경계가 시각적 개체와 일치한다는 보장이 아닙니다.
- **배울 곳:** [Vision Transformer Patch Tokens](../phases/12-multimodal-ai/01-vision-transformer-patch-tokens/)
- **관련 용어:** Vision Transformer (ViT), Image Token, Embedding, Token
- **출처:** [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

### Perplexity - 퍼플렉시티
- **분류:** 모델 및 추론
- **흔히 하는 말:** 언어 모델이 데이터셋에 대해 얼마나 놀라워하는지를 나타냅니다.
- **실제 의미:** 명시된 토큰화 및 로그 규칙에 따른 지수화된 평균 음의 로그 우도입니다. 값이 낮을수록 모델이 평가된 시퀀스에 더 높은 확률을 할당했다는 의미입니다.
- **흔한 혼동:** 퍼플렉시티는 서로 다른 토크나이저나 평가 설정 간에 비교할 수 없으며, 사실성이나 유용성을 직접 측정하지 않습니다.
- **관련 용어:** Cross-Entropy, Token, Evaluation (Eval)

### Pipeline Parallelism - 파이프라인 병렬화
- **분류:** 인프라 및 서빙
- **실제 의미:** 모델 레이어의 순차적 그룹을 여러 장치에 분할하고, 마이크로배치나 요청을 파이프라인처럼 각 단계를 통해 이동시키는 방식입니다.
- **왜 중요한가:** 모델이 단일 장치의 메모리를 초과할 수 있게 해 주지만, 스테이지 불균형, 파이프라인 버블, 활성화 전송, 장애 조정 등이 사용 가능한 성능에 영향을 미칩니다.
- **실무에서는:** 스테이지 비용을 균형 있게 조정하고, 마이크로배치 스케줄을 선택하며, 유휴 시간과 상호 연결 트래픽을 측정하고, 모델 및 체크포인트 파티션 메타데이터를 버전 관리합니다.
- **흔한 혼동:** 파이프라인 병렬 처리는 깊이에 따라 레이어를 분할합니다. 텐서 병렬 처리는 레이어 내에서 텐서 연산을 분할합니다.
- **배울 곳:** [Scaling and Distributed Training](../phases/10-llms-from-scratch/05-scaling-distributed/)
- **관련 용어:** Tensor Parallelism, Expert Parallelism, Batch Size, Model Serving
- **출처:** [GPipe](https://arxiv.org/abs/1811.06965)

### Planning - 계획
- **분류:** 에이전트 및 도구
- **실제 의미:** 현재 상태에서 목표 상태로 이동하기 위해 의도된 행동 및 의존성 시퀀스를 구성, 선택 또는 수정하는 것.
- **왜 중요한가:** 명시적인 계획은 에이전트가 비용이 많이 들거나 되돌릴 수 없는 행동을 확정하기 전에 가정과 순서를 가시화합니다.
- **실무에서는:** 의존성을 고려한 짧은 계획을 요청하고, 사용 가능한 도구 및 권한에 대해 이를 검증하며, 관찰 결과가 가정을 무효화할 경우 재계획합니다.
- **흔한 혼동:** 생성된 계획은 제안일 뿐, 단계가 실행 가능하거나 충분하거나 안전하다는 증거가 아닙니다.
- **관련 용어:** Agent State, ReAct, Orchestration, Verification Gate
- **출처:** [LLM+P](https://arxiv.org/abs/2304.11477)

### Postmortem - 사후 분석
- **분류:** 신뢰성 및 운영
- **실제 의미:** 영향, 감지, 대응, 기여 조건, 복구 및 소유된 후속 조치를 설명하는 영구적인 사고 기록으로, 분석의 대체 수단으로 책임을 배정하지 않습니다.
- **왜 중요한가:** 해결된 장애는 증거로서 가치를 가집니다. 시스템 조건과 결정을 포착하면 하나의 사건이 재발 및 대응 시간을 줄이는 개선 사항으로 전환됩니다.
- **실무에서는:** 추적 및 로그에서 타임라인을 구축하고, 트리거 이벤트와 기여 조건을 구분하며, 날짜가 지정된 조치를 할당하고, 각 조치가 관련 제어 사항을 변경했는지 검토합니다.
- **흔한 혼동:** 사후 분석(Postmortem)은 회의록을 만들거나 한 사람의 실수를 찾는 과정이 아닙니다. 검증 가능한 시스템 개선안을 도출해야 합니다.
- **관련 용어:** Incident Response, Regression Test, Audit Log, Observability
- **출처:** [Google SRE: Postmortem Culture](https://sre.google/sre-book/postmortem-culture/)

### Precision & Recall - 정밀도 및 재현율
- **분류:** 평가 및 안전
- **흔히 하는 말:** 분류 또는 검색 품질을 측정하는 두 가지 지표입니다.
- **실제 의미:** 정밀도는 플래그가 지정된 항목 중 얼마나 많은 항목이 정확했는지를 묻습니다. 재현율은 관련 항목 중 얼마나 많은 항목이 발견되었는지를 묻습니다. 고정된 스코어링 모델의 결정 임계값을 변경하면, 재현율을 높이면 정밀도가 낮아지는 경우가 많으며 그 반대도 마찬가지입니다. 더 나은 모델은 두 지표 모두를 개선할 수 있습니다. F1은 이 두 지표의 조화 평균입니다.
- **흔한 혼동:** 올바른 임계값과 지표는 각 오류의 비용과 대상 클래스의 빈도에 따라 달라집니다.
- **관련 용어:** Eval Set, Semantic Search, Guardrails

### Prefill - 프리필
- **분류:** 인프라 및 서빙
- **다른 이름:** Prefill Phase
- **실제 의미:** 제공된 모든 입력 토큰을 처리하여 표현 및 후속 자기회귀 생성에 필요한 어텐션 상태를 생성하는 초기 추론 단계입니다.
- **왜 중요한가:** 프롬프트 형태, 큐잉, 캐시 재사용은 프리필 비용에 영향을 미치며, 프리필은 디코딩과 계산 자원 경쟁 방식이 다르므로 시작 지연(latency) 및 서빙 스케줄에 큰 영향을 미칩니다.
- **실무에서는:** 프롬프트 토큰 수와 프리필 지연 시간을 기록하고, 큐잉 시간과 실행 시간을 분리하며, 캐시된 접두사와 캐시되지 않은 접두사를 비교하고, 활성 디코딩 트래픽과 함께 긴 프롬프트를 테스트합니다.
- **흔한 혼동:** 프리필은 런타임 프롬프트 처리 단계이며, 첫 번째 생성된 토큰 자체는 아닙니다. 첫 번째 토큰은 프리필 및 모든 큐잉이 완료된 후에만 나타납니다.
- **배울 곳:** [Disaggregated Prefill and Decode](../phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/)
- **관련 용어:** Decode Phase, KV Cache, Time to First Token (TTFT), Chunked Prefill
- **출처:** [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf); [DistServe](https://arxiv.org/abs/2401.09670)

### Prefix Caching - 접두어 캐싱
- **분류:** 인프라 및 서빙
- **실제 의미:** 요청 간에 동일한 자격 요건을 충족하는 토큰 접두사에 대해 생성된 KV-cache 블록을 재사용하여, 서빙 런타임이 반복적인 접두사 연산을 생략할 수 있도록 하는 것입니다.
- **왜 중요한가:** 공유된 시스템 지침, 템플릿, 문서가 상당한 사전 처리(prefill) 연산을 소비할 수 있지만, 재사용은 토큰 시퀀스와 캐시 자격 요건이 일치할 때만 도움이 됩니다.
- **실무에서는:** 요청별 콘텐츠 앞에 안정적인 토큰을 배치하고, 캐시 식별자에 모델 및 토크나이저 버전을 포함하며, 테넌트 민감 상태는 격리하고, 적중률(hit rate)을 모니터링하며, eviction(제거)은 정상적인 동작으로 취급합니다.
- **흔한 혼동:** Prefix caching은 정확한 토큰 접두사에 대해 런타임 attention state를 재사용합니다. Prompt caching은 더 넓은 범위의 제공자 또는 애플리케이션 계약이며, semantic caching은 유사한 요청에 대해 이전 결과를 재사용합니다.
- **배울 곳:** [Inference Optimization](../phases/10-llms-from-scratch/12-inference-optimization/)
- **관련 용어:** Prompt Cache, Semantic Cache, KV Cache, Paged KV Cache
- **출처:** [SGLang](https://arxiv.org/abs/2312.07104)

### Progressive Disclosure - 점진적 공개
- **분류:** AI-native development
- **실제 의미:** 사람이나 모델에 먼저 최소한의 유용한 컨텍스트를 제공하고, 작업이나 증거가 필요로 할 때 더 깊은 세부 정보를 공개하는 것입니다.
- **왜 중요한가:** 컨텍스트 노이즈와 비용을 제한하면서도, 권위 있는 세부 정보를 필요할 때 즉시 이용할 수 있도록 유지합니다.
- **실무에서는:** 코딩 에이전트에게 먼저 저장소 규칙과 맵을 제공하며, 관련 모듈을 식별한 후에만 전체 구현 파일을 로드합니다.
- **흔한 혼동:** Progressive disclosure는 세부 정보에 대한 단계적 접근이지, 의사결정에 필요한 정보를 의도적으로 숨기는 것이 아닙니다.
- **배울 곳:** [Workbench for Real Repositories](../phases/14-agent-engineering/41-workbench-for-real-repos/)
- **관련 용어:** Context Engineering, Repository Map, Token Budget, Handoff

### Prompt Cache - 프롬프트 캐시
- **분류:** Prompting & context
- **실제 의미:** 동일한 또는 자격 요건을 충족하는 프롬프트 접두사에 대해 제공자 측 또는 애플리케이션 측 연산을 재사용하여, 반복적인 추론이 일부 전처리 연산을 피하도록 하는 것입니다.
- **왜 중요한가:** 제공자의 캐시 계약이 충족될 경우, 안정적인 지침과 대규모 공유 문서가 반복적인 호출에서 더 저렴하거나 빨라질 수 있습니다.
- **실무에서는:** 안정적인 정책 텍스트를 요청별 콘텐츠 앞에 배치하고, 캐시 적중 메타데이터를 모니터링하며, 자격 요건과 수명이 제공업체에 따라 다르므로 캐시 미스를 정상적인 현상으로 간주합니다.
- **흔한 혼동:** 프롬프트 캐시는 제공업체나 애플리케이션의 재사용 계약이며, 내부적으로 접두어 캐싱을 사용할 수 있습니다. 접두어 캐싱은 자격 요건을 충족하는 정확한 토큰의 KV 상태를 재사용하는 반면, 시맨틱 캐싱은 충분히 유사한 요청에 대해 이전 결과를 재사용합니다.
- **배울 곳:** [Prompt Caching](../phases/11-llm-engineering/15-prompt-caching/)
- **관련 용어:** Semantic Cache, Prefix Caching, KV Cache, Time to First Token (TTFT)

### Prompt Engineering - 프롬프트 엔지니어링
- **분류:** 프롬팅 및 컨텍스트
- **흔히 하는 말:** 모델이 작업을 따르도록 지시문을 작성하는 것.
- **실제 의미:** 정의된 작업에 대한 모델의 동작을 개선하기 위해 모델이 보는 지시문, 예시, 제약 조건 및 출력 요구 사항을 설계하는 것.
- **흔한 혼동:** 프롬프트 작성은 증거 부족, 안전하지 않은 권한, 부실한 도구 계약, 평가 부재를 보완할 수 없습니다.
- **배울 곳:** [Prompt Engineering](../phases/11-llm-engineering/01-prompt-engineering/)
- **관련 용어:** Context Engineering, Few-Shot, System Prompt, Structured Output

### Prompt Injection - 프롬프트 인젝션
- **분류:** 평가 및 안전
- **흔히 하는 말:** 모델을 방향 전환시키는 적대적 지시문.
- **실제 의미:** 신뢰할 수 없는 콘텐츠가 모델에 영향을 미쳐 의도된 지시문을 무시하고, 데이터를 노출하며, 도구를 오용하거나, 사용자의 목표 밖의 행동을 취하게 하는 공격 또는 실패 모드입니다. 콘텐츠는 사용자로부터 직접 전달되거나, 검색된 페이지, 파일, 메시지, 도구 출력 등을 통해 간접적으로 전달될 수 있습니다.
- **왜 중요한가:** 모델은 지시문과 데이터를 동일한 언어 채널을 통해 처리하므로, 입력 필터링만으로는 모든 악성 지시문을 합법적인 콘텐츠와 신뢰할 수 있게 분리할 수 없습니다.
- **실무에서는:** 외부 콘텐츠를 신뢰할 수 없는 것으로 취급하고, 권한을 부여하는 지시문과 분리하며, 도구 권한을 최소화하고, 중요한 쓰기 작업에 대해 승인을 요구하며, 출력 및 행동을 검증합니다.
- **흔한 혼동:** 프롬프트 인젝션은 기술적으로 SQL 인젝션과 동일한 메커니즘이 아니며, 더 강력한 시스템 프롬프트는 완전한 방어책이 아닙니다.
- **배울 곳:** [Prompt Injection Defense](../phases/14-agent-engineering/27-prompt-injection-defense/)
- **출처:** [OWASP prompt injection guidance](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- **관련 용어:** Least Privilege, Sandbox, Approval Gate, Tool Contract

### Prompt Sensitivity - 프롬프트 민감도
- **분류:** 프롬팅 및 컨텍스트
- **실제 의미:** 의도된 작업을 유지하는 프롬프트의 표현, 순서, 형식, 예시 변경으로 인해 발생하는 모델 출력 또는 측정된 성능의 변동입니다.
- **왜 중요한가:** 하나의 편리한 표현으로 성공한 시스템은 실제 사용자에게는 신뢰할 수 없거나 평가 시 오해를 불러일으킬 수 있습니다.
- **실무에서는:** 의미적으로 동등한 프롬프트 변형을 생성하고, 케이스별로 변동을 측정하며, 하나의 프롬프트를 하나의 평가 세트에 대해 최적화하는 대신 변형을 회귀 테스트에 포함하여 유지합니다.
- **흔한 혼동:** 민감도는 항상 프롬프트 결함은 아닙니다. 모호성, 모델의 약한 강건성, 불안정한 디코딩, 부적절한 채점 규칙을 드러낼 수 있습니다.
- **관련 용어:** Prompt Engineering, Eval Set, Regression Test, Few-Shot
- **출처:** [ProSA](https://aclanthology.org/2024.findings-emnlp.108/)

### Provenance Attestation - 출처 증명
- **분류:** 보안 및 거버넌스
- **실제 의미:** 산출물이 어떻게, 어디서, 언제, 어떤 입력으로부터 생성되었는지에 대한 주장을 산출물과 연결하는 인증된 기계 가독 메타데이터입니다.
- **왜 중요한가:** 서명되지 않은 빌드 노트를 신뢰하는 대신, 자동화된 정책 및 리뷰어가 공급망 주장을 검증할 수 있게 합니다.
- **실무에서는:** 빌드 시스템에서 증명서를 생성하고, 산출물 다이제스트와 연결하며, 통제된 신원으로 서명하고, 릴리스 전에 검증합니다.
- **흔한 혼동:** 서명은 증명자 식별 및 무결성 보호를 수행합니다. 증명서 내부의 모든 주장이 참임을 증명하지는 않습니다.
- **관련 용어:** Data Provenance, Reproducible Build, Audit Log, Verification Gate
- **출처:** [SLSA Software Attestations](https://slsa.dev/spec/v1.2/attestation-model)

### Purpose Limitation - 목적 제한
- **분류:** 보안 및 거버넌스
- **실제 의미:** 개인 데이터를 지정된 명시적 목적에 대해서만 수집 및 사용하며, 새로운 사용은 적절한 호환성이나 승인된 근거가 있는 경우에만 허용됩니다.
- **왜 중요한가:** 한 워크플로우에서는 허용되었던 데이터가 모델 학습, 평가, 개인화, 또는 무관한 분석을 위해 조용히 재사용될 경우, 프라이버시 및 거버넌스 위험을 초래할 수 있습니다.
- **실무에서는:** 각 데이터셋에 목적을 기록하고, 접근 전에 새로운 파이프라인이 그 목적과 일치하는지 확인하며, 호환되지 않는 용도를 분리하고, 목적 변경 시 문서화된 결정을 요구해야 합니다.
- **흔한 혼동:** 목적 제한은 데이터를 '왜' 사용하는지를 규정합니다. 데이터 최소화는 그 목적에 실제로 '얼마나 많은' 데이터가 필요한지를 규정합니다.
- **관련 용어:** Data Minimization, Data Classification, AI Risk Assessment, Audit Log
- **출처:** [General Data Protection Regulation, Article 5(1)(b)](https://eur-lex.europa.eu/eli/reg/2016/679/oj)

## Q

### QLoRA
- **분류:** 수학 및 학습
- **흔히 하는 말:** 양자화된 기본 모델과 함께 사용하는 LoRA.
- **실제 의미:** 사전 학습된 기본 모델을 저비트 양자화 표현으로 동결한 상태로 유지하면서, 필요한 경우 더 높은 정밀도의 연산으로 LoRA 어댑터를 학습하는 파라미터 효율적인 미세 조정 방법입니다.
- **왜 중요한가:** 대규모 모델을 적응하는 데 필요한 메모리를 줄일 수 있지만, 절약 효과와 품질은 모델, 랭크(rank), 옵티마이저, 시퀀스 길이, 하드웨어 및 구현 방식에 따라 달라집니다.
- **흔한 혼동:** QLoRA는 특정 메모리 사용량이나 완전 미세 조정(full fine-tuning) 대비 고정된 품질 격차를 보장하지 않습니다.
- **배울 곳:** [Fine-Tuning and LoRA](../phases/11-llm-engineering/08-fine-tuning-lora/)
- **출처:** [QLoRA paper](https://arxiv.org/abs/2305.14314)
- **관련 용어:** LoRA (Low-Rank Adaptation), Quantization, Fine-tuning

### Quantization - 양자화
- **분류:** 모델 및 추론
- **흔히 하는 말:** 모델 값을 더 적은 비트로 저장하거나 계산하는 것.
- **실제 의미:** 메모리, 대역폭, 또는 연산 비용을 줄이기 위해 가중치, 활성화 값, 또는 캐시를 저정밀도 형식으로 표현하는 것입니다. 방법은 캘리브레이션(calibration), 세분화(granularity), 데이터 타입, 그리고 변환이 학습 전, 중, 후 중 언제 일어나는지에 따라 다릅니다.
- **흔한 혼동:** 하나의 명목 비트 폭에서 다른 명목 비트 폭으로 이동한다고 해서 동일한 엔드투엔드(end-to-end) 메모리 또는 속도 비율이 보장되지 않습니다. 메타데이터, 커널, 캐시, 하드웨어 지원도 중요한 요소이기 때문입니다.
- **관련 용어:** QLoRA, Mixed Precision, Parameter

## R

### RAG (Retrieval-Augmented Generation) - RAG (검색 증강 생성)
- **분류:** 검색 및 생성
- **흔히 하는 말:** 검색된 지식으로 답변하는 모델.
- **실제 의미:** 요청과 관련된 증거를 검색하고, 선택된 콘텐츠를 생성 모델에 제공하여 답변이나 행동을 수행하기 전에 사용하는 시스템 패턴입니다. 검색은 어휘적, 벡터, 구조적 또는 하이브리드 방법을 사용할 수 있습니다.
- **왜 중요한가:** RAG는 모델 가중치에 인코딩하지 않고도 현재 또는 비공개 증거를 이용할 수 있게 해 주지만, 검색과 접지(grounding)는 별도로 평가해야 합니다.
- **이름의 유래:** 검색(Retrieval)은 증거를 찾고, 증강(augmentation)은 선택된 증거를 컨텍스트에 추가하며, 생성(generation)은 응답을 만들어냅니다.
- **배울 곳:** [Retrieval-Augmented Generation](../phases/11-llm-engineering/06-rag/)
- **출처:** [Retrieval-Augmented Generation paper](https://arxiv.org/abs/2005.11401)
- **관련 용어:** Grounding, Hybrid Retrieval, Reranker, Hallucination

### Rate Limit - 속도 제한
- **분류:** AI 네이티브 개발
- **실제 의미:** 정의된 시간 또는 용량 윈도우 내에서 요청, 토큰, 동시 작업 또는 기타 자원을 제한하는 정책입니다.
- **왜 중요한가:** 제공자와 자체 시스템을 과부하, 통제되지 않은 지출, 불공정한 자원 사용으로부터 보호합니다.
- **실무에서는:** 테넌트별 토큰 및 동시성 제한을 적용하고, 제공자의 재시도 메타데이터를 읽으며, 초과 작업을 예측 가능하게 큐에 넣거나 거부합니다.
- **흔한 혼동:** Rate Limit는 허용된 사용을 제어합니다. Backpressure는 시스템 전체로 하류 용량 제약 조건을 전파합니다.
- **관련 용어:** Backpressure, Retry with Backoff, Circuit Breaker

### ReAct
- **분류:** 에이전트 및 도구
- **실제 의미:** 작업 추론, 구체적인 행동, 그리고 환경이 반환한 관찰을 교대로 수행하여 다음 단계를 결정하는 에이전트 패턴입니다.
- **왜 중요한가:** 환경 피드백은 가정을 수정하고 이후 결정을 접지(ground)하여, 모델이 내부 생성만으로 전체 작업을 완료하도록 강제하지 않도록 합니다.
- **실무에서는:** 소수의 타입이 지정된 도구를 노출하고, 간결한 관측 결과를 반환하며, 루프에 상한을 설정하고, 최종 산출물을 검증하며, 비공개 추론 흔적을 저장하지 마십시오.
- **흔한 혼동:** ReAct는 프롬팅 및 제어 패턴이며, 자율성, 정확성, 안전한 도구 사용을 보장하는 것이 아닙니다.
- **관련 용어:** Agent, Function Calling, Planning, Grounding
- **출처:** [ReAct](https://arxiv.org/abs/2210.03629)

### Readiness Probe - 준비 상태 프로브
- **분류:** 신뢰성 및 운영
- **실제 의미:** 서비스 인스턴스가 현재 요청을 수용할 수 있는지 트래픽 라우팅 계층에 알리는 진단입니다.
- **왜 중요한가:** 프로세스가 살아 있더라도 모델이 언로드되었거나, 의존성이 사용 불가능하거나, 워밍업이 완료되지 않은 상태일 수 있으므로, 너무 일찍 트래픽을 보내면 피할 수 있는 장애가 발생합니다.
- **실무에서는:** 서비스를 위해 필요한 최소한의 의존성을 확인하고, 시작 및 드레이닝(draining) 중에 준비 상태를 실패로 표시하며, 프로브가 저렴하도록 유지하고, 준비 상태가 거짓이라는 이유만으로 프로세스를 재시작하지 마십시오.
- **흔한 혼동:** Readiness는 트래픽 자격을 제어합니다. Liveness는 프로세스를 재시작해야 하는지 결정하며, 둘 다 모든 모델 응답이 정확하다는 것을 증명하지는 않습니다.
- **배울 곳:** [Production LLM Application](../phases/11-llm-engineering/13-production-app/)
- **관련 용어:** Autoscaling, Model Serving, Availability, Graceful Degradation
- **출처:** [Kubernetes Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)

### Recall@K
- **분류:** 검색 및 생성
- **실제 의미:** 하나의 쿼리에 대해 Recall@K는 `|relevant items intersecting the top k| / |relevant items|`입니다. 데이터셋 점수는 명시된 규칙에 따라 쿼리별 값들을 집계합니다.
- **왜 중요한가:** 검색 단계가 다운스트림 생성이나 리랭킹(reranking)에 충분한 관련 후보를 공급하는지 알려줍니다.
- **실무에서는:** 관련성 판단, k, 집계 방법, 그리고 판단된 관련 항목이 없는 쿼리에 대한 정책을 정의한 후, 리콜된 증거가 0인 쿼리를 검사하십시오.
- **흔한 혼동:** 높은 Recall@K는 상위 결과가 좋다는 뜻이 아니며, 순위가 잘 정렬되어 있다는 뜻도 아니고, 최종 답변이 근거에 기반한다는 뜻도 아닙니다. 관련 항목이 없는 쿼리는 분모가 0이 되므로 명시적인 제외 정책이나 할당된 값 정책이 필요합니다.
- **관련 용어:** Precision & Recall, Eval Set, Reranker, Approximate Nearest Neighbor (ANN)
- **출처:** [BEIR](https://openreview.net/forum?id=wCu6T5xFjeJ)

### Reciprocal Rank Fusion (RRF) - 상호 랭킹 융합 (RRF)
- **분류:** 검색 및 생성
- **실제 의미:** 각 목록에서 항목의 순위에 따라 기여도가 감소하는 방식으로 여러 결과 목록을 합산하는 순위 융합 방법입니다.
- **왜 중요한가:** 원시 점수가 동일한 스케일을 공유한다고 가정하지 않고 어휘 기반, 밀집(dense), 다중 쿼리 순위 목록을 병합할 수 있습니다.
- **실무에서는:** 독립적인 후보 목록을 검색하고, 안정적인 문서 식별자로 중복을 제거하며, 버전이 지정된 융합 상수를 적용하고, 각 개별 검색기와 비교하여 평가합니다.
- **흔한 혼동:** RRF는 임베딩이나 관련성 점수가 아닌 순위를 결합하며, 모든 입력 목록에 없는 항목을 복원할 수 없습니다.
- **관련 용어:** Hybrid Retrieval, BM25, Dense Retrieval, Reranker
- **출처:** [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114)

### Red Teaming - 레드 티밍
- **분류:** 보안 및 거버넌스
- **실제 의미:** 문서화된 목표, 위협 가정, 사례 및 증거를 사용하여 실패를 찾는 구조화된 적대적 테스트 프로세스입니다.
- **왜 중요한가:** 일반적인 품질 테스트는 조작, 오용, 목표 충돌, 통제 우회를 위한 단호한 시도 하에서 시스템이 어떻게 동작하는지 거의 탐구하지 않습니다.
- **실무에서는:** 위협 모델에서 공격을 도출하고, 격리된 환경에서 실행하며, 재현 가능한 사례를 기록하고, 계층별로 시정 조치하며, 확인된 실패를 회귀 테스트로 전환합니다.
- **흔한 혼동:** jailbreak 프롬프트 목록은 완전한 red-team 프로그램이 아니며, red teaming은 알려지지 않은 실패의 부재를 증명할 수 없습니다.
- **관련 용어:** Threat Model, Guardrails, Prompt Injection, Eval Set
- **출처:** [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)

### Regression Test - 회귀 테스트
- **분류:** AI 네이티브 개발
- **실제 의미:** 코드, 프롬프트, 모델, 검색 또는 도구 변경 후 특히 잘 알려진 동작을 보호하는 반복 가능한 점검입니다.
- **왜 중요한가:** AI 시스템의 변경은 평균 품질을 개선하는 동시에, 이전에 수정된 실패를 조용히 다시 도입할 수 있습니다.
- **실무에서는:** 수정된 프롬프트 주입 사건을, 다음 배포 전에 반드시 통과해야 하는 영구적인 평가 케이스로 전환합니다.
- **흔한 혼동:** 회귀 테스트는 특정 기대 동작을 보호합니다. 광범위한 벤치마크는 더 넓은 작업 분포에 걸친 성능을 추정합니다.
- **배울 곳:** [Eval-Driven Agent Development](../phases/14-agent-engineering/30-eval-driven-agent-development/)
- **관련 용어:** Eval Set, Verification Gate, Patch, Evaluation (Eval)

### ReLU
- **분류:** 수학 및 학습
- **흔히 하는 말:** 단순한 활성화 함수입니다.
- **실제 의미:** Rectified Linear Unit로, `f(x) = max(0, x)`로 정의됩니다. 비용이 저렴하고 비포화 양의 분기를 가지지만, 음의 입력에 대한 0의 기울기는 비활성화된 유닛을 생성할 수 있습니다.
- **관련 용어:** Activation Function, Gradient, CNN (Convolutional Neural Network)

### Repository Instructions - 저장소 지침
- **분류:** AI 네이티브 개발
- **실제 의미:** 저장소가 어떻게 조직되어 있는지, 어떤 명령과 규약이 적용되는지, 어떤 경계를 준수해야 하는지, 그리고 작업을 어떻게 검증하는지 코딩 에이전트에게 알려주는 버전 관리된 지침입니다.
- **왜 중요한가:** 반복적인 암묵적 지식을 코드와 함께 이동하며 하위 프로젝트에 따라 달라질 수 있는 지역적 컨텍스트로 전환합니다.
- **실무에서는:** 저장소 루트에 `AGENTS.md`를 유지하고, 하위 디렉터리에 더 좁은 파일을 추가하며, 정확한 빌드, 테스트, 생성 파일, 보안 및 기여 규칙을 포함합니다.
- **흔한 혼동:** 저장소 지침은 소스 코드와 인간 문서의 보완재입니다. 사용자의 현재 요청을 덮어쓰지 않으며, 에이전트가 이를 올바르게 따르도록 보장하지도 않습니다.
- **관련 용어:** Repository Map, Scope Contract, Coding Agent, Progressive Disclosure
- **출처:** [AGENTS.md specification](https://agents.md/)

### Repository Map - 저장소 맵
- **분류:** AI 네이티브 개발
- **실제 의미:** 저장소의 중요한 디렉터리, 소유권 경계, 진입점, 빌드 명령, 테스트, 생성 파일 및 지역 지침에 대한 간결하고 유지 관리되는 설명입니다.
- **왜 중요한가:** 코딩 에이전트가 큰 파일을 로드하거나 잘못된 하위 시스템을 편집하기 전에 올바른 증거를 찾는 데 도움을 줍니다.
- **실무에서는:** 트리 및 매니페스트에서 인덱스를 생성한 후, 모듈 경계와 검증 명령에 대한 권위 있는 노트로 이를 보강합니다.
- **흔한 혼동:** 원시 파일 트리는 이름만 보여줍니다. 저장소 맵은 어떤 경로가 중요한지, 그리고 그 경로가 작업과 어떻게 관련되는지 설명합니다.
- **배울 곳:** [Repository Memory and State](../phases/14-agent-engineering/34-repo-memory-and-state/)
- **관련 용어:** Coding Agent, Progressive Disclosure, Scope Contract, Context Engineering

### Reproducible Build - 재현 가능한 빌드
- **분류:** AI-native development
- **실제 의미:** 선언된 소스, 환경 및 지침을 독립적으로 재실행하여 지정된 산출물을 비트 단위로 동일하게 생성할 수 있는 빌드입니다.
- **왜 중요한가:** 산출물이 원래 생성한 기계나 에이전트를 넘어서 검증 가능하게 만들며, 숨겨진 빌드 입력을 드러냅니다.
- **실무에서는:** 툴체인과 의존성을 고정하고, 타임스탬프 및 불안정한 순서를 제거하며, 환경을 캡처한 후, 독립적으로 재빌드된 산출물 다이제스트를 비교합니다.
- **흔한 혼동:** 두 번 성공하는 빌드는 반복 가능한 증거이지만, 재현성(reproducibility)은 선언된 독립적인 조건과 동일한 출력을 요구합니다.
- **관련 용어:** Repository Instructions, Verification Gate, Provenance Attestation, Software Bill of Materials (SBOM)
- **출처:** [Reproducible Builds definition](https://reproducible-builds.org/docs/definition/)

### Reranker - 리랭커
- **분류:** Retrieval & generation
- **실제 의미:** 쿼리와 각 후보 간의 더 풍부한 비교를 사용하여 작은 후보 집합을 재정렬하는 2단계 모델 또는 스코어링 함수입니다.
- **왜 중요한가:** 빠른 1단계 검색은 후보 커버리지의 최대화를 목표로 하는 반면, 재정렬(reranking)은 제한된 컨텍스트 윈도우에 도달하는 증거의 품질을 개선할 수 있습니다.
- **실무에서는:** 하이브리드 검색으로 50개의 후보를 검색하고, 크로스 인코더로 각 쿼리-문서 쌍을 점수화하며, 상위 5개의 지원되는 청크를 생성 단계로 전달합니다.
- **흔한 혼동:** Reranker는 전체 코퍼스를 검색하지 않습니다. 검색이 이미 찾은 후보만 재정렬할 뿐입니다.
- **관련 용어:** Hybrid Retrieval, Semantic Search, RAG (Retrieval-Augmented Generation)

### Retry Budget - 재시도 예산
- **분류:** 신뢰성 및 운영
- **실제 의미:** 재시도 트래픽에 대한 상한으로, 일반적으로 원본 요청 대비 비율이나 시간 윈도우 기준으로 표현되며, 재시도가 무제한적인 용량을 소모하는 것을 방지합니다.
- **왜 중요한가:** 의존성이 느려지거나 실패할 때, 제한 없는 재시도는 시스템이 가장 여유 용량이 부족한 시점에 부하를 배가시킵니다.
- **실무에서는:** 첫 시도와 재시도를 별도로 계산하고, 서비스 및 테넌트별로 상한을 설정하며, 마감 기한을 준수하고, 지터가 포함된 백오프를 사용하며, 비일시적(non-transient)이거나 비멱등적(non-idempotent)인 실패에 대해서는 재시도를 중단합니다.
- **흔한 혼동:** Retry Budget은 추가 시도 횟수를 제한합니다. Error Budget은 SLO가 허용하는 사용자 가시적 신뢰성 저하를 측정합니다.
- **관련 용어:** Retry with Backoff, Error Budget, Rate Limit, Admission Control
- **출처:** [Google SRE: Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)

### Retry with Backoff - 백오프 재시도
- **분류:** AI 네이티브 개발
- **실제 의미:** 점진적으로 길어지는 지연 후 실패한 일시적 작업을 반복하는 것으로, 일반적으로 랜덤 지터와 엄격한 재시도 한도를 포함합니다.
- **왜 중요한가:** 즉각적인 동기화된 재시도는 장애를 악화시키고, 속도 제한(rate limits)을 소모하며, 사이드 이펙트를 중복시킬 수 있습니다.
- **실무에서는:** 제한된 지수적 지연 후 제공자(provider)의 시간 초과를 재시도하고, 서버의 재시도 가이드라인을 준수하며, 모든 쓰기 작업에 대해 멱등성 키(idempotency key)를 재사용합니다.
- **흔한 혼동:** 영구적인 유효성 검사 또는 권한 오류는 재시도하지 않으며, 중복 방지 전략 없이 비멱등적(non-idempotent) 작업을 재시도하지 않습니다.
- **관련 용어:** Idempotency, Rate Limit, Circuit Breaker, Backpressure

### Reviewer Agent - 리뷰어 에이전트
- **분류:** AI 네이티브 개발
- **실제 의미:** 명시된 기준에 따라 다른 에이전트의 산출물이나 결정을 검사하고, 발견 사항이나 판정을 반환하도록 지정된 에이전트입니다.
- **왜 중요한가:** 역할 분리(separation of roles)는 누락을 포착하는 데 도움이 될 수 있지만, 리뷰어가 독립적인 증거와 구체적인 평가 기준(rubric)을 받을 때에만 효과가 있습니다.
- **실무에서는:** 한 에이전트가 패치를 생성한 후, 별도의 리뷰어에게 diff, 범위 계약, 저장소 규칙 및 테스트 출력 결과를 전달하고, 라인별 구체적인 발견 사항을 요구합니다.
- **흔한 혼동:** 두 번째 모델 호출이 자동으로 독립적이거나 정확하지는 않습니다. 공유된 컨텍스트, 모델 편향 및 모호한 기준은 동일한 실수를 반복할 수 있습니다.
- **배울 곳:** [Reviewer Agent](../phases/14-agent-engineering/39-reviewer-agent/)
- **관련 용어:** Coding Agent, Verification Gate, Scope Contract, LLM-as-a-Judge

### RLHF (Reinforcement Learning from Human Feedback) - RLHF (인간 피드백 기반 강화 학습)
- **분류:** 수학 및 학습
- **흔히 하는 말:** 인간의 선호도를 바탕으로 모델을 학습하는 것.
- **실제 의미:** 인간의 피드백을 사용하여 보상 또는 선호 신호를 학습하고, 해당 신호에 대해 모델 정책을 최적화하는 파이프라인 계열입니다. 구현 방식은 다양하며, 모든 구현이 동일한 강화 학습 알고리즘을 사용할 필요는 없습니다.
- **흔한 혼동:** RLHF는 수집된 피드백에서 학습된 대리(proxy)를 최적화합니다. 모든 사용자나 상황과의 광범위한 정렬(alignment)을 보장하지는 않습니다.
- **배울 곳:** [Reinforcement Learning from Human Feedback](../phases/10-llms-from-scratch/07-rlhf/)
- **출처:** [InstructGPT paper](https://arxiv.org/abs/2203.02155)
- **관련 용어:** DPO (Direct Preference Optimization), SFT (Supervised Fine-Tuning), Alignment

### Rollback - 롤백
- **분류:** 신뢰성 및 운영
- **실제 의미:** 현재 릴리스가 운영, 품질 또는 안전 기준을 위반할 경우, 이전에 알려진 배포나 구성으로 복원하는 것.
- **왜 중요한가:** 에이전트 및 모델 변경은 사전 배포 평가에도 불구하고 프로덕션 환경에서 실패할 수 있으므로, 롤아웃 전에 복구 절차를 설계해야 합니다.
- **실무에서는:** 버전화된 아티팩트와 구성을 보관하고, 롤백 트리거를 정의하며, 명령 및 데이터의 영향을 연습(rehearse)하고, 복원 후 서비스 건강 상태를 검증합니다.
- **흔한 혼동:** 코드 롤백은 데이터베이스 마이그레이션, 외부 사이드 이펙트, 캐시된 출력물, 또는 나쁜 릴리스가 작성한 데이터를 자동으로 되돌리지 않습니다.
- **관련 용어:** Canary Release, Checkpoint, Regression Test, Durable Execution
- **출처:** [Kubernetes Deployments: Rolling Back](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#rolling-back-a-deployment)

### ROUGE
- **분류:** 평가 및 안전
- **흔히 하는 말:** 요약에 자주 사용되는 참조 겹침 지표입니다.
- **실제 의미:** n-gram 겹침이나 최장 공통 부분 수열과 같은 단위를 사용하여 생성된 텍스트를 참조 텍스트와 비교하는 지표 계열입니다.
- **흔한 혼동:** 표면적인 겹침은 의미적 등가성을 놓칠 수 있으며, 사실적 품질을 증명하지 못한 채 복사된 표현을 보상할 수 있습니다.
- **관련 용어:** Evaluation (Eval), Precision & Recall, LLM-as-a-Judge

## S

### Sandbox - 샌드박스
- **분류:** 에이전트 및 도구
- **실제 의미:** 에이전트가 파일, 프로세스, 네트워크 대상, 자격 증명 및 호스트 자원에 접근하는 것을 제한하는 격리된 실행 환경입니다.
- **왜 중요한가:** 생성된 코드와 도구 호출은 잘못되거나 악성일 수 있습니다. 격리는 그 영향 범위를 제한하고 일회성 검증을 실용적으로 만듭니다.
- **실무에서는:** 읽기 전용 기본 이미지, 범위가 지정된 쓰기 가능한 작업 공간, 프로덕션 비밀이 없는 환경, 명시적인 네트워크 허용 목록을 갖춘 임시 컨테이너에서 테스트를 실행합니다.
- **흔한 혼동:** 샌드박스는 영향을 줄입니다. 내부 코드가 정확하거나 무해하다는 것을 보장하지는 않습니다.
- **배울 곳:** [Production Agent Runtimes](../phases/14-agent-engineering/29-production-runtimes/)
- **관련 용어:** Least Privilege, Approval Gate, Coding Agent, Guardrails

### Saturation - 포화
- **분류:** 신뢰성 및 운영
- **실제 의미:** 제한된 자원이나 서비스가 용량을 소진한 정도이며, 즉시 시작할 수 없는 대기 중인 작업도 포함합니다.
- **왜 중요한가:** 사용률만으로는 허용 가능한 수준으로 보일 수 있지만, 메모리, 가속기 슬롯, 큐 깊이 또는 다운스트림 할당량이 이미 유용한 처리량을 제한하고 있을 수 있습니다.
- **실무에서는:** 각 핵심 자원을 식별하고, 활성 및 대기 작업을 측정하며, 포화 상태를 꼬리 지연 시간 및 오류와 연관시키고, 큐가 불안정한 성장 영역에 진입하기 전에 경고를 발생시킵니다.
- **흔한 혼동:** 포화 상태는 하나의 보편적인 백분율이 아닙니다. 제한 자원과 그 큐잉 동작은 워크로드와 아키텍처에 따라 다릅니다.
- **관련 용어:** Observability, Autoscaling, Backpressure, Tail Latency
- **출처:** [Google SRE: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)

### Scope Contract - 범위 계약
- **분류:** AI 네이티브 개발
- **실제 의미:** 작업의 목표, 허용 및 금지된 영역, 예상 산출물, 검증 요구 사항 및 중단 조건을 정의하는 구체적인 합의입니다.
- **왜 중요한가:** 에이전트가 좁은 수정을 검토 불가능한 리팩토링으로 바꾸거나, 증거 없이 완료를 주장하는 것을 방지합니다.
- **실무에서는:** 파서 모듈과 그 테스트만 변경할 수 있고, 공개 API는 호환성을 유지해야 하며, 지정된 테스트 스위트가 통과해야 한다고 명시합니다.
- **흔한 혼동:** 작업 설명은 원하는 것을 나타냅니다. 범위 계약은 경계와 증명을 정의합니다.
- **배울 곳:** [Scope Contracts](../phases/14-agent-engineering/36-scope-contracts/)
- **관련 용어:** Coding Agent, Patch, Verification Gate, Handoff

### Self-Attention - 셀프 어텐션
- **분류:** 모델 및 추론
- **흔히 하는 말:** 토큰이 다른 토큰 중 어떤 것이 중요한지 결정하는 것.
- **실제 의미:** 쿼리, 키 및 값이 동일한 시퀀스 표현에서 파생되는 어텐션입니다. 스케일링된 유사성 점수는 정규화되어 값 결합에 사용되며, 인과적, 패딩, 지역적 또는 기타 마스크의 영향을 받습니다.
- **왜 중요한가:** 문맥에 민감한 토큰 표현을 구축하지만, 허용된 어텐션 패턴은 아키텍처에 따라 다릅니다.
- **흔한 혼동:** 모든 토큰이 항상 다른 모든 토큰에 어텐션할 수 있는 것은 아닙니다. 인과적 및 희소 모델은 연결을 의도적으로 제한합니다.
- **배울 곳:** [Self-Attention from Scratch](../phases/07-transformers-deep-dive/02-self-attention-from-scratch/)
- **관련 용어:** Attention, Transformer, Context Window

### Semantic Cache - 시맨틱 캐시
- **분류:** AI 네이티브 개발
- **실제 의미:** 선택된 표현과 임계값 하에서 새 요청이 충분히 유사하다고 판단될 때 이전 결과를 재사용하는 캐시입니다.
- **왜 중요한가:** 반복적인 의도에 대해 지연 시간과 비용을 줄일 수 있지만, 잘못된 매칭은 오래되거나 사용자에게 부적절한 출력을 반환할 수 있습니다.
- **실무에서는:** 정규화된 의도별로 저위험 FAQ 답변을 캐시하고, 키에 테넌트 및 정책 버전을 포함하며, 개인화되거나 시간 민감한 요청에 대해서는 캐시를 우회합니다.
- **흔한 혼동:** 의미적 유사성이 두 요청의 정답이 동일하다는 것을 보장하지는 않습니다. 의미 캐시는 이전 결과를 재사용하는 반면, 접두어 캐시는 정확한 토큰의 KV 상태를 재사용하며, 프롬프트 캐시는 제공자 또는 애플리케이션의 자격 요건 규칙을 따릅니다.
- **관련 용어:** Prompt Cache, Embedding, Cost per Successful Task, Grounding

### Semantic Search - 시맨틱 검색
- **분류:** 검색 및 생성
- **흔히 하는 말:** 정확한 단어가 아닌 의미로 검색하는 것.
- **실제 의미:** 쿼리와 후보를 임베딩 공간으로 표현하고, 벡터 유사도 함수를 사용하여 후보를 순위를 매기는 검색 기법입니다.
- **왜 중요한가:** 유의어 및 개념적으로 관련된 텍스트를 검색할 수 있지만, 정확한 식별자나 희소한 문자열은 여전히 어휘 검색이 필요할 수 있습니다.
- **관련 용어:** Embedding, Hybrid Retrieval, Vector Database, Reranker

### Separation of Duties - 직무 분리
- **분류:** 보안 및 거버넌스
- **실제 의미:** 상충하는 책임이나 권한을 독립적인 역할로 나누어, 하나의 주체가 다른 주체의 승인된 결정 없이는 고위험 작업을 완료할 수 없도록 하는 것입니다.
- **왜 중요한가:** 침해된 계정이나 실수한 에이전트가 동일한 중대한 변경을 제안, 승인, 실행 및 은폐할 수 있어서는 안 됩니다.
- **실무에서는:** 아티팩트 생성과 릴리스 승인을 분리하고, 서로 다른 신원을 사용하며, 두 결정을 모두 감사 로그에 보존하고, 사후 검토가 포함된 긴급 접근을 정의합니다.
- **흔한 혼동:** 직무 분리는 상충하는 권한에 관한 것이지, 단순히 동일한 자격 증명을 공유하는 여러 사람이나 에이전트에게 작업을 할당하는 것이 아닙니다.
- **관련 용어:** Approval Gate, Reviewer Agent, Audit Log, Least Privilege
- **출처:** [NIST SP 800-53 Rev. 5](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final)

### Service Level Indicator (SLI) - 서비스 수준 지표 (SLI)
- **분류:** 신뢰성 및 운영
- **실제 의미:** 성공 요청 비율이나 임계값 미만의 지연 시간 등, 정의된 사용자 관련 경계에서의 서비스 동작을 정량적으로 측정하는 지표입니다.
- **왜 중요한가:** 관찰된 행동, 대상 이벤트, 측정 지점이 명시될 때에만 신뢰성 논의가 실행 가능한 것이 됩니다.
- **실무에서는:** 분자, 분모, 제외 항목, 데이터 소스, 집계 기간을 정의한 후, 해당 지표가 사용자가 실제로 경험하는 결과를 추적하는지 검증합니다.
- **흔한 혼동:** SLI는 측정값입니다. SLO는 정의된 기간에 대해 그 측정값에 적용되는 목표입니다.
- **관련 용어:** Service Level Objective (SLO), Availability, Tail Latency, Observability
- **출처:** [Google SRE: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)

### Service Level Objective (SLO) - 서비스 수준 목표 (SLO)
- **분류:** 신뢰성 및 운영
- **실제 의미:** 명시된 대상 집단과 측정 기간에 대한 서비스 수준 지표의 목표 범위 또는 임계값입니다.
- **왜 중요한가:** 예상되는 사용자 결과를 모니터링, 용량, 릴리스 위험, 사고 결정에 대한 운영 경계로 변환합니다.
- **실무에서는:** 사용자가 중요하게 여기는 지표를 선택하고, 현재 성능이 아닌 제품 요구 사항에 따라 목표를 설정하며, 기간과 제외 항목을 정의하고, 에러 버짓(error budget) 정책을 첨부합니다.
- **흔한 혼동:** SLO는 내부 신뢰성 목표입니다. 계약상 서비스 수준 합의(SLA)는 구제 조치를 포함할 수 있으며, 다른 정의를 사용할 수 있습니다.
- **배울 곳:** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **관련 용어:** Service Level Indicator (SLI), Error Budget, Availability, Goodput
- **출처:** [Google SRE: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)

### SFT (Supervised Fine-Tuning) - 지도 미세 조정 (SFT)
- **분류:** 수학 및 학습
- **흔히 하는 말:** 예제 입력과 원하는 출력에 대한 학습입니다.
- **실제 의미:** 사전 학습된 모델을 입력과 원하는 응답의 쌍으로 미세 조정하여, 학습 분포 하에서 시연된 행동을 학습하도록 하는 것입니다.
- **흔한 혼동:** SFT는 채팅을 넘어 많은 행동을 적응시킬 수 있으며, 예제의 품질이 어떤 행동이 강화될지를 결정합니다.
- **관련 용어:** Fine-tuning, DPO (Direct Preference Optimization), RLHF (Reinforcement Learning from Human Feedback)

### Shadow Traffic - 섀도 트래픽
- **분류:** 신뢰성 및 운영
- **실제 의미:** 후보 시스템의 응답이 주요 사용자 응답 경로 밖에 있는 동안, 후보 시스템을 관찰하기 위해 라이브 요청 트래픽의 사본을 전송하는 것입니다. 사본된 요청은 여전히 실행되므로, 그 사이드 이펙트(부수 효과)는 격리되어야 합니다.
- **왜 중요한가:** 합성 테스트에서는 발견되지 않는 실패를 드러낼 수 있으며, 실제 입력 형태와 부하를 후보 시스템에 노출시키면서도 사용자 영향은 제한합니다.
- **실무에서는:** 민감한 필드를 제거하거나 토큰화하고, 도구 및 의존성을 샌드박스화되거나 동작하지 않는(no-op) 대상으로 라우팅하며, 기능 경계에서 쓰기를 차단하고, 요청 상관관계를 유지하며, 섀도 부하가 사용자 트래픽과 경쟁하지 않도록 방지합니다.
- **흔한 혼동:** 후보 응답을 주요 경로에서 제외한다고 해서 실행의 사이드 이펙트가 없어지는 것은 아닙니다. 카나리 릴리스는 트래픽의 통제된 비율에 대해 후보 시스템이 실제 사용자에게 서비스를 제공한다는 점에서 다릅니다.
- **배울 곳:** [Shadow, Canary, and Progressive Delivery](../phases/17-infrastructure-and-production/20-shadow-canary-progressive/)
- **관련 용어:** Canary Release, Evaluation (Eval), Trace, Model Serving
- **출처:** [Istio Traffic Mirroring](https://istio.io/latest/docs/tasks/traffic-management/mirroring/)

### Shared Embedding Space - 공유 임베딩 공간
- **분류:** 멀티모달 시스템
- **실제 의미:** 서로 다른 모달리티의 표현들을 동일한 유사도 함수로 비교할 수 있는 공통 벡터 공간입니다.
- **왜 중요한가:** 두 항목이 원시 표현(raw representation)을 공유할 필요가 없이, 텍스트에서 이미지를 찾는 것과 같은 교차 모달리티 검색 및 매칭을 가능하게 합니다.
- **실무에서는:** 짝이 있는 데이터와 짝이 없는 음(negative) 데이터를 의도적으로 학습하고, 목적 함수가 요구할 경우 벡터를 정규화하며, 양방향 검색 성능을 평가하고, 하위 그룹 및 언어별 성능을 점검합니다.
- **흔한 혼동:** 벡터 차원을 공유한다고 해서 공유된 의미론적 공간이 생성되는 것은 아닙니다. 학습 목적 함수와 데이터가 교차 모달리티 비교 가능성을 확립해야 합니다.
- **배울 곳:** [CLIP Contrastive Pretraining](../phases/12-multimodal-ai/02-clip-contrastive-pretraining/)
- **관련 용어:** Embedding, Cosine Similarity, Modality Alignment, Semantic Search
- **출처:** [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html)

### Skill Bundle - 스킬 번들
- **분류:** 에이전트 및 도구
- **실제 의미:** `SKILL.md`를 포함하여 워크플로우가 요구하는 모든 참조, 스크립트, 자산, 픽스처 및 동반 파일을 포함한 완전한 설치 가능한 스킬 디렉토리입니다.
- **왜 중요한가:** 진입 파일만 복사하면 자원이 누락된 유효해 보이는 지침이 남거나, 워크플로우가 의존하는 결정적 코드가 손실될 수 있습니다.
- **실무에서는:** 트리를 하나의 단위로 설치하고, 해시와 소스 리비전을 기록하며, 설치된 사본을 검증하고, 기존 번들을 교체하기 전에 충돌을 표시합니다.
- **흔한 혼동:** `SKILL.md`는 진입점이며, 전체 산출물이 아닐 수 있습니다.
- **배울 곳:** [Skill Evals, Packaging, and Portability](../phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/)
- **관련 용어:** Agent Skill, Skill Catalog, Reproducible Build, Provenance Attestation
- **출처:** [Agent Skills specification](https://agentskills.io/specification)

### Skill Catalog - 스킬 카탈로그
- **분류:** Agents & tools
- **실제 의미:** 자격을 갖춘 스킬의 간결한 모델 가시 인벤토리로, 일반적으로 이름, 설명 및 내부 소스 식별자와 같은 라우팅 메타데이터를 포함하며 모든 스킬 본문을 포함하지는 않습니다.
- **왜 중요한가:** 카탈로그는 에이전트가 모든 설치된 패키지를 작업 컨텍스트에 로드하지 않고도 관련 절차를 발견할 수 있게 합니다.
- **실무에서는:** 패키지를 먼저 검증하고, 명시적인 중복 이름 정책을 적용하며, 직렬화된 카탈로그 예산을 측정하고, 단축, 생략 또는 가려진 항목에 대한 진단 정보를 유지합니다.
- **흔한 혼동:** 카탈로그 항목은 스킬이 발견 가능함을 의미합니다. 본문이 활성화되거나 도구가 승인되었음을 의미하지는 않습니다.
- **배울 곳:** [Skill Discovery and Progressive Disclosure](../phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/)
- **관련 용어:** Skill Discovery, Skill Invocation, Progressive Disclosure, Token Budget
- **출처:** [Agent Skills specification](https://agentskills.io/specification)

### Skill Discovery - 스킬 발견
- **분류:** Agents & tools
- **실제 의미:** 설정된 루트를 검색하고, 후보 스킬 디렉토리를 식별하며, 패키지 계약을 검증하고, 범위와 출처를 첨부하며, 충돌을 해결하고, 자격을 갖춘 카탈로그 항목을 게시하는 런타임 파이프라인입니다.
- **왜 중요한가:** 결정론적 발견은 모델 라우팅이 시작되기 전에 누락, 형식 오류, 가려짐, 안전하지 않은 패키지를 진단할 수 있게 합니다.
- **실무에서는:** 검색 범위와 중복 동작을 선언하고, 심링크 처리 방식을 결정하며, 자원 탈출을 거부하고, 각 후보가 승인되거나 거부된 이유를 기록합니다.
- **흔한 혼동:** 스킬 발견은 `SKILL.md`라는 파일명을 찾기 위한 무제한 재귀 검색이 아닙니다. 설치 위치와 우선순위는 런타임 정책입니다.
- **배울 곳:** [Skill Discovery and Progressive Disclosure](../phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/)
- **관련 용어:** Skill Catalog, Skill Bundle, Progressive Disclosure, Trust Boundary
- **출처:** [Agent Skills client implementation guide](https://agentskills.io/client-implementation/adding-skills-support)

### Skill Invocation - 스킬 호출
- **분류:** 에이전트 및 도구
- **실제 의미:** 자격을 갖춘 인간, 모델, 애플리케이션 또는 다른 스킬이 스킬을 선택하고 그 지침이 작업 컨텍스트에 진입하도록 하는 런타임 중개 프로세스입니다.
- **왜 중요한가:** 명시적 사용자 접근, 암시적 모델 라우팅, 활성화, 인자 바인딩, 도구 권한, 실행은 각각 다른 실패 모드를 가진 별개의 결정입니다.
- **실무에서는:** 행위자 정책을 정의하고, 긍정적 요청과 근접 실패 요청으로 설명을 평가하며, 선택된 패키지 식별자를 기록하고, 호스트별 호출 필드를 테스트된 어댑터에 유지합니다.
- **흔한 혼동:** 호출은 지침을 활성화합니다. 자동으로 명령을 실행하거나 승인 및 샌드박스 정책을 우회하지는 않습니다.
- **배울 곳:** [Skill Invocation and Routing](../phases/13-tools-and-protocols/25-skill-invocation-and-routing/)
- **관련 용어:** Agent Skill, Skill Catalog, Approval Gate, Sandbox
- **출처:** [Evaluating Agent Skills](https://agentskills.io/skill-creation/evaluating-skills)

### Softmax - 소프트맥스
- **분류:** 수학 및 학습
- **흔히 하는 말:** 로짓을 정규화된 양의 값으로 변환하는 함수입니다.
- **실제 의미:** `softmax(x_i) = exp(x_i) / sum(exp(x_j))`로 정의되며 수치적 안정화 기법으로 구현되는 함수입니다. 출력값은 양수이며 합이 1이므로 범주형 분포를 매개변수화하는 데 사용할 수 있습니다.
- **흔한 혼동:** Softmax 값은 실제 세계의 정확성에 대한 보정된 확률로 자동적으로 간주되지 않습니다.
- **관련 용어:** Temperature, Cross-Entropy, Attention

### Software Bill of Materials (SBOM) - 소프트웨어 부품 목록 (SBOM)
- **분류:** 보안 및 거버넌스
- **다른 이름:** SBOM
- **실제 의미:** 제품이나 산출물과 관련된 소프트웨어 구성 요소 및 관계를 구조화된 형태로 나열한 목록으로, 보통 버전, 공급자, 라이선스, 식별자 등이 포함됩니다.
- **왜 중요한가:** 소프트웨어 변경이나 취약점이 발생했을 때, 영향을 받는 의존성, 라이선스 의무, 공급망 노출을 평가하려면 구성 요소 목록이 필요합니다.
- **실무에서는:** 신뢰할 수 있는 빌드 과정에서 SBOM을 생성하고, 릴리스 산출물에 이를 연결하며, 정책 검사를 통해 이를 검증하고, 의존성이나 패키징이 변경될 때마다 업데이트합니다.
- **흔한 혼동:** SBOM은 목록일 뿐이며, 생성 과정과 출처가 신뢰할 수 있는 경우가 아니라면 구성 요소가 보안적이며, 라이선스가 정확하고, 실제로 존재한다는 증거가 아닙니다.
- **관련 용어:** Provenance Attestation, Reproducible Build, Data Provenance, Audit Log
- **출처:** [SPDX 3.0.1 specification](https://spdx.github.io/spdx-spec/v3.0/)

### Speculative Decoding - 추론적 디코딩
- **분류:** 모델 및 추론
- **실제 의미:** 더 저렴한 초안 생성 과정이 여러 토큰을 제안하고, 대상 모델이 이러한 초안 위치를 병렬로 점수화하는 추론 방법입니다. 정확한 샘플링 변형에서는 수용 및 보정 규칙을 통해 대상 모델의 출력 분포를 유지합니다.
- **왜 중요한가:** 초안이 수용될 경우 순차적인 대상 모델 디코딩 작업을 줄일 수 있으며, 대상 모델의 학습된 가중치를 변경할 필요가 없습니다.
- **실무에서는:** 실제 프롬프트에서 수용률과 엔드투엔드 지연 시간을 측정하고, 초안 모델의 오버헤드를 포함하며, 구현이 의도된 디코딩 분포를 유지하는지 검증합니다.
- **흔한 혼동:** 추론적 디코딩은 일반적인 모델 라우팅이나 검증되지 않은 자동 완성이 아닙니다. 정확한 변형은 수용 및 보정을 통해 대상 분포를 유지하는 반면, 근사 변형은 속도를 위해 그 보증을 희생할 수 있습니다.
- **관련 용어:** Autoregressive, KV Cache, Decoding Strategy, Tokens per Second (TPS)
- **출처:** [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)

### Stateless MCP - 상태 비저장 MCP
- **분류:** 에이전트 및 도구
- **실제 의미:** MCP 2026-07-28 요청 모델입니다. 모든 요청은 `params._meta`에 프로토콜 버전과 클라이언트 기능을 포함하며, 결과는 명시적인 `resultType`를 포함합니다. 초기화 핸드셰이크, 연결, `Mcp-Session-Id`에 의해 프로토콜 상태가 키화되지 않습니다.
- **왜 중요한가:** 모든 워커는 요청의 내용과 인증 컨텍스트를 통해 요청을 검증하고 처리할 수 있습니다. 이는 숨겨진 연결 친화성을 피하고 수평 라우팅을 더 쉽게 추론할 수 있게 합니다.
- **실무에서는:** `server/discover`를 구현하고, 모든 호출에서 요청 메타데이터를 재구성하며, JSON-RPC 본문에 대해 전송 헤더를 검증하고, 연속성이 필요할 때 서버가 발급한 애플리케이션 핸들을 일반적인 도구 인자로 전달합니다.
- **흔한 혼동:** 상태 비저장(stateless) MCP는 프로토콜 세션을 제거하는 것이지, 애플리케이션 상태, 전송 연결, 스트리밍 응답, 작업, 명시적 핸들을 제거하는 것이 아닙니다.
- **배울 곳:** [MCP Fundamentals](../phases/13-tools-and-protocols/06-mcp-fundamentals/)
- **관련 용어:** MCP (Model Context Protocol), Multi Round-Trip Request (MRTR), Tool Contract, Idempotency
- **출처:** [MCP 2026-07-28 key changes](https://modelcontextprotocol.io/specification/2026-07-28/changelog); [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)

### Stochastic Gradient Descent (SGD) - 확률적 경사 하강법 (SGD)
- **분류:** 수학 및 학습
- **다른 이름:** SGD
- **실제 의미:** 전체 학습 데이터셋이 아닌 샘플링된 예제나 미니배치에서 추정된 그래디언트로부터 파라미터를 업데이트하는 옵티마이저 계열입니다.
- **왜 중요한가:** 그래디언트 노이즈, 모멘텀, 배치 스케일링, 현대 학습에서 사용되는 적응형 옵티마이저를 이해하는 기준점입니다.
- **실무에서는:** 배치 샘플링, 학습률, 사용된 경우 모멘텀, 스케줄을 기록한 후, 동일한 업데이트 또는 토큰 예산 하에서 검증 동작을 비교합니다.
- **흔한 혼동:** 현재 실무에서는 SGD가 보통 미니배치 SGD를 의미하며, 유용한 학습률은 하나의 보편적인 배치 스케일링 규칙을 따르지 않습니다.
- **관련 용어:** Gradient Descent, Batch Size, Learning Rate, Optimizer
- **출처:** [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838); [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)

### Stop Sequence - 중단 시퀀스
- **분류:** 모델 및 추론
- **실제 의미:** 디코딩 시스템이 이를 만나면 생성을 중단하도록 애플리케이션이 지정한 토큰이나 텍스트 패턴입니다.
- **왜 중요한가:** 스톱 시퀀스는 모델이 완료했다고 의미적으로 판단할 때까지 기다리지 않고 출력 프로토콜과 다중 부분 생성을 바인딩합니다.
- **실무에서는:** 모호하지 않은 구분자를 선택하고, 토큰화 및 부분 스트리밍 매칭을 테스트하며, 출력 길이 및 스키마 검증을 여전히 강제해야 합니다.
- **흔한 혼동:** 스톱 시퀀스는 기계적인 디코딩 조건일 뿐, 답변이 완결되었거나 에이전트 목표가 충족되었다는 증거가 아닙니다.
- **관련 용어:** Decoding Strategy, Structured Output, Token, Termination Condition
- **출처:** [Transformers text-generation documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

### Streaming - 스트리밍
- **분류:** 모델 및 추론
- **흔히 하는 말:** 생성되는 대로 출력을 보여주는 것.
- **실제 의미:** 완전한 결과가 준비되기 전에 점진적인 응답 이벤트를 전달하는 것입니다. 스트림은 API에 따라 토큰 텍스트, 구조화된 델타, 도구 호출 인자, 사용량 메타데이터 또는 상태 이벤트를 포함할 수 있습니다.
- **왜 중요한가:** 체감 응답성을 개선하지만, 모델이 완전한 답변을 생성하는 실제 시간을 줄이지는 않습니다.
- **흔한 혼동:** 네트워크 전송, 이벤트 형식, 청크 경계는 제공자마다 다르며, 단어 또는 토큰과 일치한다고 보장되지 않습니다.
- **배울 곳:** [Production LLM Application](../phases/11-llm-engineering/13-production-app/)
- **관련 용어:** Time to First Token (TTFT), Autoregressive, Observability

### Structured Output - 구조화된 출력
- **분류:** 에이전트 및 도구
- **실제 의미:** 기계가 읽을 수 있는 스키마에 따라 모델 출력이 제한되거나 검증되어, 애플리케이션 코드가 자유 형식 산문(prose)을 파싱하지 않고도 필드를 소비할 수 있도록 하는 것입니다.
- **왜 중요한가:** 모델과 소프트웨어 경계에서의 형식 모호성을 줄이고 필드 단위 검증 및 재시도를 가능하게 합니다.
- **실무에서는:** 허용된 심각도 enum, 증거 배열, nullable한 에스컬레이션 이유를 포함하는 인시던트 트리아지 결과를 요구하며, 스키마를 통과하지 못하는 응답은 거부해야 합니다.
- **흔한 혼동:** 스키마가 유효한 출력에도 잘못된 값이 포함될 수 있습니다. 구조가 사실적 검증을 의미하지는 않습니다.
- **배울 곳:** [Structured Outputs](../phases/11-llm-engineering/03-structured-outputs/)
- **관련 용어:** Function Calling, Tool Contract, Verification Gate

### Swarm - 스웜
- **분류:** 에이전트 및 도구
- **흔히 하는 말:** 하나의 고정된 제어자 없이 많은 에이전트가 협력하는 것.
- **실제 의미:** 지역적 에이전트 결정과 메시지 교환이 시스템 수준의 행동을 만들어내는 느슨하게 조정된 다중 에이전트 패턴입니다. 이 용어는 일관되지 않게 사용되므로, 실제 토폴로지, 상태 소유권 및 종료 규칙을 명시해야 합니다.
- **흔한 혼동:** 여러 개의 이름 붙은 에이전트가 유용한 전문화나 창발적 조정을 보장하지는 않습니다.
- **관련 용어:** Agent, Reviewer Agent, Handoff, Agent State

### System Prompt - 시스템 프롬프트
- **분류:** 프롬팅 및 컨텍스트
- **흔히 하는 말:** 모델 상호작용을 위한 개발자 제어 지침.
- **실제 의미:** 해당 제공자의 지침 계층 내에서 행동과 제약 조건을 설정하기 위해 애플리케이션이 제공하는 제공자 정의 지침 메시지 또는 구성입니다.
- **왜 중요한가:** 시스템 지침은 행동을 안내할 수 있지만, 비밀로 유지될 것이 보장되지 않으며 보안 경계로 취급해서는 안 됩니다.
- **흔한 혼동:** 우선순위 규칙, 메시지 역할, 지속성 및 가시성은 API마다 다릅니다. 현재 제공자 계약을 확인하십시오.
- **배울 곳:** [Instructions as Executable Constraints](../phases/14-agent-engineering/33-instructions-as-executable-constraints/)
- **관련 용어:** Prompt Engineering, Prompt Injection, Context Engineering, Guardrails

## T

### Tail Latency - 꼬리 지연
- **분류:** 신뢰성 및 운영
- **실제 의미:** 요청 중 가장 느린 부분이 경험하는 지연으로, 일반적으로 명시된 작업 부하와 시간 창에서 높은 백분위로 요약됩니다.
- **왜 중요한가:** 평균은 건강해 보일 수 있지만, 큐잉, 경합, 재시도 또는 가변적인 요청 비용으로 인해 의미 있는 사용자 그룹이 훨씬 더 오래 기다릴 수 있습니다.
- **실무에서는:** 경로와 워크로드별로 여러 백분위수를 보고하고, 문서화된 규칙에 따라 타임아웃을 검열(censored)된 관측값이나 실패한 관측값으로 유지하며, 느린 요청을 의존성 전체에 걸쳐 추적합니다.
- **흔한 혼동:** 꼬리 지연(tail latency)은 가장 느린 단일 요청이 아니며, 백분위수, 모집단, 측정 경계 없이는 의미가 없습니다.
- **배울 곳:** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **관련 용어:** Time to First Token (TTFT), Time per Output Token (TPOT), Saturation, Goodput
- **출처:** [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)

### Temperature - 온도
- **분류:** 모델 및 추론
- **흔히 하는 말:** 창의성 설정입니다.
- **실제 의미:** 확률 분포가 형성되기 전에 로짓(logits)을 재스케일링하는 디코딩 파라미터입니다. 양의 값이 높을수록 분포가 평평해지고, 양의 값이 낮을수록 날카로워집니다.
- **왜 중요한가:** Temperature는 샘플링 동작을 변경할 뿐, 모델의 지식이나 사실성(factuality)을 변경하지 않습니다.
- **흔한 혼동:** 0 설정은 종종 그리디 디코딩(greedy decoding)으로 구현되지만, 정확한 동작과 결정성(determinism)은 제공자, 샘플러, 시드 지원 및 서빙 시스템에 따라 다릅니다.
- **관련 용어:** Softmax, Autoregressive, Token

### Tensor - 텐서
- **분류:** 데이터 및 표현
- **흔히 하는 말:** 수치 연산에 사용되는 다차원 배열입니다.
- **실제 의미:** 프레임워크가 입력, 파라미터, 활성화 값 및 그래디언트를 표현하기 위해 사용하는, 형태(shape), 데이터 타입, 장치 배치를 가진 타입 지정 배열입니다. 자동 미분 메타데이터는 프레임워크 및 연산에 의존하며, 모든 텐서의 고유한 속성이 아닙니다.
- **관련 용어:** Autograd, Parameter, Mixed Precision

### Tensor Parallelism - 텐서 병렬화
- **분류:** 인프라 및 서빙
- **실제 의미:** 모델 레이어 내의 텐서 연산을 여러 장치에 분할하고, 레이어 계산 중 부분 결과를 결합하기 위해 집단 통신(collective communication)을 사용하는 것입니다.
- **왜 중요한가:** 하나의 레이어가 여러 장치의 메모리와 연산 자원을 사용할 수 있게 하지만, 인터커넥트나 분할 방식이 적합하지 않으면 빈번한 통신이 성능을 지배할 수 있습니다.
- **실무에서는:** 파티션 차원을 모델의 형태에 맞추고, 집합 통신(collective traffic)을 벤치마킹하며, 랭크(rank)를 고속 인터커넥트(interconnect)에 배치하고, 체크포인트 및 서빙 구성에 샤딩(sharding) 레이아웃을 기록합니다.
- **흔한 혼동:** 텐서 병렬화(Tensor parallelism)는 레이어 내부에서 작업을 분할합니다. 파이프라인 병렬화(Pipeline parallelism)는 서로 다른 레이어 그룹을 서로 다른 장치에 배치합니다.
- **배울 곳:** [Scaling and Distributed Training](../phases/10-llms-from-scratch/05-scaling-distributed/)
- **관련 용어:** Tensor, Pipeline Parallelism, Expert Parallelism, Parameter
- **출처:** [Megatron-LM](https://arxiv.org/abs/1909.08053)

### Termination Condition - 종료 조건
- **분류:** 에이전트 및 도구
- **실제 의미:** 에이전트 실행이 성공, 실패, 예산 소진, 안전 경계 도달, 또는 에스컬레이션(escalation) 필요 시에 종료하거나 일시 중지하는 명시적인 규칙입니다.
- **왜 중요한가:** 종료 조건이 없으면 에이전트가 루프에 빠지거나, 사이드 이펙트(side effects)를 반복하거나, 예산을 낭비하거나, 목표를 충족하지 못한 채 완료를 주장할 수 있습니다.
- **실무에서는:** 루프를 시작하기 전에 성공 증거, 최대 단계 및 비용, 재시도 불가능한 오류, 에스컬레이션 상태를 정의합니다.
- **흔한 혼동:** 스톱 시퀀스(stop sequence)는 텍스트 생성을 종료합니다. 종료 조건은 작업이나 워크플로우가 멈춰야 하는지 여부를 결정합니다.
- **관련 용어:** Agent Harness, Token Budget, Verification Gate, Stop Sequence
- **출처:** [AutoGen](https://arxiv.org/abs/2308.08155)

### Test Oracle - 테스트 오라클
- **분류:** AI 네이티브 개발
- **실제 의미:** 관찰된 프로그램의 동작이 올바른지 결정하는 데 사용되는 메커니즘, 사양, 참조, 불변 조건(invariant), 또는 인간 판단입니다.
- **왜 중요한가:** 테스트 입력을 생성하는 것만으로는 충분하지 않습니다. 자동화된 검증은 각 결과를 분류하기 위한 독립적인 기준이 필요합니다.
- **실무에서는:** 실행 가능한 불변 조건, 참조 구현, 스키마, 결정론적(expected) 출력 값을 우선적으로 사용하며, 인간 판단이 여전히 필요한 부분을 문서화합니다.
- **흔한 혼동:** 코드를 작성한 모델은 단순히 자신의 출력값이 올바른지 물어본다고 해서 독립적인 오라클로 취급해서는 안 됩니다.
- **관련 용어:** Regression Test, Verification Gate, Eval Set, Human-in-the-Loop (HITL)
- **출처:** [The Oracle Problem in Software Testing](https://www.computer.org/csdl/journal/ts/2015/05/06963470/13rRUx0geBw)

### Threat Model - 위협 모델
- **분류:** 보안 및 거버넌스
- **실제 의미:** 보호 대상 자산, 신뢰 경계, 잠재적 적대자, 가정된 능력, 공격 경로, 영향 및 계획된 통제에 대한 문서화된 설명입니다.
- **왜 중요한가:** 무엇을 방어하는지, 누구를 방어하는지, 그리고 어떤 가정 하에 방어하는지를 명시하지 않으면 보안 통제 사항을 평가할 수 없습니다.
- **실무에서는:** 모델, 검색, 도구, 사용자 및 외부 서비스 전반에 걸쳐 데이터와 권한을 매핑한 후, 신뢰할 수 있는 남용 경로를 레드팀 사례 및 완화 조치로 전환합니다.
- **흔한 혼동:** 위협 모델은 개연성 있는 위험을 우선시합니다. 시스템이 안전함을 증명하거나 모든 미래의 공격을 예측하는 체크리스트가 아닙니다.
- **관련 용어:** Least Privilege, Prompt Injection, Sandbox, Red Teaming
- **출처:** [NIST SP 800-154](https://csrc.nist.gov/pubs/sp/800/154/ipd); [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.600-1.pdf)

### Time per Output Token (TPOT) - 출력 토큰당 시간 (TPOT)
- **분류:** 인프라 및 서빙
- **실제 의미:** `N > 1`개의 출력 토큰을 가진 단일 요청에 대해, 첫 번째 토큰 이후의 평균 간격은 `(t_N - t_1) / (N - 1)`입니다. 시스템 분포는 이러한 요청별 평균을 집계합니다.
- **왜 중요한가:** 사용자는 첫 번째 토큰을 빠르게 받을 수 있지만, 나머지 답변은 느리게 스트리밍될 수 있으므로, 시작 지연만으로는 생성 응답성을 설명할 수 없습니다.
- **실무에서는:** 각 요청에 대해 TPOT를 별도로 계산하고, 출력 길이 및 동시성에 따라 요청 간 백분위를 보고하며, 모든 토큰 간격을 풀링(pooling)하거나 다른 토크나이저와 측정 경계를 가진 시스템들을 비교하지 않도록 주의합니다.
- **흔한 혼동:** TPOT는 요청별 평균입니다. 개별 토큰 간 지연은 연속된 토큰 사이의 하나의 간격인 반면, 첫 번째 토큰까지의 시간(TTFT)은 출력 시작 전의 대기 시간을 포함합니다.
- **배울 곳:** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **관련 용어:** Decode Phase, Time to First Token (TTFT), Streaming, Goodput
- **출처:** [DistServe](https://arxiv.org/abs/2401.09670)

### Time to First Token (TTFT) - 첫 토큰까지의 시간 (TTFT)
- **분류:** 모델 및 추론
- **다른 이름:** TTFT
- **실제 의미:** 정의된 측정 경계 내에서 생성 요청을 제출한 시점부터 클라이언트가 첫 번째 출력 토큰이나 콘텐츠 이벤트를 수신할 때까지의 경과 시간입니다.
- **왜 중요한가:** TTFT는 체감 응답성에 큰 영향을 미치며, 큐잉, 프롬프트 처리, 캐시, 또는 네트워크 지연을 드러낼 수 있습니다.
- **실무에서는:** 모델, 프롬프트 길이, 지역, 캐시 상태별로 클라이언트 측 TTFT를 기록한 후, 총 완료 시간과 분리하여 분석합니다.
- **흔한 혼동:** TTFT는 초당 토큰 수(tokens per second)가 아닙니다. TTFT는 시작 지연(latency)을 측정하며, 초당 토큰 수는 출력 시작 후의 생성 처리량(throughput)을 측정합니다.
- **관련 용어:** Streaming, Prompt Cache, Observability, Token Budget

### Token - 토큰
- **분류:** 데이터 및 표현
- **흔히 하는 말:** 모델 입력이나 출력의 단어 단위 조각.
- **실제 의미:** 모델 전용 토크나이저가 텍스트, 바이트, 이미지, 오디오 또는 기타 입력 표현에서 생성한 정수 식별자입니다. 토큰은 단어 전체, 단어의 일부, 구두점, 공백, 바이트 시퀀스, 또는 특수 제어 기호일 수 있습니다.
- **흔한 혼동:** 문자 대 토큰 비율은 언어, 콘텐츠, 토크나이저에 따라 다르므로, 대상 모델의 토크나이저나 제공자 도구를 사용하여 개수를 세어야 합니다.
- **배울 곳:** [Tokenizers](../phases/10-llms-from-scratch/01-tokenizers/)
- **관련 용어:** Token Budget, Context Window, Autoregressive

### Token Budget - 토큰 예산
- **분류:** 프롬팅 및 컨텍스트
- **실제 의미:** 지시문, 증거, 히스토리, 도구 결과, 추론 또는 작업 공간, 출력에 걸쳐 토큰 용량을 명시적으로 배분하는 것입니다.
- **왜 중요한가:** 포함된 모든 토큰은 컨텍스트 용량, 지연 시간, 비용에 대해 경쟁합니다. 예산 배분은 고가치 증거를 우선적으로 보존하도록 강제합니다.
- **실무에서는:** 출력 용량을 예약하고, 검색된 청크를 상한으로 제한하며, 오래된 도구 결과를 상태로 요약하고, 모델 한도에 도달하기 전에 중단하거나 압축합니다.
- **흔한 혼동:** 토큰 예산은 계획상의 제약 조건입니다. 모델의 최대 컨텍스트 윈도우와 같은 것이 아닙니다.
- **배울 곳:** [Context Engineering](../phases/11-llm-engineering/05-context-engineering/)
- **관련 용어:** Context Window, Context Engineering, Progressive Disclosure, Cost per Successful Task

### Tokenization - 토큰화
- **분류:** 데이터 및 표현
- **실제 의미:** 입력 표현을 특정 모델이나 토크나이저가 허용하는 순서화된 토큰 식별자로 변환하는 것입니다.
- **왜 중요한가:** 토큰화는 시퀀스 길이, 어휘 경계, 비용 계산, 잘림(truncation) 동작, 그리고 임베딩(embedding) 전 텍스트나 코드가 표현되는 방식을 결정합니다.
- **실무에서는:** 대상 모델에 정확한 토크나이저를 사용하고, 산출물(artifacts)과 함께 버전을 관리하며, 다국어 텍스트, 코드, 공백 및 특수 토큰을 테스트합니다.
- **흔한 혼동:** 토큰화는 항상 단어 분할을 의미하지는 않으며, 두 모델이 동일한 입력에 대해 서로 다른 토큰 수와 ID를 할당할 수 있습니다.
- **관련 용어:** Token, Vocabulary, Byte Pair Encoding (BPE), Embedding
- **출처:** [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

### Tokens per Second (TPS) - 초당 토큰 수 (TPS)
- **분류:** 인프라 및 서빙
- **다른 이름:** TPS, 출력 토큰 처리량
- **실제 의미:** 서빙 시스템이 명시된 범위와 워크로드 하에서 단위 시간당 생성하는 출력 토큰의 수를 보고하는 처리량 측정 지표입니다.
- **왜 중요한가:** 출력 시작 후 생성이 얼마나 빠르게 진행되는지, 그리고 부하 하에서 서빙이 어떻게 동작하는지를 보여줌으로써 시작 지연(latency)을 보완합니다.
- **실무에서는:** TPS가 요청별인지 집계된 것인지 명시하고, 프리필(prefill)을 제외하거나 식별하며, 배치(batch), 동시성, 시퀀스 길이, 하드웨어 및 백분위(latency percentile)를 보고합니다.
- **흔한 혼동:** TPS는 서로 다른 토크나이저, 워크로드, 품질 설정, 측정 경계 간에 직접 비교할 수 없습니다.
- **관련 용어:** Time to First Token (TTFT), Streaming, Prefill, Observability
- **출처:** [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf)

### Tool Contract - 도구 계약
- **분류:** 에이전트 및 도구
- **실제 의미:** 도구 경계에 대한 완전한 합의입니다: 목적, 타입이 지정된 입력, 출력, 검증, 권한, 사이드 이펙트, 오류, 타임아웃, 멱등성(idempotency), 그리고 호출자에게 반환되는 증거를 포함합니다.
- **왜 중요한가:** 스키마는 모델에게 어떤 필드가 존재하는지 알려줍니다. 계약은 시스템에게 도구가 안전한 시점과 실패를 어떻게 처리해야 하는지 알려줍니다.
- **실무에서는:** 허용된 루트, 예상 기본 리비전, 최대 크기, dry-run 모드, 명시적인 충돌 오류, 반환된 patch hash를 포함하여 파일 쓰기 도구를 정의합니다.
- **흔한 혼동:** JSON Schema는 도구 계약의 일부일 뿐, 전체 계약이 아닙니다.
- **배울 곳:** [Tool Use and Function Calling](../phases/14-agent-engineering/06-tool-use-and-function-calling/)
- **관련 용어:** Function Calling, Structured Output, Least Privilege, Idempotency

### Top-k Sampling - Top-k 샘플링
- **분류:** 모델 및 추론
- **실제 의미:** 다음 토큰 분포를 점수가 가장 높은 k개의 후보로 제한하고, 그 확률을 정규화한 후 해당 집합에서 샘플링하는 디코딩 방법입니다.
- **왜 중요한가:** 고정된 최대 후보 수를 유지하면서 샘플링에서 확률이 낮은 긴 꼬리를 제거합니다.
- **실무에서는:** k를 temperature, top-p, stop 설정과 함께 평가하고, 생성된 결과와 함께 완전한 sampler 구성을 기록합니다.
- **흔한 혼동:** Top-k는 고정된 후보 수를 사용하지만, top-p는 단계마다 후보 수가 변하는 확률 질량 임계값을 사용합니다.
- **관련 용어:** Nucleus Sampling (Top-p), Temperature, Decoding Strategy, Logits
- **출처:** [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

### Trace - 추적
- **분류:** AI 네이티브 개발
- **실제 의미:** 모델 호출, 검색, 도구, 상태 전환, 재시도, 승인, 평가에 걸쳐 하나의 요청이나 작업을 상관관계 지어 기록한 것입니다.
- **왜 중요한가:** 다단계 워크플로우에서 시간, 비용, 실패가 어디에서 발생했는지 재구성할 수 있게 해줍니다.
- **실무에서는:** 하나의 trace 식별자를 agent harness를 통해 전파하고, 각 모델 및 도구 작업에 대해 마스킹된 span을 첨부합니다.
- **흔한 혼동:** Trace는 운영 증거를 기록해야 하며, 숨겨진 모델 추론, 비밀, 마스킹되지 않은 민감한 콘텐츠를 노출해서는 안 됩니다.
- **배울 곳:** [OpenTelemetry GenAI Conventions](../phases/14-agent-engineering/23-otel-genai-conventions/)
- **출처:** [OpenTelemetry traces](https://opentelemetry.io/docs/concepts/signals/traces/)
- **관련 용어:** Observability, Agent State, Time to First Token (TTFT), Evaluation (Eval)

### Transfer Learning - 전이 학습
- **분류:** 수학 및 학습
- **흔히 하는 말:** 사전 학습된 모델을 새로운 작업에 재사용하는 것.
- **실제 의미:** 하나의 데이터 분포나 목적 함수에서 학습된 표현이나 파라미터를 시작점으로 삼아, 다른 작업에 적응시키는 것입니다. 전이 가능한 구성 요소와 업데이트 전략은 아키텍처와 작업에 따라 다릅니다.
- **흔한 혼동:** 전이는 후반 레이어로만 제한되지 않으며, 소스 작업과 타겟 작업이 크게 다를 경우 성공적인 전이가 보장되지 않습니다.
- **관련 용어:** Fine-tuning, Feature, SFT (Supervised Fine-Tuning)

### Transformer - 트랜스포머
- **분류:** 모델 및 추론
- **흔히 하는 말:** 많은 최신 언어 모델의 기반 아키텍처.
- **실제 의미:** 어텐션, 위치 정보, 피드포워드 서레이어, 잔여 연결(residual connections), 정규화 등으로 구성된 신경망 아키텍처입니다. 인코더, 디코더, 인코더-디코더 변형은 서로 다른 마스크와 정보 흐름을 사용합니다.
- **왜 중요한가:** 학습 시에는 많은 시퀀스 위치를 병렬로 처리할 수 있지만, 오토리gressive 생성은 여전히 출력 단계별로 생성됩니다.
- **흔한 혼동:** 자기 어텐션(self-attention)이 모든 트랜스포머에서 무제한의 all-to-all 어텐션을 의미하지는 않습니다.
- **배울 곳:** [Build a Full Transformer](../phases/07-transformers-deep-dive/05-full-transformer/)
- **출처:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **관련 용어:** Attention, Self-Attention, Encoder, Decoder

### Trust Boundary - 신뢰 경계
- **분류:** 보안 및 거버넌스
- **실제 의미:** 서로 다른 신뢰 가정을 가진 구성 요소나 주체(principal) 간에 데이터, 지시, 신원, 권한이 교차하는 인터페이스입니다.
- **왜 중요한가:** 경계를 넘나드는 지점에서는 시스템이 행위자를 인증하고, 데이터를 검증하며, 권한을 제한하고, 어떤 주장이 행동에 영향을 미칠 수 있는지 결정해야 합니다.
- **실무에서는:** 사용자, 모델 컨텍스트, 검색 소스, 도구, 네트워크, 데이터 저장소 주변에 경계를 설정한 후, 모든 교차점에 대한 검증 및 권한 부여를 명시합니다.
- **흔한 혼동:** 네트워크 경계는 신뢰 경계의 한 종류일 뿐입니다. 신뢰받지 않는 문서 텍스트가 권한이 있는 에이전트 컨텍스트로 들어오는 것도 신뢰 경계를 넘나듭니다.
- **배울 곳:** [Jailbreak Taxonomy](../phases/19-capstone-projects/82-jailbreak-taxonomy/)
- **관련 용어:** Threat Model, Least Privilege, Sandbox, Indirect Prompt Injection
- **출처:** [Microsoft Learn: Trust Boundary, the Trust Zone Change Element](https://learn.microsoft.com/en-us/training/modules/tm-create-a-threat-model-using-foundational-data-flow-diagram-elements/6-trust-boundary-the-trust-zone-change-element); [OWASP Threat Modeling](https://owasp.org/www-community/Threat_Modeling)

## U

### Underfitting - 과소 적합
- **분류:** 수학 및 학습
- **흔히 하는 말:** 모델이 학습 작업을 충분히 잘 맞추지 못합니다.
- **실제 의미:** 모델이나 학습 설정이 학습 데이터의 유용한 패턴을 포착하기 위해 필요한 효과적인 용량, 최적화, 기능, 또는 학습 신호가 부족합니다.
- **실무에서는:** 먼저 데이터와 최적화를 진단한 후, 학습 시간을 늘리거나, 기능을 변경하거나, 과도한 정규화를 줄이거나, 적절한 용량을 늘리는 것을 고려하십시오.
- **관련 용어:** Overfitting, Loss Function, Hyperparameter

## V

### VAE (Variational Autoencoder) - 변분 오토인코더 (VAE)
- **분류:** 모델 및 추론
- **흔히 하는 말:** 확률적 생성 오토인코더입니다.
- **실제 의미:** 재구성 목적 함수와 근사 사후 분포를 선택된 사전 분포에 가깝게 유지하는 정규화 항으로 학습된 잠재 변수 모델입니다. 재파라미터화 추정자는 확률적 잠재 샘플링을 통해 그래디언트를 전달할 수 있게 합니다.
- **흔한 혼동:** VAE는 모든 잠재 분포를 하나의 고정된 가우시안으로 강제하지 않습니다. 정확한 사전 분포와 근사 사후 분포는 모델링 선택 사항입니다.
- **출처:** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- **관련 용어:** Latent Space, Encoder, Decoder, Diffusion Model

### Vector Database - 벡터 데이터베이스
- **분류:** 검색 및 생성
- **흔히 하는 말:** 벡터 유사성 검색에 최적화된 데이터베이스입니다.
- **실제 의미:** 벡터 표현에 대한 최근접 이웃 쿼리를 지원하며, 종종 메타데이터 필터링, 지속성, 근사 인덱스를 포함하는 저장 및 인덱싱 시스템입니다.
- **흔한 혼동:** 벡터 데이터베이스는 벡터를 저장하고 검색합니다. 고품질 임베딩을 생성하거나 관련성 있는 검색을 보장하지는 않습니다.
- **관련 용어:** Embedding, Semantic Search, Hybrid Retrieval

### Verification Gate - 검증 게이트
- **분류:** 평가 및 안전
- **실제 의미:** 정의된 증거가 정확성 또는 품질 기준을 충족할 때까지 진행을 차단하는 통제 지점입니다.
- **왜 중요한가:** 모델의 완료 주장을 증거에 기반한 결정으로 전환합니다.
- **실무에서는:** 패치가 적용되고, 범위 내 테스트가 통과하며, 금지된 파일이 변경되지 않고, 필수 산출물이 존재할 때까지 코딩 작업의 완료를 방지합니다.
- **흔한 혼동:** 검증은 증거가 기준을 충족하는지 확인합니다. 승인은 증거가 이미 알려진 경우에도 진행할 권한을 부여합니다.
- **배울 곳:** [Verification Gates](../phases/14-agent-engineering/38-verification-gates/)
- **관련 용어:** Approval Gate, Regression Test, Scope Contract, Structured Output

### Vision-Language Model (VLM) - 비전-언어 모델 (VLM)
- **분류:** 멀티모달 시스템
- **실제 의미:** 검색, 설명, 질문 답변, 근거 기반 생성 등의 작업을 위해 시각 및 언어 표현 간의 관계를 학습하거나 이를 jointly 처리하는 모델입니다.
- **왜 중요한가:** VLM의 성능은 단일한 일반적인 능력 라벨이 아니라 시각 인코더, 언어 구성 요소, 연결 메커니즘, 학습 데이터 및 해상도 정책에 의존합니다.
- **실무에서는:** 텍스트 전용 및 시각 전용 통제 기능을 평가하고, 이미지 해상도와 레이아웃을 다양하게 변경하며, 가능한 경우 증거 위치 파악을 요구하고, 시각적 기술 및 언어별로 실패를 보고합니다.
- **흔한 혼동:** 이미지를 수용하는 것은 모델이 이를 올바르게 사용한다는 것을 증명하지 않으며, VLM은 반드시 이미지를 생성할 수 있는 것은 아닙니다.
- **배울 곳:** [Vision-Language Models](../phases/04-computer-vision/25-vision-language-models/)
- **관련 용어:** Multimodal Model, Vision Transformer (ViT), Cross-Attention, Visual Grounding
- **출처:** [CLIP](https://arxiv.org/abs/2103.00020); [Flamingo](https://arxiv.org/abs/2204.14198)

### Vision Transformer (ViT) - 비전 트랜스포머 (ViT)
- **분류:** 멀티모달 시스템
- **실제 의미:** 이미지를 위치 정보가 포함된 패치 임베딩의 시퀀스로 표현하고, 해당 시퀀스를 트랜스포머 인코더 블록으로 처리하는 시각 아키텍처입니다.
- **왜 중요한가:** 시각 데이터에 시퀀스 모델 인터페이스를 제공하지만, 성능과 연산량은 패치 크기, 해상도, 사전 학습, 귀납적 편향에 의존합니다.
- **실무에서는:** 패치 처리와 정규화를 학습 시와 일관되게 유지하고, 새로운 해상도에서의 위치 임베딩 동작을 고려하며, 대상 데이터셋에서 적절한 시각적 기준선과 비교합니다.
- **흔한 혼동:** ViT는 아키텍처 계열이며, 이미지를 입력으로 받는 모든 트랜스포머가 ViT는 아니며, ViT의 패치는 본질적으로 의미 있는 객체가 아닙니다.
- **배울 곳:** [Vision Transformers](../phases/04-computer-vision/14-vision-transformers/)
- **관련 용어:** Transformer, Patch Embedding, Self-Attention, Encoder
- **출처:** [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

### Visual Grounding - 시각적 접지
- **분류:** 멀티모달 시스템
- **실제 의미:** 이미지나 비디오의 공간적 증거(예: 영역, 객체, 마스크, 추적된 엔티티)와 언어 표현을 연결하는 것.
- **왜 중요한가:** 유창한 시각적 답변은 근거가 없을 수 있으며, 접지(grounding)는 주장된 참조 대상을 검증 가능하게 만들고 영역 수준 평가를 가능하게 합니다.
- **실무에서는:** 답변과 함께 박스, 마스크, 또는 시간적 세그먼트를 요구하고, 모호하거나 존재하지 않는 참조 대상을 테스트하며, 언어적 정확성과 위치 파악을 별도로 점수화합니다.
- **흔한 혼동:** 시각적 접지는 참조된 증거가 어디에 있는지를 식별합니다. 일반적인 이미지 캡셔닝은 각 주장을 위치화하지 않고 장면을 설명할 수 있습니다.
- **배울 곳:** [Cross-Attention Fusion](../phases/19-capstone-projects/61-cross-attention-fusion/)
- **관련 용어:** Grounding, Vision-Language Model (VLM), Attention, Evaluation (Eval)
- **출처:** [MDETR](https://arxiv.org/abs/2104.12763)

### Vocabulary - 어휘
- **분류:** 데이터 및 표현
- **실제 의미:** 토큰 식별자와 토크나이저가 생성할 수 있는 단위(일반 토큰, 바이트 수준 토큰, 특수 제어 토큰 포함) 사이의 유한한 매핑.
- **왜 중요한가:** 어휘 설계는 시퀀스 길이, 다국어 커버리지, 코드 표현, 임베딩 크기, 토크나이저와 모델 가중치 간의 호환성에 영향을 미칩니다.
- **실무에서는:** 어휘와 특수 토큰 할당을 모델과 함께 버전 관리하고, 인코딩-디코딩 왕복 테스트를 수행하며, 토큰 이름이 단순히 유사하다는 이유로 토크나이저를 교체하지 마십시오.
- **흔한 혼동:** 모델 어휘는 인간 단어의 사전이 아닙니다. 많은 항목은 조각, 바이트, 공백 패턴 또는 제어 기호입니다.
- **관련 용어:** Tokenization, Byte Pair Encoding (BPE), Token, Embedding
- **출처:** [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

## W

### Warmup - 워밍업
- **분류:** 수학 및 학습
- **실제 의미:** 학습률이 더 작은 값에서 메인 스케줄의 목표 값으로 상승하는 초기 학습 단계입니다.
- **왜 중요한가:** 초기 기울기와 옵티마이저 통계는 불안정할 수 있으며, 특히 대규모 배치나 트랜스포머 학습에서는 갑작스러운 전체 크기 업데이트가 최적화를 손상할 수 있습니다.
- **실무에서는:** 스텝이나 처리된 토큰 수로 워밍업을 정의하고, 실현된 곡선을 기록하며, 배치, 옵티마이저 및 총 학습 예산을 가시적으로 유지하면서 이를 조정하십시오.
- **흔한 혼동:** 워밍업은 모든 모델에 필수적이지 않으며, 적합하지 않은 학습률을 안전하게 만들지 않습니다.
- **관련 용어:** Learning Rate Schedule, Learning Rate, Batch Size, AdamW
- **출처:** [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)

### Weight - 가중치
- **분류:** 수학 및 학습
- **흔히 하는 말:** 모델 내부의 학습된 숫자입니다.
- **실제 의미:** 모델 변환에서의 학습 가능한 계수입니다. 가중치는 일반적으로 텐서로 조직되며, 최적화는 학습 목적 함수를 줄이기 위해 이를 조정합니다.
- **흔한 혼동:** 모든 매개변수가 가중치라고 불리지는 않습니다. 편향, 임베딩 및 정규화 스케일도 매개변수입니다.
- **관련 용어:** Parameter, Tensor, Optimizer

### Weight Decay - 가중치 감쇠
- **분류:** 수학 및 학습
- **흔히 하는 말:** 최적화 중에 가중치를 축소하는 정규화입니다.
- **실제 의미:** 학습 동안 선택된 매개변수의 크기를 줄이는 업데이트 규칙으로, 종종 기울기 업데이트와 분리된 축소 계수로 가중치를 곱하여 수행합니다.
- **왜 중요한가:** 일반화 성능을 개선할 수 있지만, 유용한 계수와 제외된 파라미터 그룹은 모델, 옵티마이저, 스케줄 및 데이터에 따라 달라집니다.
- **흔한 혼동:** 분할 가중치 감소(decoupled weight decay)는 일부 단순한 옵티마이저에서는 L2 손실 페널티와 동일하지만, Adam과 같은 적응형 옵티마이저에서는 일반적으로 동일하지 않습니다.
- **관련 용어:** AdamW, Overfitting, Optimizer

### Worktree - 워크트리
- **분류:** AI 네이티브 개발
- **실제 의미:** Git에서 저장소 및 브랜치나 커밋에 연결된 작업 디렉토리로, 객체 저장소는 공유하지만 체크아웃된 파일과 인덱스는 각각 독립적으로 가집니다.
- **왜 중요한가:** 분리된 worktree를 사용하면 한 체크아웃을 계속 전환하거나 덮어쓰지 않고도 사람과 에이전트가 동시에 작업할 수 있습니다.
- **실무에서는:** 각 코딩 에이전트에게 이름이 지정된 기능 브랜치와 정확한 worktree 경로를 부여한 후, 일반적인 Git 히스토리를 통해 패치를 검토하고 통합합니다.
- **흔한 혼동:** worktree는 체크아웃된 파일을 격리할 뿐, 머신 상의 모든 프로세스, 포트, 캐시, 데이터베이스 또는 비밀을 격리하지는 않습니다.
- **배울 곳:** [Workbench for Real Repositories](../phases/14-agent-engineering/41-workbench-for-real-repos/)
- **출처:** [git-worktree documentation](https://git-scm.com/docs/git-worktree)
- **관련 용어:** Coding Agent, Patch, Scope Contract, Handoff

## Z

### Zero-Shot - 제로샷
- **분류:** 프롬팅 및 컨텍스트
- **흔히 하는 말:** 현재 프롬프트에 예시 없이 작업을 요청하는 것.
- **실제 의미:** 즉각적인 입력에 작업별 시연(demonstration)을 포함하지 않고, 지시문이나 작업 구성(framing)만으로 작업을 수행하는 것.
- **흔한 혼동:** Zero-shot은 모델이 관련 사전 학습, 지시문 튜닝, 도구, 검색된 컨텍스트가 전혀 없음을 의미하지 않습니다.
- **관련 용어:** Few-Shot, Prompt Engineering, Transfer Learning

### Zero Trust - 제로 트러스트
- **분류:** 보안 및 거버넌스
- **실제 의미:** 네트워크 위치나 자산 소유권에서 암묵적인 신뢰를 부여하지 않고, 대신 각 접근 요청을 신원, 디바이스, 리소스, 정책 및 현재 컨텍스트에 대해 평가하는 보안 모델입니다.
- **왜 중요한가:** AI 도구와 에이전트는 로컬 파일, 클라우드 서비스, 모델, 외부 콘텐츠에 걸쳐 존재하므로, 신뢰할 수 있는 내부 네트워크는 권한 부여의 근거로서 너무 광범위합니다.
- **실무에서는:** 모든 행위자와 워크로드를 인증하고, 각 리소스 작업에 대해 권한을 부여하며, 단기 유효 자격 증명을 발급하고, 접근을 세분화하며, 정책 관련 신호를 지속적으로 기록하고 재평가합니다.
- **흔한 혼동:** 제로 트러스트는 아무것도 신뢰하지 않거나 모든 자동화를 차단하는 것을 의미하지 않습니다. 신뢰 결정을 명시적이고, 범위가 한정되며, 지속적으로 검증 가능하게 만드는 것을 의미합니다.
- **배울 곳:** [Security, Secrets, and Audit](../phases/17-infrastructure-and-production/25-security-secrets-audit/)
- **관련 용어:** Least Privilege, Trust Boundary, Approval Gate, Audit Log
- **출처:** [NIST SP 800-207](https://csrc.nist.gov/pubs/sp/800/207/final)
