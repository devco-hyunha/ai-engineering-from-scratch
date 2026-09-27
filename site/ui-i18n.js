// site/ui-i18n.js: UI localization dictionary for AI Engineering from Scratch
(function (root) {
  'use strict';

  var UI_I18N = {
    ko: {
      actionTitle: '이 레슨 실습하기',
      actionIntro: '학습 내용을 실제 증거로 전환하는 실습 루프입니다. 각 체크포인트는 퀴즈 통과 여부와 별개로 자유롭게 체크할 수 있습니다.',
      checkpointsStatus: '{done} / 5 체크포인트 완료',
      quizPassed: '퀴즈 통과',
      quizNotPassed: '퀴즈 미통과',
      stepReadTitle: '읽기',
      stepReadCopy: '핵심 주장과 제약 조건을 이해합니다.',
      stepBuildTitle: '구현',
      stepBuildCopy: '레슨의 결과물이나 코드를 직접 작성/수정합니다.',
      stepRunTitle: '실행',
      stepRunCopy: '관련 프로그램이나 실습 코드를 실행합니다.',
      stepProveTitle: '검증',
      stepProveCopy: '주장을 뒷받침하는 실행 결과(증거)를 확보합니다.',
      stepContinueTitle: '완료',
      stepContinueCopy: '준비가 되었을 때 레슨을 완료로 표시합니다.',
      runRepoLabel: '저장소에서 실행',
      checkingRunnable: '실행 가능한 엔트리포인트 파일 확인 중',
      checkingNote: '레슨 코드 디렉터리에서 독립 실행 가능한 엔트리포인트를 확인하고 있습니다.',
      copyCmd: '명령어 복사',
      copied: '복사됨',
      prevLesson: '이전 레슨',
      nextLesson: '다음 레슨',
      locked: '잠김',
      testUnderstanding: '이해도 확인 (퀴즈)',
      didYouGetIt: '개념을 잘 이해하셨나요?',
      questionOf: '문제 {i} / {n}',
      quizScorePrompt: '모든 문제를 풀면 점수가 표시됩니다',
      quizPassedBadge: '통과',
      quizRetryBadge: '다시 시도',
      artifactDeliverable: '이 레슨은 별도 산출물 파일이 없습니다. 레슨 본문 자체가 산출물입니다.',
      openFile: '파일 열기',
      copyPrompt: '프롬프트 복사',
      localProgress: '로컬 진행도',
      runNoteRepoRoot: 'README.md가 있는 저장소 루트 디렉터리에서 실행하세요.',
      runNoteNoFile: '예제 코드를 참고용으로 활용하세요. 적절한 예제나 연습 코드를 실행한 후 "실행"을 표시하세요. 코드 블록이 자동으로 실행 가능한 프로그램인 것은 아닙니다.',
      noStandaloneFile: '독립 실행 가능한 메인 파일이 감지되지 않았습니다',
      preQuizTitle: '레슨 시작 전 점검',
      midQuizTitle: '레슨 중간 점검',
      postQuizTitle: '레슨 마무리 퀴즈',
      quizTitleDefault: '퀴즈',
      retryCheck: '이 점검 다시 시도',
      quizScoreCorrect: '{correct}/{total} 정답',
      quizScoreAnswered: '{answered}/{total} 완료. 설명을 검토한 후 준비가 되면 다시 시도하세요.',
      quizResetStatus: '점검이 초기화되었습니다. 각 문제의 새 정답을 선택하세요.',
      onThisPage: '목차',
      learningObjectivesTitle: '학습 목표',
      pathMinutes: '코스 {min}분',
      lessonMinutes: '레슨 {min}분',
      groupPrefix: '그룹 {group}',
      focusedPathContext: '집중 학습 경로 정보',
      captureCheckpointEvidence: '체크포인트 실행 증거를 확보하세요:',
      checkpointLabel: '체크포인트:',
      quickStartLabel: '빠른 시작:',
      quickStartDefaultGoal: '첫 번째 실행 가능한 체크포인트를 완료하세요.',
      quickStartMinutes: '첫 실행에는 약 {min}분이 소요됩니다.',
      quickStartCommandLabel: '빠른 시작 명령어:',
      quickStartEvidenceLabel: '빠른 시작 확인 결과(증거):',
      diagramTitle: '다이어그램',
      diagramRendering: '다이어그램 렌더링 중...',
      diagramFailed: '다이어그램을 렌더링할 수 없습니다.',
      diagramExpand: '크게 보기',
      outputsPanelTitle: '레슨 산출물 (배포물)',
      outputsPanelSubtitle: '지금 바로 실무에 활용할 수 있는 프롬프트, 스킬, 아티팩트',
      outputsLoading: '산출물 불러오는 중...',
      outputsFallback: '이 레슨에는 별도의 산출물(아티팩트) 파일이 없습니다.',
      codeFallback: '이 레슨에는 별도의 실행 파일이 없습니다. 레슨 본문의 실습 예제를 참고하세요.',
      codeFallbackFetchFailed: '이 레슨에는 별도의 실행 파일이 감지되지 않았습니다.',
      viewLessonSource: '레슨 소스 보기',
      codePanelTitle: '코드 실행하기',
      codePanelSubtitle: '로컬 복제본이 필요합니다. 복사한 명령어는 README.md 및 phases/가 있는 저장소 루트 디렉터리에서 실행하세요.',
      codeLoading: '코드 파일 불러오는 중...',
      installSkillLabel: '설치',
      copyInstallCmd: '설치 명령어 복사',
      installPromptHint: 'Claude, Cursor, Codex, OpenClaw, Hermes 등 프롬프트를 인식하는 모든 AI 에이전트에 붙여넣어 사용하세요.',
      openSkillMd: 'SKILL.md 열기',
      viewKoreanVersion: '한국어 번역본 보기',
      openOriginalFile: '원본 파일 열기',
      quizDeeperIntro: '더 깊이 있는 퀴즈를 풀어보고 싶으신가요?',
      quizDeeperCodex: 'Codex에서는 <code>check-understanding {phase}</code>를 입력하거나 <code>/skills</code>에서 선택하세요.',
      quizDeeperClaude: 'Claude Code에서는 <code>/check-understanding {phase}</code>를 사용하세요.',
      quizDeeperOther: '다른 호환 에이전트에서는 다음과 같이 입력하세요: <code>Use check-understanding to quiz me on Phase {phase}.</code>',
      quizScorePerfect: '완벽합니다!',
      quizScoreGreat: '훌륭합니다!',
      quizScoreStudy: '조금 더 복습해 보세요!',
      quizAnswerCorrect: '정답입니다. ',
      quizAnswerIncorrect: '아쉽네요. 다시 확인해 보세요. ',
      learningPathTitle: '학습 경로',
      lessonOfTotal: '{total}개 중 {current}번째 레슨',
      earlierLessons: '{n}개 이전',
      laterLessons: '{n}개 이후',
      phaseCompletedProgress: '이 Phase의 {total}개 레슨 중 {done}개를 완료했습니다.',
      readyForNextPhase: '다음 단계인 Phase {phase}: {name}를 학습할 준비가 되었습니다!',

      // Homepage (index.html) translations
      home: {
        navContents: '목차',
        navBooks: '도서',
        navCatalog: '카탈로그',
        navRoadmap: '로드맵',
        navGlossary: '용어집',
        navAbout: '소개',
        navCertifications: '자격증',
        navReport: '오류 보고',
        openSourceMit: '오픈소스 · MIT',
        taglineSuffix: '단 하나의 프레임워크도 임포트하기 전에 모든 알고리즘을 순수 수학으로부터 밑바닥부터 구현합니다.',
        attribution: 'Rohit Ghumare 및 기여자들이 관리합니다. 내 컴퓨터에서 직접 실행하세요.',
        btnStartCourse: '코스 시작하기',
        btnExplorePaths: '학습 경로 탐색',
        btnStarGitHub: 'GitHub 스타',
        btnFollow: '@rohitg00 팔로우',
        terminalLearn: '터미널에서 학습하기',
        terminalCopy: '복사',
        terminalCopied: '복사됨!',
        terminalCaption: 'AI 에이전트가 나만의 튜터가 됩니다: 실력 진단 퀴즈, 맞춤형 학습 경로, 터미널 대화형 레슨 진행.',
        figPrevious: '이전',
        figNext: '다음',
        plateCaption1: 'Plate 1 of 3. 이 다이어그램의 모든 계층은 직접 손으로 구현하는 레슨입니다.',
        plateCaption2: 'Plate 2 of 3. AI 튜터가 터미널에서 가르치고, 코드를 작성하면 테스트를 실행합니다.',
        plateCaption3: 'Plate 3 of 3. 진짜를 배우기 때문에 러닝 커브가 가파릅니다. 마법은 없습니다.',
        learnersEyebrow: '다음 기관의 엔지니어와 학생들이 함께 읽고 있습니다',
        learnersQuote: '“AI Engineering from Scratch 저장소에 푹 빠졌습니다.” <span class="learners-quote-attr">- Google AI 엔지니어</span>',
        prefaceEyebrow: '학습 방식',
        prefaceP1: '대부분의 AI 학습 자료는 파편화되어 있습니다. 논문 하나, 파인튜닝 글 하나, 화려한 에이전트 데모가 제각각 흩어져 있어 유기적으로 연결되지 않습니다. 챗봇을 배포하더라도 loss 곡선을 설명하지 못하고, 에이전트에 함수를 연결하면서도 모델 내부에서 attention 메커니즘이 어떤 역할을 하는지 설명하지 못합니다.',
        prefaceP2Prefix: '본 커리큘럼은 척추 역할을 합니다.',
        prefaceP2Suffix: '4개 언어: Python, TypeScript, Rust, Julia. 한쪽 끝에는 선형대수학이, 다른 쪽 끝에는 자율 스웜이 있습니다. 역전파, 토크나이저, 어텐션, 에이전트 루프까지 모든 알고리즘을 순수 수학으로부터 먼저 구축합니다. PyTorch를 임포트할 때쯤이면 내부에서 일어나는 일을 이미 완벽히 이해하게 됩니다.',
        prefaceP3: '모든 레슨은 동일한 루프로 진행됩니다: 문제를 읽고, 수학을 유도하고, 코드를 작성하고, 테스트를 실행하고, 결과물(아티팩트)을 확보합니다. 5분짜리 영상도, 복사-붙여넣기식 배포도, 피상적인 안내도 없습니다. 무료 오픈소스이며, 내 컴퓨터에서 직접 실행되도록 제작되었습니다.',
        coursePathsTitle: '목표에 맞는 학습 경로 선택',
        coursePathsHeaderCopy: 'AI 엔지니어링은 모델 코드보다 훨씬 넓은 영역을 다룹니다. 4대 핵심 학습 경로 중 하나를 선택하여 브라우저나 GitHub에서 동일한 소스, 랩, 테스트, 아티팩트로 학습하세요.',
        viewLearningPaths: '학습 경로 전체 보기',
        browseCareerRoutes: '커리어 루트 탐색',
        learningPathsRoot: 'AI 엔지니어링',
        learningPathsSub: '4대 연계 핵심 역량',
        pathAppLabel: 'AI 애플리케이션 구축 및 배포',
        pathSoftwareLabel: '소프트웨어 공학 기초',
        pathAgentsLabel: '에이전트 보조 엔지니어링',
        pathShapingLabel: '제품 판단 및 딜리버리',
        statBlockTitle: '현재 학습 진행도',
        statFinishedLessons: '완료한 레슨',
        statPhases: '단계 (Phase)',
        statLanguages: '지원 언어',
        statGlossaryTerms: '용어집 단어',
        tocTitleSuffix: '전체 커리큘럼',
        tocSubtitle: '단계를 탭하여 상세 레슨을 확인하세요. 각 레슨은 수학, 코드, 테스트가 모두 검증될 때 완성됩니다.',
        legendComplete: '완료',
        legendInProgress: '진행 중',
        legendPlanned: '예정',
        modalFooterNote: '진행 상황은 브라우저에만 로컬로 저장됩니다',
        modalResetBtn: '진행도 초기화',
        modalResetConfirm: '모든 로컬 진행 상황(퀴즈 답변 및 완료한 레슨)을 초기화하시겠습니까? 이 작업은 되돌릴 수 없습니다.',
        openLesson: '레슨 열기',
        reviewLesson: '복습하기',
        comingSoon: '준비 중',
        done: '완료',
        markDone: '완료로 표시',
        markNotDone: '미완료로 표시',
        booksTitle: '도서 에디션 · 전 6권',
        booksSubtitle: '커리큘럼을 책으로 엮었습니다. 동일한 레슨으로 빌드된 EPUB 및 PDF가 모든 GitHub 릴리스에 제공됩니다. 웹사이트는 상호작용 가능한 다이어그램, 퀴즈, 코드를 담은 살아있는 에디션으로 유지됩니다.',
        booksNote: '링크는 최신 <a href="https://github.com/rohitg00/ai-engineering-from-scratch/releases" target="_blank" rel="noopener">GitHub 릴리스</a>로 연결됩니다 · 릴리스마다 레슨에서 CI로 자동 빌드됨 · <a href="https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/book/README.md" target="_blank" rel="noopener">제작 과정 안내</a>',
        colophonEyebrow: '소개 및 판권 (Colophon)',
        colophonText: '모든 커리큘럼은 GitHub에 공개되어 있습니다. 클론하고 포크하여 나만의 속도로 학습하세요. 유료 결제나 가입이 필요 없습니다. 모든 레슨은 개념에 가장 적합한 Python, TypeScript, Rust, Julia 중 하나로 작성된 실행 가능한 코드를 제공합니다.',
        footerCopy: '© 2026 · 오픈소스 · 영구 무료'
      },
      routes: {
        recFirst: '추천 시작점',
        coreDomain: '핵심 도메인',
        focusedPath: '집중 경로',
        practiceEvidence: '실무 증거 기반 학습',
        newToAiTitle: 'AI 엔지니어링 입문',
        newToAiDesc: '전문 과정을 선택하기 전에 개발 환경을 구축하고 저장소를 실행하며 레슨 학습 방식을 익힙니다.',
        appTitle: 'AI 애플리케이션 구축 및 배포',
        appDesc: '프롬프트, 구조화된 출력, 임베딩, 검색(RAG)부터 평가, 서빙, 관측 가능성 및 안전한 배포까지 다룹니다.',
        softwareTitle: '소프트웨어 공학 기초',
        softwareDesc: 'AI 시스템이 의존하는 저장소, 개발 환경, 인터페이스, 디버깅, 검증, 보안, 배포 및 운영 기반을 구축합니다.',
        agentsTitle: '에이전트 보조 엔지니어링',
        agentsDesc: '태스크 정의, 저장소 증거 기반 계획 수립, 에이전트 루프 및 하네스 설계, 위임 격리, 결과 검증 및 피드백 보존을 학습합니다.',
        shapingTitle: '제품 판단 및 딜리버리',
        shapingDesc: '관찰된 작업을 가치 있는 결과물, 가정, 테스트 가능한 최소 단위, 실행 가능한 명세, 측정 계획, 단계적 배포로 전환합니다.',
        mcpTitle: '모델 컨텍스트 프로토콜 (MCP)',
        mcpDesc: '와이어 엔벨로프부터 릴리스 게이트까지 상태 비저장(stateless) MCP 시스템을 구축, 보호, 검증 및 운영합니다.',
        skillsTitle: '에이전트 스킬 (Agent Skills)',
        skillsDesc: '실제 에이전트 호스트에서 포터블 스킬을 구축, 호출, 라우팅, 보안 검증, 평가, 패키징 및 검증합니다.',
        certTitle: '인증 자격증 준비',
        certDesc: '인증 경로를 선택하고, 실습 랩을 완료하며, 학습자 산출물을 관리하고, 오리지널 진단 평가를 활용합니다.',
        openLesson: '레슨 열기',
        ghSource: 'GitHub 소스',
        startPath: '경로 시작',
        ghPath: 'GitHub 경로',
        explorePaths: '경로 탐색',
        ghTutor: 'GitHub 튜터'
      },
      phaseNames: {
        0: '개발 환경 및 툴링',
        1: '수학 기초',
        2: '머신러닝 기본기',
        3: '밑바닥부터 구현하는 딥러닝',
        4: '컴퓨터 비전',
        5: '자연어 처리: 기초부터 고급까지',
        6: '음성 및 오디오',
        7: '트랜스포머 집중 탐구',
        8: '생성형 AI',
        9: '강화학습',
        10: 'LLM 밑바닥부터 만들기',
        11: 'LLM 엔지니어링',
        12: '멀티모달 AI',
        13: '도구 및 프로토콜',
        14: '에이전트 엔지니어링',
        15: '자율 시스템',
        16: '다중 에이전트 및 스웜',
        17: '인프라 및 프로덕션',
        18: '윤리, 안전 및 정렬',
        19: '캡스톤 프로젝트'
      },
      phaseDescs: {
        0: '이후의 모든 과정을 시작하기 위한 개발 환경을 완벽하게 구축합니다.',
        1: '코드를 통해 모든 AI 알고리즘의 직관과 수학적 원리를 이해합니다.',
        2: '대부분의 프로덕션 AI 시스템의 척추가 되는 전통적 머신러닝을 구현합니다.',
        3: '기초 원리부터 신경망을 구축합니다. 직접 만들기 전까지는 프레임워크를 쓰지 않습니다.',
        4: '픽셀에서 이해로 — 이미지, 비디오, 3D, VLM 및 세계 모델까지.',
        5: '언어는 지능과 소통하는 인터페이스입니다.',
        6: '듣고, 이해하고, 말하기.',
        7: 'AI의 모든 것을 바꾼 아키텍처를 밑바닥부터 파헤칩니다.',
        8: '이미지, 비디오, 오디오, 3D 등을 생성하는 생성 모델의 모든 것.',
        9: 'RLHF와 게임 플레이 AI의 핵심이 되는 강화학습 기초.',
        10: '대규모 언어 모델을 직접 만들고, 학습시키고, 깊이 이해합니다.',
        11: 'LLM을 실제 프로덕션 환경에서 동작하도록 엔지니어링합니다.',
        12: 'ViT 패치부터 컴퓨터 사용 에이전트까지 여러 모달리티에 걸쳐 보고, 듣고, 읽고 추론합니다.',
        13: 'AI와 실제 현실 세계를 연결하는 인터페이스와 프로토콜(MCP).',
        14: '에이전트를 원리부터 구축하고, 코딩 에이전트를 안정적으로 활용하며, 구현 전 작업을 설계합니다.',
        15: '장기 실행 에이전트, 자기 개선 메커니즘 및 2026 안전 스택.',
        16: '에이전트 간의 조율, 창발성, 그리고 집단 지성.',
        17: 'AI 시스템을 실제 프로덕션 환경에 배포하고 운영합니다.',
        18: '인류에게 도움이 되는 안전한 AI를 구축합니다. 선택이 아닌 필수입니다.',
        19: '17개의 엔드투엔드 프로덕트 + 9개의 심층 빌드 트랙. 프로젝트당 20~40시간 소요.'
      },
      roadmap: {
        navContents: '목차',
        navCatalog: '카탈로그',
        navRoadmap: '로드맵',
        navGlossary: '용어집',
        navAbout: '소개',
        heroEyebrow: '커리큘럼 탐색',
        heroTitle: '기초 원리부터 프로덕션 AI까지.',
        heroLede: '20개 연계 단계(Phase)가 하나의 상하 의존성 그래프를 형성합니다. 아래로 경로를 따라가며 각 단계의 선수지식을 확인하고, 로컬 진행도에 맞춰 바로 학습을 이어가세요.',
        statPhases: '단계 (Phases)',
        statLessons: '레슨 (Lessons)',
        statProgress: '내 진행도',
        statNext: '추천 다음 단계',
        zonesTitle: '4대 그래프 권역',
        zonesSub: '해당 권역으로 이동',
        zonePrefix: '권역',
        stageNames: {
          'foundations': '기초 원리',
          'model-disciplines': '모델 전문 분야',
          'engineering-systems': '엔지니어링 시스템',
          'capstone-proof': '캡스톤 실증'
        },
        mapTitle: '대화형 학습 지도',
        mapGuide: '노드를 선택하면 연결된 경로가 강조 표시됩니다. 선택한 노드나 빈 공간을 클릭하면 전체 그래프로 돌아갑니다.',
        keyboardHelp: '방향키로 단계 노드 사이를 이동할 수 있습니다. Enter나 Space를 눌러 단계를 선택하고, Esc 키를 눌러 경로 선택을 해제하세요.',
        findPhaseLabel: '단계 찾기',
        zoomOut: '축소',
        zoomIn: '확대',
        backToFull: '전체 그래프로 복귀',
        legendPrereqs: '선수 조건 경로',
        legendUnlocks: '이후 해금 단계',
        scrollHint: '아래로 스크롤하여 경로 탐색 · 좌우로 드래그하여 분기 탐색',
        stateComplete: '완료',
        stateInProgress: '진행 중',
        stateReady: '시작 가능',
        stateUpcoming: '예정',
        inspectorEmptyEyebrow: '경로 탐색기',
        inspectorEmptyTitle: '단계를 선택하세요',
        inspectorEmptyCopy: '노드를 선택하면 해당 단계로 이어지는 정확한 경로, 이 단계가 해금하는 모든 후속 단계, 그리고 로컬 진행도에 맞춘 최적의 다음 레슨을 확인할 수 있습니다.',
        recNextLabel: '추천 다음 단계',
        inspectorYourProgress: '내 진행도',
        inspectorAllPrereqs: '전체 선수 단계',
        inspectorPhasesUnlocked: '해금되는 단계',
        inspectorDirectPrereqs: '직접적인 선수 단계',
        inspectorDirectUnlocks: '직후 해금 단계',
        startPointMsg: '이곳이 커리큘럼의 출발점입니다.',
        finalDestMsg: '최종 목적지 단계입니다.',
        btnReviewPhase: '단계 복습하기',
        btnContinuePhase: '단계 이어하기',
        btnStartPhase: '단계 시작하기',
        btnViewGithub: 'GitHub에서 단계 소스 보기',
        dataLoadError: '로드맵 데이터를 불러올 수 없습니다. 사이트를 다시 빌드한 후 페이지를 새로고침하세요.',
        completeUpper: '완료'
      },
      learningPaths: {
        pageTitle: 'AI 엔지니어링 학습 경로 - AI Engineering from Scratch',
        skipLink: '본문으로 건너뛰기',
        navContents: '목차',
        navCatalog: '카탈로그',
        navRoadmap: '로드맵',
        navGlossary: '용어집',
        navAbout: '소개',
        heroEyebrow: '4대 핵심 경로 · 6개 커리어 루트',
        heroTitle: 'AI 엔지니어링 학습 경로',
        heroLede: '깊이 있는 전문성을 쌓기 위한 핵심 도메인을 선택하거나, 실제 현업에서 수행하고자 하는 업무를 중심으로 구성된 커리어 루트를 선택하세요.',
        exploreByKnowledge: '지식 체계별 탐색',
        browseFourPaths: '4대 핵심 경로 보기',
        exploreByOutcome: '목표 결과별 탐색',
        chooseCareerRoute: '커리어 루트 선택',
        overviewTitle: '4대 핵심 경로. 하나의 학문.',
        overviewCopy: '각 도메인은 체계적인 레슨 시퀀스와 이를 통해 함양되는 역량을 제공합니다. 모든 역량은 가장 밀접한 실습 레슨으로 연결됩니다.',
        compareSixRoutes: '6개 커리어 루트 비교하기',
        rootTitle: 'AI 엔지니어링',
        rootSubtitle: '시스템, 실무 작업, 그리고 구현을 학습합니다',
        domain1Title: 'AI 애플리케이션 구축 및 배포',
        domain1Sub: '인터페이스 · 그라운딩 · 평가 · 프로덕션',
        domain2Title: '소프트웨어 공학 기초',
        domain2Sub: '풀스택 · 데이터 · 아키텍처 · 신뢰성 · 확장성',
        domain3Title: '에이전트 보조 엔지니어링',
        domain3Sub: '문제정의 · 계획 · 위임 · 검증 · 개선',
        domain4Title: '제품 판단 및 딜리버리',
        domain4Sub: '결과 · 증거 · 위험 · 지표 · 피드백',
        careerEyebrow: '커리어 방향 · 공통 커리큘럼',
        careerTitle: '직함이 아닌 실제 수행하는 일로 선택하세요.',
        careerCopy: '직함은 팀마다 다릅니다. 내가 직접 책임지고 해결하고 싶은 문제에서 출발하세요. 전문 레슨을 선택하기 전에 담당 업무, 기준선, 포트폴리오 증거 및 갭을 확인하세요.',
        readThisFirst: '시작 전 필독',
        truthNoteCopy: '이 커리어 루트들은 공통 기초 과정을 이수한 후 적용하는 전문 특화 오버레이입니다. 과정을 완료하면 실무 수행 증거가 축적되지만, 취업을 직접 보장하는 것은 아닙니다. 표시된 시간은 가이드된 레슨 학습 시간 기준이며 기초 다지기, 개인 프로젝트 및 실무 경험 시간은 별도입니다.',
        step1Title: '소프트웨어 엔지니어링 기초',
        step1Desc: '풀스택 경계, 데이터, 아키텍처, 신뢰성, 보안 및 프로덕션 운영.',
        step2Title: 'AI 애플리케이션 기초',
        step2Desc: '모델 인터페이스, 그라운딩, 평가, 프로덕션 동작 및 운영.',
        step3Title: '전문 분야 실무 실습',
        step3Desc: '직무 패밀리를 선택하고, 기준선 갭을 메우며, 직무 형태에 맞춘 실무 증거를 확보합니다.',
        chooserEyebrow: '결정 질문',
        chooserTitle: '매주 반복해서 수행하고 싶은 일은 무엇인가요?',
        prompt1Sub: '사용자 곁에서 하나의 워크플로우를 프로덕션까지 완성하시겠습니까?',
        prompt1Title: '고객 AI 배포',
        prompt2Sub: '개발자를 위해 API, 예제, 기술 학습 경험을 더 쉽게 만들고 싶으신가요?',
        prompt2Title: '개발자 경험 및 교육',
        prompt3Sub: 'AI 기능 뒤에 있는 데이터, 검색 및 품질 시스템을 구축하시겠습니까?',
        prompt3Title: 'AI 데이터 시스템',
        prompt4Sub: '도구 사용 루프, 메모리, 오케스트레이션 및 런타임 제어를 엔지니어링하시겠습니까?',
        prompt4Title: '에이전트 시스템 엔지니어링',
        prompt5Sub: '인터페이스부터 프로덕션까지 모델 기반 제품 동작을 출시하시겠습니까?',
        prompt5Title: 'LLM 제품 엔지니어링',
        prompt6Sub: '실패를 측정하고 릴리스를 게이팅하며 AI 시스템을 안정적으로 운영하시겠습니까?',
        prompt6Title: 'AI 평가 및 신뢰성',
        openGuide: '가이드 열기',
        closeGuide: '가이드 닫기',
        workFamily: '직무 패밀리',
        guidedRouteLabel: '가이드 루트',
        baselineLabel: '요구 기준선',
        whatYouOwn: '담당하게 되는 주요 업무',
        fitAndBoundary: '적합성 및 직무 경계',
        portfolioProof: '포트폴리오 증명 (증거물)',
        coverageAndGaps: '코스 범위 및 추가로 채워야 할 점',
        goodFitIf: '이런 분께 적합합니다:',
        boundaryLabel: '직무 경계:',
        stillEarnedLabel: '코스 밖에서 쌓아야 할 것:',
        studySpecialist: '전문 레슨 학습하기',
        openFullPath: '전체 경로 열기',
        openLesson: '레슨 열기',
        openRepLesson: '대표 레슨 열기',
        backToFourDomains: '4대 도메인으로 돌아가기',
        compareFourDomains: '4대 도메인 비교하기',
        followConnected: '모든 연계 역량을 순서대로 학습하세요.',
        unsureFooter: '어떤 방향이 맞는지 고민되시나요? 담당 업무와 결과물 증거를 비교해 보고, 전문화하기 전에 공통 기초를 먼저 다지세요.',
        footerCopy: 'AI Engineering from Scratch · 오픈소스 · 영구 무료.',
        footerHome: '홈',
        footerRoadmap: '커리큘럼 로드맵',
        footerCatalog: '카탈로그',
        careers: {
          'forward-deployed-ai-engineer': {
            title: '고객 AI 배포',
            aliases: 'Forward-Deployed AI Engineer · Field AI Engineer · AI Solutions Engineer',
            mission: '실제 고객 워크플로우를 작고 측정 가능한 AI 시스템으로 바꾼 뒤, 롤아웃 현장에서 무엇이 깨지는지 가까이에서 학습합니다.',
            route: '전문 레슨 12개 · 865분',
            baseline: '소프트웨어 딜리버리 + AI 애플리케이션 기초',
            own: [
              '워크플로우, 사용자, 예외, 숨은 핸드오프를 관찰합니다.',
              '요청을 가장 작은 유용한 엔드투엔드 슬라이스로 줄입니다.',
              '그라운딩, 평가, 프로덕션 제어를 통합합니다.',
              '측정된 파일럿을 운영하고 피드백을 다음 시스템 변경으로 전환합니다.'
            ],
            fit: '모호한 사용자 문제, 빠른 기술 반복, 출시 후 공동 소유권을 선호합니다.',
            boundary: '세일즈 엔지니어링이나 일반 컨설팅이 아닙니다. 증명은 직접 운영할 수 있는 동작하는 측정 시스템입니다.',
            portfolio: '워크플로우 도сье, 그라운딩된 프로토타입, 평가 세트, 파일럿 계획을 하나의 증거 번들로 출시합니다.',
            evidence: ['명시된 가정과 가장 위험한 테스트', '측정된 작업 품질과 실패 사례', '롤아웃, 롤백, 피드백 소유권'],
            coverage: '이 루트는 디스커버리, 리스크, RAG, 평가, 프로덕션, 지표, 롤아웃, 피드백을 다룹니다.',
            stillEarned: '고객 도메인 전문성, 이해관계자 신뢰, 조달 제약, 라이브 프로덕션 압박 하의 소유권.',
            footer: '공통 기초 점검 후 시작하세요. 모든 레슨 산출물을 최종 번들 증거로 남기세요.'
          },
          'ai-developer-relations-engineer': {
            title: '개발자 경험 및 교육',
            aliases: 'AI Developer Relations Engineer · AI Developer Advocate · Developer Experience Engineer',
            mission: 'AI 기능을 개발자가 이해하고 실행하며 신뢰할 수 있게 만든 뒤, 그들의 마찰을 제품에 다시 반영합니다.',
            route: '전문 레슨 11개 · 905분',
            baseline: '소프트웨어 기초, API 활용, 명확한 기술 글쓰기',
            own: [
              '깨끗한 환경에서도 동작하는 통합과 예제를 만듭니다.',
              'API, 도구, 프로토콜, 스킬 계약을 정확하게 설명합니다.',
              '추측하지 않고 개발자 마찰을 재현합니다.',
              '지원 신호를 문서, 툴링, 제품 피드백으로 전환합니다.'
            ],
            fit: '구축, 교육, 다른 개발자와 함께 디버깅, 어려운 시스템을 읽기 쉽게 만드는 일을 즐깁니다.',
            boundary: '콘텐츠 전용 마케팅이 아닙니다. 신뢰는 실행 가능한 기술 작업과 정확한 설명에서 나옵니다.',
            portfolio: '동작하는 통합, 예제, 재사용 에이전트 패키지, 마찰 리포트가 담긴 개발자 온보딩 패키지를 공개합니다.',
            evidence: ['신규 환경 셋업 증거', '긍정·부정·실패 예제', '구체적 개선으로 연결된 피드백'],
            coverage: '이 루트는 API, 도구 계약, MCP, Agent Skills, 패키징, 평가, 피드백을 다룹니다.',
            stillEarned: '라이브 오디언스 실습, 커뮤니티 판단력, 도입 분석, 편집 깊이, 지속적인 개발자 지원.',
            footer: '레슨으로 흩어진 데모 더미가 아니라 하나의 일관된 온보딩 경험을 만드세요.'
          },
          'ai-data-engineer': {
            title: 'AI 데이터 시스템',
            aliases: 'AI Data Engineer · Machine Learning Data Engineer · Retrieval Engineer',
            mission: '학습, 평가, 프로덕션 AI 동작이 신뢰할 수 있는 증거를 쓰도록 데이터·검색 파이프라인을 구축합니다.',
            route: '전문 레슨 11개 · 915분',
            baseline: 'Python, 자료구조, 통계, 파이프라인 기초',
            own: [
              '학습·검색 데이터를 수집, 변환, 버전 관리, 검증합니다.',
              '임베딩, 인덱싱, 검색, 평가 파이프라인을 구축합니다.',
              '데이터 품질 검사를 정의하고 조용한 드리프트를 조사합니다.',
              '계보, 신선도, 비용, 런타임 상태를 노출합니다.'
            ],
            fit: '파이프라인, 데이터 품질, 재현성, UI에서 멀리 떨어진 시스템 실패 디버깅을 즐깁니다.',
            boundary: 'AI 데이터 제품에 초점을 둡니다. 데이터 엔지니어링의 웨어하우스·DB·플랫폼 깊이를 대체하지는 않습니다.',
            portfolio: '품질 게이트, 평가 데이터, 운영 리포트가 있는 버전 관리된 문서-검색 파이프라인을 출시합니다.',
            evidence: ['재현 가능한 수집과 계보', '검색 품질·신선도 지표', '실패 복구와 관측 가능성 증거'],
            coverage: '이 루트는 데이터 관리, 피처, 파이프라인, 임베딩, 컨텍스트, RAG, 평가, 프로덕션, 관측 가능성을 다룹니다.',
            stillEarned: '고급 SQL, 웨어하우스 아키텍처, 거버넌스, 개인정보 운영, 대규모 분산 데이터 시스템.',
            footer: '전문 시퀀스를 순서 있는 작업으로 다루기 전에 통계·데이터 엔지니어링 기준선을 먼저 닫으세요.'
          },
          'agentic-ai-engineer': {
            title: '에이전트 시스템 엔지니어링',
            aliases: 'Agent Systems Engineer · Agentic AI Engineer · AI Agent Engineer',
            mission: '도구를 쓰는 모델 주변 런타임을 설계해 컨텍스트, 메모리, 권한, 오케스트레이션, 실패, 증거가 명시되도록 만듭니다.',
            route: '전문 레슨 14개 · 865분',
            baseline: 'LLM 애플리케이션 기초 + 타입이 있는 도구 인터페이스',
            own: [
              '도구 계약과 관찰·결정·행동 루프를 설계합니다.',
              '컨텍스트, 메모리, 상태, 내구성 있는 실행을 제어합니다.',
              '오케스트레이션 경계와 종료 정책을 선택합니다.',
              '권한을 위협 모델링하고 전체 궤적을 평가합니다.',
              '트레이스와 명시적 실패 제어로 런타임을 운영합니다.'
            ],
            fit: '런타임 설계, 상태 머신, 분산 조율, 안전 경계, 어려운 실패 분석을 즐깁니다.',
            boundary: '모델 동작 주변의 시스템 엔지니어링이며, 에이전트 루프를 넣는다고 제품이 자율화된다는 약속이 아닙니다.',
            portfolio: '메모리, 오케스트레이션, 위협 모델, 궤적 평가, 실패 런북이 있는 경계 있는 도구 사용 런타임을 출시합니다.',
            evidence: ['결정적 도구·상태 트레이스', '권한·샌드박스·인젝션 제어', '종료·복구·평가 증거'],
            coverage: '이 루트는 도구, MCP, 루프, 컨텍스트, 메모리, 그래프, 오케스트레이션, 보안, 평가, 런타임, 관측 가능성을 다룹니다.',
            stillEarned: '제공자별 인프라, 대규모 분산 운영, 지연 엔지니어링, 팀과 함께하는 프로덕션 소유권.',
            footer: 'LLM·도구 인터페이스 기준선을 먼저 완료하세요. 모든 에이전트 주장은 런타임이 증명해야 합니다.'
          },
          'applied-ai-engineer': {
            title: 'LLM 제품 엔지니어링',
            aliases: 'Applied AI Engineer · LLM Engineer · AI Product Engineer',
            mission: '모델 능력을 그라운딩되고 평가되며 보호되고 비용을 인지하며 프로덕션에서 복구 가능한 유용한 제품 동작으로 바꿉니다.',
            route: '전문 레슨 12개 · 885분',
            baseline: '소프트웨어 엔지니어링 + LLM 기초',
            own: [
              '모델 대면 인터페이스와 구조화된 계약을 설계합니다.',
              '컨텍스트, 검색, 도구로 동작을 그라운딩합니다.',
              '기능을 최적화하기 전에 작업 평가를 만듭니다.',
              '안전, 비용, 지연, 캐싱, 폴백을 제어합니다.',
              '관측 가능한 동작으로 완전한 기능을 릴리스합니다.'
            ],
            fit: '제품 요구를 모델 동작에 연결하고 모델 주변 소프트웨어를 소유하고 싶습니다.',
            boundary: '기반 모델 연구나 모델 학습이 아닙니다. 모델 능력이 실제 제품 제약을 만나는 지점에서 시작합니다.',
            portfolio: '구조화 출력, 도구, 평가 세트, 비용·지연 예산, 보호된 릴리스가 있는 그라운딩된 제품 기능을 출시합니다.',
            evidence: ['대표 성공·실패 사례', '품질·비용·지연 트레이드오프', '폴백·릴리스·롤백 증거'],
            coverage: '이 루트는 프롬프팅, 구조화 출력, 임베딩, 컨텍스트, RAG, 도구, 평가, 비용, 가드레일, 프로덕션, 게이트웨이, 릴리스를 다룹니다.',
            stillEarned: '제품 디스커버리, 인터랙션 디자인, 실제 사용자 리서치, 도메인 규제, 지속 트래픽 하의 기능 운영.',
            footer: 'LLM 기초 점검 후에만 시작하세요. 포트폴리오 증명은 12개 페이지 완료가 아니라 통합된 동작입니다.'
          },
          'ai-evaluation-reliability-engineer': {
            title: 'AI 평가 및 신뢰성',
            aliases: 'AI Evaluation Engineer · AI Reliability Engineer · Machine Learning Site Reliability Engineer',
            mission: '모델·에이전트 동작을 측정 가능하게 만들고, 릴리스 전 실패를 드러내며, 프로덕션에서 남는 실패를 위한 운영 제어를 구축합니다.',
            route: '전문 레슨 12개 · 750분',
            baseline: '통계, 소프트웨어 테스트, 프로덕션 시스템',
            own: [
              '평가 세트, 지표, 채점기, 실패 분류를 정의합니다.',
              '모델, 에이전트, 서빙 동작을 계측합니다.',
              '릴리스 게이트, 실험, 회귀 탐지를 구축합니다.',
              '부하, 저하, 복구, 인시던트 대응을 테스트합니다.',
              '증거를 롤아웃·운영 결정에 연결합니다.'
            ],
            fit: '통계, 적대적 테스트, 관측 가능성, 릴리스 판단, 인시던트에서 배우는 일을 즐깁니다.',
            boundary: '오프라인 모델 정확도보다 넓습니다. 신뢰성은 애플리케이션, 런타임, 인프라, 대응 프로세스를 포함합니다.',
            portfolio: '트레이스에 연결된 행동 평가 하네스, 릴리스 게이트, 부하·실패 실험, 인시던트 런북을 출시합니다.',
            evidence: ['버전 관리된 사례와 지표 근거', '회귀·롤아웃 결정', '관찰된 복구와 잔여 위험'],
            coverage: '이 루트는 모델·LLM·에이전트 평가, 관측 가능성, 서빙 지표, 실험, 부하, 카나리 릴리스, 카오스, SRE를 다룹니다.',
            stillEarned: '실제 온콜 경험, 조직별 인시던트 프로세스, 프로덕션 트래픽, 컴플라이언스 증거, 크로스팀 릴리스 권한.',
            footer: '통계·프로덕션 기준선을 먼저 닫으세요. 모든 릴리스 게이트는 증거에 기반한 결정으로 다루세요.'
          }
        },
        domains: {
          'building-and-deploying': {
            number: '도메인 01 · 애플리케이션 시스템',
            title: 'AI 애플리케이션 구축 및 배포',
            copy: '첫 모델 대면 인터페이스에서 그라운딩된 동작, 평가, 안전장치, 프로덕션 운영까지 이동합니다. 애플리케이션은 모델 주변의 전체 시스템입니다.',
            root: '완전한 AI 애플리케이션 구축',
            meta: '12개 레슨 경로 · 780분',
            nodes: [
              { index: '01 · 인터페이스', title: '모델 상호작용 계약', desc: '의도를 명시적 입력·출력·실패 동작이 있는 경계 있는 요청으로 바꿉니다.' },
              { index: '02 · 계약', title: '구조화 생성 계약', desc: '생성 데이터를 파싱·검증 가능하고 애플리케이션 코드에 안전하게 넘길 수 있게 만듭니다.' },
              { index: '03 · 그라운딩', title: '증거 표현', desc: '모델이 쓸 수 있는 위치에 증거를 표현·검색·배치합니다.' },
              { index: '04 · 검색', title: '검색과 신선도', desc: '생성 주변의 수집, 검색, 랭킹, 인용, 신선도 루프를 구축합니다.' },
              { index: '05 · 증거', title: '행동 평가 게이트', desc: '허용 동작을 정의하고 사례를 수집하며 결과를 채점하고 회귀를 게이팅합니다.' },
              { index: '06 · 운영', title: '서빙과 복구', desc: '실제 트래픽 아래에서 서빙, 관측, 릴리스, 복구, 비용을 제어합니다.' }
            ]
          },
          'software-fundamentals': {
            number: '도메인 02 · 엔지니어링 기반',
            title: '소프트웨어 공학 기초',
            copy: '코딩 에이전트는 타이핑을 줄이지, 엔지니어링 판단을 대체하지 않습니다. 애플리케이션 스택, 데이터, 아키텍처, 보안, 신뢰성, 프로덕션 운영의 트레이드오프를 조율하는 법을 배웁니다.',
            root: 'AI 시스템 뒤의 소프트웨어 트레이드오프 조율',
            meta: '13개 레슨 기초 경로 · 730분',
            footerNote: '기초 경로로 순서를 따르거나, 지금 필요한 역량에 맞는 대표 분기로 진입하세요.',
            nodes: [
              { index: '01 · 애플리케이션', title: '엔드투엔드 애플리케이션 딜리버리', desc: '요청 처리, 스트리밍, 영속성, 폴백, 헬스 체크, 배포를 하나의 동작 시스템으로 연결합니다.' },
              { index: '02 · 데이터', title: '데이터 수명주기와 저장', desc: '애플리케이션이 필요로 하는 접근 패턴에 맞춰 표현, 검증, 버전, 보존, 신선도를 선택합니다.' },
              { index: '03 · 아키텍처', title: '시스템 아키텍처와 경계', desc: '입출력, 오류, 권한, 상태를 하나의 명시적 경계로 설계한 뒤 더 큰 시스템에 넣습니다.' },
              { index: '04 · 보증', title: '안전하고 회복력 있는 시스템', desc: '프로덕션 전에 시크릿, 권한, 의존성, 데이터 처리, 릴리스 증거를 감사합니다.' },
              { index: '05 · 운영', title: '프로덕션 규모와 서비스 소유', desc: '서비스 목표를 정의하고 헬스 신호를 보며 런북을 준비하고 증거 기반 인시던트 대응을 연습합니다.' }
            ]
          },
          'coding-agents': {
            number: '도메인 03 · 에이전트 보조 엔지니어링',
            title: '에이전트 보조 엔지니어링',
            copy: '코딩 에이전트는 태스크, 컨텍스트, 도구, 피드백, 종료 조건이 믿을 수 있는 하네스를 이룰 때 유용합니다. 실제 저장소 작업 주변에 그 시스템을 설계하는 법을 배웁니다.',
            root: '모델 능력을 믿을 수 있는 작업으로 전환',
            meta: '16개 레슨 경로 · 900분',
            nodes: [
              { index: '01 · 프레이밍', title: '태스크 프레이밍', desc: '요청을 범위, 제약, 권한, 증거, 중지 규칙으로 바꿉니다.' },
              { index: '02 · 계획', title: '증거 기반 계획', desc: '가장 작은 일관된 변경을 제안하기 전에 저장소를 먼저 조사합니다.' },
              { index: '03 · 하네스', title: '에이전트 워크벤치', desc: '루프, 컨텍스트 경계, 도구, 트랜스크립트, 종료 정책을 엔지니어링합니다.' },
              { index: '04 · 컨텍스트', title: '지시와 메모리', desc: '내구성 있는 지침을 올바른 범위에 두고 런타임 상태를 관측 가능하게 유지합니다.' },
              { index: '05 · 피드백', title: '런타임 피드백', desc: '컴파일러, 테스트, 브라우저, 와이어 증거를 다음 결정에 다시 넣습니다.' },
              { index: '06 · 검증', title: '검증과 리뷰', desc: '에이전트의 완료 주장과 독립적으로 요청된 동작을 증명합니다.' },
              { index: '07 · 위임', title: '격리된 위임', desc: '모호한 소유권이나 상태를 공유하지 않고 경계 있는 작업을 에이전트에 나눕니다.' },
              { index: '08 · 개선', title: '지속 가능한 개선', desc: '수정을 테스트, 지시, 툴링, 재사용 가능한 제약으로 전환합니다.' }
            ]
          },
          'shaping-the-build': {
            number: '도메인 04 · 제품 판단',
            title: '제품 판단 및 딜리버리',
            copy: '구현 전에 어떤 결과가 중요한지, 어떤 증거가 작업을 뒷받침하는지, 어떤 위험에 주의해야 하는지, 변경이 도움이 되었는지 어떻게 알지를 결정합니다.',
            root: '출력을 만들기 전에 올바른 빌드를 선택',
            meta: '8개 레슨 경로 · 550분',
            nodes: [
              { index: '01 · 결과', title: '출력보다 결과', desc: '기능이나 구현을 논의하기 전에 원하는 변화 상태를 정의합니다.' },
              { index: '02 · 관찰', title: '워크플로우 디스커버리', desc: '예외, 핸드오프, 숨은 노동을 포함해 현재 일이 어떻게 일어나는지 연구합니다.' },
              { index: '03 · 위험', title: '가정과 위험', desc: '참이어야 하는 것을 드러내고 빌드를 무효화할 수 있는 불확실성을 테스트합니다.' },
              { index: '04 · 슬라이스', title: '테스트 가능한 슬라이스', desc: '의사결정 품질의 증거를 낼 수 있는 가장 작은 엔드투엔드 변경을 선택합니다.' },
              { index: '05 · 명세', title: '실행 가능한 명세', desc: '구현 판단을 제거하지 않으면서 제약과 수용 기준을 관측 가능하게 만듭니다.' },
              { index: '06 · 측정', title: '성공 지표', desc: '제품 결과를 선행·가드레일·운영 지표에 연결합니다.' },
              { index: '07 · 단계', title: '릴리스 전략', desc: '다음에 필요한 증거에 맞춰 프로토타입·파일럿·프로덕션 투자를 맞춥니다.' },
              { index: '08 · 소유', title: '피드백 소유권', desc: '신호를 읽고, 결정을 내리고, 시스템을 바꾸는 사람을 지정합니다.' }
            ]
          }
        }
      },
      glossary: {
        navContents: '목차',
        navCatalog: '카탈로그',
        navRoadmap: '로드맵',
        navGlossary: '용어집',
        navAbout: '소개',
        kicker: '참조 원장 · 커리큘럼 v1.0',
        title: 'AI 엔지니어링 용어집',
        deck1: '내가 직접 구축하는 시스템을 위한 정밀한 실무 정의. 용어의 뜻에서 출발하여 예시, 구별점, 연계 레슨을 통해 어휘를 공학적 판단력으로 전환하세요.',
        deck2: '각 항목은 용어의 실무 정의, AI 시스템에서 중요한 이유, 실무에서의 발현 형태, 엔지니어들이 흔히 혼동하는 개념과의 구별점을 명확히 분리합니다. 용어, 별칭 또는 아이디어로 검색하고, 학습 영역별로 원장을 좁혀보며, 전체 유도가 필요할 때는 연결된 코스 레슨이나 1차 출처를 확인하세요. 모든 항목에는 고유한 프래그먼트 링크가 있어 특정 정의를 바로 인용할 수 있습니다.',
        statTerms: '색인된 용어',
        statCategories: '학습 영역',
        statDeepLinks: '안정적인 딥링크',
        searchLabel: '원장 검색',
        searchPlaceholder: '용어, 별칭 또는 개념 입력',
        clearBtn: '지우기',
        learningAreaLabel: '학습 영역',
        jumpToLetterLabel: '알파벳으로 이동',
        liveReferenceLabel: '실시간 레퍼런스',
        resultsHeading: '참조 항목',
        loadingEntries: '참조 항목을 불러오는 중...',
        showMore: '더 보기',
        showAll: '전체 보기',
        whyItMatters: '중요한 이유',
        inPractice: '실무 예시',
        notToBeConfusedWith: '구별해야 할 개념',
        relatedTerms: '관련 용어',
        courseLessons: '관련 레슨',
        primarySources: '1차 공식 출처',
        copyLink: '링크 복사',
        linkCopied: '복사됨',
        readAloud: '소리 내어 읽기',
        readingAloud: '읽는 중...',
        stopReading: '중지',
        emptyTitle: '일치하는 용어가 없습니다',
        emptyCopy: '다른 검색어를 입력하거나 필터를 초기화해 보세요.',
        resetFilters: '필터 초기화',
        workingDefinition: '실무 정의',
        whyCalled: '이 명칭이 붙은 이유',
        commonShortcut: '흔한 표현',
        alsoCalled: '다른 명칭',
        distinctionsAndEvidence: '구별점 및 출처 근거',
        allTerms: '전체 용어'
      },
      about: {
        navContents: '목차',
        navCatalog: '카탈로그',
        navRoadmap: '로드맵',
        navGlossary: '용어집',
        navAbout: '소개',
        eyebrow: '프로젝트 소개',
        title: 'AI Engineering from Scratch 소개',
        lede: 'AI Engineering from Scratch는 모든 핵심 AI 알고리즘을 손으로 직접 밑바닥부터 구현하는 무료 오픈소스 커리큘럼입니다. 선형대수학부터 자율 에이전트까지, Python, TypeScript, Rust, Julia 4개 언어로 20개 단계(Phase)에 걸쳐 <span id="aboutLessonCount">모든 공개 레슨</span>을 제공합니다.',
        whyTitle: '프로젝트가 존재하는 이유',
        whyP1: '대부분의 AI 학습 자료는 파편화되어 있습니다. 논문 하나, 파인튜닝 글 하나, 화려한 프레임워크 데모가 제각각 흩어져 있습니다. 이로 인해 챗봇을 배포하더라도 손실 곡선(loss curve)의 원리를 설명하지 못하거나, 에이전트에 도구를 연결하면서도 모델 내부에서 attention 메커니즘이 어떻게 동작하는지 설명하지 못하는 경우가 많습니다.',
        whyP2: '본 커리큘럼은 AI 엔지니어링의 단단한 척추 역할을 합니다. 모든 알고리즘을 순수 수학으로부터 먼저 작성한 뒤 상용 라이브러리를 실행하여 라이브러리가 내부에서 무엇을 계산하는지 확인합니다. PyTorch가 등장할 때쯤이면 내부 동작 원리를 이미 꿰뚫어 보게 됩니다. 각 레슨은 실무에 즉시 활용할 수 있는 프롬프트, 스킬, 에이전트, MCP 서버 등 재사용 가능한 아티팩트를 남깁니다.',
        howTitle: '제작 및 검증 방식',
        howP1: '레슨은 AI 보조를 받아 집필되며, 1차 원문(Primary Source)을 기준으로 사람이 직접 검토합니다. 사실이나 개념을 서술할 때는 2차 요약글이 아닌 RFC, 공식 명세, 학술 논문 원문을 직접 인용합니다. 모든 오류 수정과 개선 제안은 GitHub에서 투명하게 공개 진행됩니다.',
        howP2: '웹사이트는 불필요한 프레임워크 없이 순수 HTML, CSS, 바닐라 JavaScript로 가볍고 명료하게 제작되었습니다. 단 하나의 빌드 스크립트(<code>site/build.js</code>)가 저장소의 마크다운 레슨을 파싱하여 카탈로그, 검색 색인, 사이트맵, <code>llms.txt</code>를 생성합니다.',
        whoTitle: '원작자 및 크레딧',
        whoP1: '<a href="https://github.com/rohitg00" target="_blank" rel="noopener">Rohit Ghumare</a> 님과 글로벌 기여자들이 관리하고 있습니다. MIT 라이선스 기반의 영구 무료 오픈소스이며, 유료 결제나 폐쇄형 콘텐츠가 전혀 없습니다.',
        koreanTitle: '한국어 에디션 & AX Academy',
        koreanP1: '글로벌 최고 수준의 AI 엔지니어링 실습 커리큘럼을 한국 AI 개발자와 학습자 생태계에 널리 공유하기 위해 <strong>AX Academy</strong>에서 공식 현지화 및 한국어 번역 프로젝트를 주도하고 있습니다. 한국어판 역시 원작의 철학을 그대로 이어받아 완전 무료 오픈소스로 운영됩니다.',
        involvedTitle: '참여 및 기여 방법',
        involved1: '원본 저장소: <a href="https://github.com/rohitg00/ai-engineering-from-scratch" target="_blank" rel="noopener">github.com/rohitg00/ai-engineering-from-scratch</a>',
        involved2: '오류 제보 또는 레슨 아이디어 제안: <a href="https://github.com/rohitg00/ai-engineering-from-scratch/issues/new/choose" target="_blank" rel="noopener">이슈 열기</a>',
        involved3: '학습 시작하기: <a href="catalog.html">카탈로그 둘러보기</a> 또는 <a href="prereqs.html">로드맵 따라가기</a>'
      },
      catalog: {
        pageTitle: '레슨 카탈로그 - AI Engineering from Scratch',
        skipLink: '본문으로 건너뛰기',
        navContents: '목차',
        navCatalog: '카탈로그',
        navRoadmap: '로드맵',
        navGlossary: '용어집',
        navAbout: '소개',
        title: '레슨 카탈로그',
        subtitle: '20개 단계의 모든 레슨. 검색, 필터, 정렬.',
        searchPlaceholder: '레슨 검색...',
        searchAria: '레슨 검색',
        phaseFilterAria: '단계별 필터',
        statusFilterAria: '레슨 상태별 필터',
        allPhases: '모든 단계',
        allStatus: '모든 상태',
        statusComplete: '완료',
        statusPlanned: '예정',
        loading: '레슨 색인 불러오는 중...',
        noscript: '로컬 필터링에는 JavaScript가 필요합니다. 모든 레슨은 <a href="https://github.com/rohitg00/ai-engineering-from-scratch#contents">GitHub 커리큘럼 색인</a>에서도 확인할 수 있습니다.',
        colPhase: '단계',
        colLesson: '레슨',
        colType: '유형',
        colLanguage: '언어',
        colStatus: '상태',
        showing: '{shown} / {matched}개 일치 · 전체 {total}개',
        empty: '필터 조건에 맞는 레슨이 없습니다.',
        loadError: '레슨 색인을 불러올 수 없습니다.',
        loadErrorHint: '페이지를 새로고침하거나 GitHub 커리큘럼 색인을 이용하세요.',
        phaseGroup: '단계 {id}: {name}',
        lessonCount: '{n}개 레슨',
        remaining: '일치하는 레슨 {n}개 더 있음.',
        showMore: '더 보기',
        showNext: '다음 {n}개 보기',
        showAll: '전체 보기',
        typeLearn: '학습',
        typeBuild: '구현',
        typeCapstone: '캡스톤',
        typeReference: '참고',
        footerCopy: 'AI Engineering from Scratch · 오픈소스 · 영구 무료.',
        footerHome: '홈',
        footerGlossary: '용어집',
        footerReport: '오류 보고 / 제안'
      }
    },
    en: {
      actionTitle: 'Act on this lesson',
      actionIntro: 'Use this loop to turn reading into evidence. Each checkpoint is yours to mark and stays separate from quiz correctness.',
      checkpointsStatus: '{done} of 5 checkpoints',
      quizPassed: 'passed',
      quizNotPassed: 'not yet passed',
      stepReadTitle: 'Read',
      stepReadCopy: 'Understand the claim and constraints.',
      stepBuildTitle: 'Build',
      stepBuildCopy: 'Create or modify the lesson artifact.',
      stepRunTitle: 'Run',
      stepRunCopy: 'Execute the relevant program or exercise.',
      stepProveTitle: 'Prove',
      stepProveCopy: 'Capture output that supports the claim.',
      stepContinueTitle: 'Continue',
      stepContinueCopy: 'Mark the lesson complete when you are ready.',
      runRepoLabel: 'Run from the repository',
      checkingRunnable: 'Checking for a standalone runnable file',
      checkingNote: 'The page is checking the lesson code directory for a standalone entry point.',
      copyCmd: 'Copy command',
      copied: 'Copied',
      prevLesson: 'Previous',
      nextLesson: 'Next',
      locked: 'Locked',
      testUnderstanding: 'Test Your Understanding',
      didYouGetIt: 'Did you get it?',
      questionOf: 'Question {i} of {n}',
      quizScorePrompt: 'Complete all questions to see your score',
      quizPassedBadge: 'Passed',
      quizRetryBadge: 'Try again',
      artifactDeliverable: 'This lesson has no separate output artifact. The lesson itself is the deliverable.',
      openFile: 'Open file',
      copyPrompt: 'Copy prompt',
      localProgress: 'Local progress',
      runNoteRepoRoot: 'Run from the repository root, the folder containing README.md.',
      runNoteNoFile: 'Use the worked examples as guidance. Mark Run only after you execute an appropriate example or exercise. A code fence is not automatically a runnable program.',
      noStandaloneFile: 'No standalone main file detected',
      preQuizTitle: 'Pre-Lesson Check',
      midQuizTitle: 'Mid-Lesson Check',
      postQuizTitle: 'Post-Lesson Quiz',
      quizTitleDefault: 'Quiz',
      retryCheck: 'Retry this check',
      quizScoreCorrect: '{correct}/{total} correct',
      quizScoreAnswered: '{answered}/{total} answered. Review the feedback, then retry when ready.',
      quizResetStatus: 'This check was reset. Choose a new answer for each question.',
      onThisPage: 'On this page',
      learningObjectivesTitle: 'Learning Objectives',
      pathMinutes: 'Path {min} min',
      lessonMinutes: 'Lesson {min} min',
      groupPrefix: 'Group {group}',
      focusedPathContext: 'Focused path context',
      captureCheckpointEvidence: 'Capture this checkpoint evidence:',
      checkpointLabel: 'Checkpoint:',
      quickStartLabel: 'Quick start:',
      quickStartDefaultGoal: 'Complete the first executable checkpoint.',
      quickStartMinutes: 'Allow about {min} minutes for this first run.',
      quickStartCommandLabel: 'Quick start command:',
      quickStartEvidenceLabel: 'Quick start evidence:',
      diagramTitle: 'Diagram',
      diagramRendering: 'Rendering diagram...',
      diagramFailed: 'Diagram could not be rendered.',
      diagramExpand: 'Expand',
      outputsPanelTitle: 'What This Lesson Ships',
      outputsPanelSubtitle: 'Prompts, skills, and artifacts you can use right now',
      outputsLoading: 'Loading outputs...',
      outputsFallback: 'No separate output artifacts were found for this lesson.',
      codeFallback: 'This lesson has no separate runnable file. Use the worked examples in the lesson.',
      codeFallbackFetchFailed: 'No separate runnable files were found for this lesson.',
      viewLessonSource: 'View lesson source',
      codePanelTitle: 'Run the Code',
      codePanelSubtitle: 'Requires a local clone. Run copied commands from the repository root, the directory containing README.md and phases/.',
      codeLoading: 'Loading code files...',
      installSkillLabel: 'Install',
      copyInstallCmd: 'Copy install',
      installPromptHint: 'Paste into Claude, Cursor, Codex, OpenClaw, Hermes, or any agent that reads prompts',
      openSkillMd: 'Open SKILL.md',
      viewKoreanVersion: 'Korean version',
      openOriginalFile: 'Open file',
      quizDeeperIntro: 'Want a deeper quiz?',
      quizDeeperCodex: 'In Codex use <code>check-understanding {phase}</code> or choose it from <code>/skills</code>.',
      quizDeeperClaude: 'In Claude Code use <code>/check-understanding {phase}</code>.',
      quizDeeperOther: 'In another compatible host say: <code>Use check-understanding to quiz me on Phase {phase}.</code>',
      quizScorePerfect: 'Perfect score!',
      quizScoreGreat: 'Great work!',
      quizScoreStudy: 'Keep studying!',
      quizAnswerCorrect: 'Correct. ',
      quizAnswerIncorrect: 'Not quite. ',
      learningPathTitle: 'Learning Path',
      lessonOfTotal: 'Lesson {current} of {total}',
      earlierLessons: '{n} earlier',
      laterLessons: '{n} later',
      phaseCompletedProgress: "You've completed {done} of {total} lessons in this phase",
      readyForNextPhase: 'Ready for Phase {phase}: {name}',

      // Homepage (index.html) translations
      home: {
        navContents: 'Contents',
        navBooks: 'Books',
        navCatalog: 'Catalog',
        navRoadmap: 'Roadmap',
        navGlossary: 'Glossary',
        navAbout: 'About',
        navCertifications: 'Certifications',
        navReport: 'Report',
        openSourceMit: 'open source · MIT',
        taglineSuffix: 'Every algorithm built from raw math before a single framework gets imported.',
        attribution: 'Maintained by Rohit Ghumare and contributors. Run on your own machine.',
        btnStartCourse: 'Start the Course',
        btnExplorePaths: 'Explore Learning Paths',
        btnStarGitHub: 'Star on GitHub',
        btnFollow: 'Follow @rohitg00',
        terminalLearn: 'Learn in your terminal',
        terminalCopy: 'copy',
        terminalCopied: 'copied!',
        terminalCaption: 'Your agent becomes your tutor: placement quiz, personalized path, lessons taught interactively in your terminal.',
        figPrevious: 'Previous',
        figNext: 'Next',
        plateCaption1: 'Plate 1 of 3. Every layer of this diagram is a lesson you implement by hand.',
        plateCaption2: 'Plate 2 of 3. The AI tutor teaches in your terminal. You write code, it runs tests.',
        plateCaption3: 'Plate 3 of 3. The learning curve is steep because it’s real. No magic.',
        learnersEyebrow: 'Read by engineers and students at',
        learnersQuote: '“Obsessed with the AI Engineering from Scratch repo.” <span class="learners-quote-attr">- AI engineer at Google</span>',
        prefaceEyebrow: 'How this works',
        prefaceP1: 'Most AI material teaches in scattered pieces. A paper here, a fine-tuning post there, a flashy agent demo somewhere else. The pieces rarely line up. You ship a chatbot but can\'t explain its loss curve. You hook a function to an agent but can\'t say what attention does inside the model that\'s calling it.',
        prefaceP2Prefix: 'This curriculum is the spine.',
        prefaceP2Suffix: 'four languages: Python, TypeScript, Rust, Julia. Linear algebra at one end, autonomous swarms at the other. Every algorithm gets built from raw math first. Backprop. Tokenizer. Attention. Agent loop. By the time PyTorch shows up, you already know what it\'s doing under the hood.',
        prefaceP3: 'Each lesson runs the same loop: read the problem, derive the math, write the code, run the test, keep the artifact. No five-minute videos, no copy-paste deploys, no hand-holding. Free, open source, and built to run on your own laptop.',
        coursePathsTitle: 'Choose the work you want to do',
        coursePathsHeaderCopy: 'AI engineering is larger than model code. Choose one of four core learning paths, then learn from the same source, labs, tests, and artifacts in the browser or on GitHub.',
        viewLearningPaths: 'View Learning Paths',
        browseCareerRoutes: 'Browse career routes',
        learningPathsRoot: 'AI Engineering',
        learningPathsSub: '4 connected domains',
        pathAppLabel: 'Building and Deploying AI Applications',
        pathSoftwareLabel: 'Software Engineering Fundamentals',
        pathAgentsLabel: 'Agent-Assisted Engineering',
        pathShapingLabel: 'Product Judgment and Delivery',
        statBlockTitle: 'Current Progress',
        statFinishedLessons: 'Finished Lessons',
        statPhases: 'Phases',
        statLanguages: 'Languages',
        statGlossaryTerms: 'Glossary Terms',
        tocTitleSuffix: 'Curriculum',
        tocSubtitle: 'Tap a phase to expand its lessons. Each one ships when its math, code, and test are all written.',
        legendComplete: 'Complete',
        legendInProgress: 'In progress',
        legendPlanned: 'Planned',
        modalFooterNote: 'Progress saved in browser only',
        modalResetBtn: 'Reset progress',
        modalResetConfirm: 'Clear all your local progress (quiz answers and completed lessons)? This cannot be undone.',
        openLesson: 'Open lesson',
        reviewLesson: 'Review',
        comingSoon: 'Coming soon',
        done: 'Done',
        markDone: 'Mark done',
        markNotDone: 'Mark as not done',
        booksTitle: 'The book edition · six volumes',
        booksSubtitle: 'The course, compiled. EPUB and PDF built from the same lessons and attached to every GitHub release. The site stays the living edition. Every chapter links back here for the animated figures, quizzes, and code.',
        booksNote: 'Links resolve to the newest <a href="https://github.com/rohitg00/ai-engineering-from-scratch/releases" target="_blank" rel="noopener">GitHub release</a> · rebuilt by CI from the lessons on every release · <a href="https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/book/README.md" target="_blank" rel="noopener">how it\'s made</a>',
        colophonEyebrow: 'Colophon',
        colophonText: 'The entire curriculum is on GitHub. Clone it, fork it, learn at your own pace. No paywall, no signup. Every lesson has runnable code in Python, TypeScript, Rust, or Julia, depending on what fits the concept best.',
        footerCopy: '© 2026 · open source · free forever'
      },
      routes: {
        recFirst: 'Recommended first',
        coreDomain: 'Core domain',
        focusedPath: 'Focused path',
        practiceEvidence: 'Practice by evidence',
        newToAiTitle: 'New to AI engineering',
        newToAiDesc: 'Set up a working environment, run the repository, and learn the lesson workflow before choosing a specialization.',
        appTitle: 'Building and Deploying AI Applications',
        appDesc: 'Move from prompts, structured outputs, embeddings, and retrieval through evaluation, serving, observability, and safe release.',
        softwareTitle: 'Software Engineering Fundamentals',
        softwareDesc: 'Build the repository, environment, interface, debugging, verification, security, release, and operational foundations AI systems depend on.',
        agentsTitle: 'Agent-Assisted Engineering',
        agentsDesc: 'Frame the task, plan from repository evidence, engineer the loop and harness, isolate delegation, verify the result, and preserve feedback.',
        shapingTitle: 'Product Judgment and Delivery',
        shapingDesc: 'Turn observed work into outcomes, assumptions, testable slices, executable specifications, measurement plans, staged releases, and owned feedback.',
        mcpTitle: 'Model Context Protocol (MCP)',
        mcpDesc: 'Build, secure, verify, and operate stateless MCP systems from wire envelopes through release gates.',
        skillsTitle: 'Agent Skills',
        skillsDesc: 'Build, invoke, route, secure, evaluate, package, and verify portable skills in real agent hosts.',
        certTitle: 'Certification preparation',
        certDesc: 'Choose a certification route, complete practical labs, keep learner-owned artifacts, and use original assessments.',
        openLesson: 'Open lesson',
        ghSource: 'GitHub source',
        startPath: 'Start path',
        ghPath: 'GitHub path',
        explorePaths: 'Explore paths',
        ghTutor: 'GitHub tutor'
      },
      phaseNames: {
        0: 'Setup & Tooling',
        1: 'Math Foundations',
        2: 'ML Fundamentals',
        3: 'Deep Learning Core',
        4: 'Computer Vision',
        5: 'NLP: Foundations to Advanced',
        6: 'Speech & Audio',
        7: 'Transformers Deep Dive',
        8: 'Generative AI',
        9: 'Reinforcement Learning',
        10: 'LLMs from Scratch',
        11: 'LLM Engineering',
        12: 'Multimodal AI',
        13: 'Tools & Protocols',
        14: 'Agent Engineering',
        15: 'Autonomous Systems',
        16: 'Multi-Agent & Swarms',
        17: 'Infrastructure & Production',
        18: 'Ethics, Safety & Alignment',
        19: 'Capstone Projects'
      },
      phaseDescs: {
        0: 'Get your environment ready for everything that follows.',
        1: 'The intuition behind every AI algorithm, through code.',
        2: 'Classical ML — still the backbone of most production AI.',
        3: 'Neural networks from first principles. No frameworks until you build one.',
        4: 'From pixels to understanding — image, video, 3D, VLMs, and world models.',
        5: 'Language is the interface to intelligence.',
        6: 'Hear, understand, speak.',
        7: 'The architecture that changed everything.',
        8: 'Create images, video, audio, 3D, and more.',
        9: 'The foundation of RLHF and game-playing AI.',
        10: 'Build, train, and understand large language models.',
        11: 'Put LLMs to work in production.',
        12: 'See, hear, read, and reason across modalities — from ViT patches to computer-use agents.',
        13: 'The interfaces between AI and the real world.',
        14: 'Build agents from first principles, use coding agents reliably, and shape the work before implementation.',
        15: 'Long-horizon agents, self-improvement, and the 2026 safety stack.',
        16: 'Coordination, emergence, and collective intelligence.',
        17: 'Ship AI to the real world.',
        18: 'Build AI that helps humanity. Not optional.',
        19: '17 end-to-end products + 9 deep-build tracks. 20-40 hours per project; 4-12 lessons per track.'
      },
      roadmap: {
        navContents: 'Contents',
        navCatalog: 'Catalog',
        navRoadmap: 'Roadmap',
        navGlossary: 'Glossary',
        navAbout: 'About',
        heroEyebrow: 'Curriculum navigation',
        heroTitle: 'From first principles to production AI.',
        heroLede: 'Twenty connected phases form one top-to-bottom dependency graph. Follow the route downward, illuminate the prerequisites behind any phase, and continue exactly where your local progress left off.',
        statPhases: 'Phases',
        statLessons: 'Lessons',
        statProgress: 'Your progress',
        statNext: 'Recommended next',
        zonesTitle: 'Four graph zones',
        zonesSub: 'Jump down the map',
        zonePrefix: 'ZONE',
        stageNames: {
          'foundations': 'Foundations',
          'model-disciplines': 'Model disciplines',
          'engineering-systems': 'Engineering systems',
          'capstone-proof': 'Capstone proof'
        },
        mapTitle: 'Interactive learning map',
        mapGuide: 'Select a node to illuminate its route. Click the selected node or blank space to return to the full graph.',
        keyboardHelp: 'Use the arrow keys to move between phase nodes. Press Enter or Space to select a phase. Press Escape to clear the route.',
        findPhaseLabel: 'Find a phase',
        zoomOut: 'Zoom out',
        zoomIn: 'Zoom in',
        backToFull: 'Back to full graph',
        legendPrereqs: 'Prerequisite route',
        legendUnlocks: 'Downstream unlocks',
        scrollHint: 'Scroll down to follow the route · drag sideways across branches',
        stateComplete: 'Complete',
        stateInProgress: 'In progress',
        stateReady: 'Ready',
        stateUpcoming: 'Upcoming',
        inspectorEmptyEyebrow: 'Route inspector',
        inspectorEmptyTitle: 'Choose a phase',
        inspectorEmptyCopy: 'Select a node to illuminate the exact route into it, every phase it unlocks, and the best lesson to continue from your local progress.',
        recNextLabel: 'Recommended next',
        inspectorYourProgress: 'Your progress',
        inspectorAllPrereqs: 'All prerequisites',
        inspectorPhasesUnlocked: 'Phases unlocked',
        inspectorDirectPrereqs: 'Direct prerequisites',
        inspectorDirectUnlocks: 'Immediately unlocks',
        startPointMsg: 'This is the starting point.',
        finalDestMsg: 'This is a final destination.',
        btnReviewPhase: 'Review phase',
        btnContinuePhase: 'Continue phase',
        btnStartPhase: 'Start phase',
        btnViewGithub: 'View phase on GitHub',
        dataLoadError: 'Roadmap data could not be loaded. Rebuild the site and refresh this page.',
        completeUpper: 'COMPLETE'
      },
      learningPaths: {
        pageTitle: 'AI Engineering Learning Paths - AI Engineering from Scratch',
        skipLink: 'Skip to content',
        navContents: 'Contents',
        navCatalog: 'Catalog',
        navRoadmap: 'Roadmap',
        navGlossary: 'Glossary',
        navAbout: 'About',
        heroEyebrow: '4 core paths · 6 career routes',
        heroTitle: 'AI Engineering Learning Paths',
        heroLede: 'Choose a core domain to build depth, or a career route that sequences the same lessons around the work you want to become capable of doing.',
        exploreByKnowledge: 'Explore by knowledge',
        browseFourPaths: 'Browse four core paths',
        exploreByOutcome: 'Explore by outcome',
        chooseCareerRoute: 'Choose a career route',
        overviewTitle: 'Four core paths. One discipline.',
        overviewCopy: 'Each domain opens a guided lesson sequence and the capabilities it develops. Every capability links to the closest practical lesson.',
        compareSixRoutes: 'Compare six career routes',
        rootTitle: 'AI Engineering',
        rootSubtitle: 'learn the system, the work, and the build',
        domain1Title: 'Building and Deploying AI Applications',
        domain1Sub: 'interfaces · grounding · evaluation · production',
        domain2Title: 'Software Engineering Fundamentals',
        domain2Sub: 'full stack · data · architecture · reliability · scale',
        domain3Title: 'Agent-Assisted Engineering',
        domain3Sub: 'frame · plan · delegate · verify · improve',
        domain4Title: 'Product Judgment and Delivery',
        domain4Sub: 'outcomes · evidence · risk · metrics · feedback',
        careerEyebrow: 'Career directions · shared curriculum',
        careerTitle: 'Choose by the work, not the title.',
        careerCopy: 'Titles vary between teams. Start with the problems you want to own, then inspect the responsibilities, baseline, evidence, and gaps before choosing specialist lessons.',
        readThisFirst: 'Read this first',
        truthNoteCopy: 'These routes are specialist overlays after shared foundations. Completion builds evidence of practice, but it does not guarantee a job. Displayed minutes are guided lesson time only. They exclude foundation work, independent projects, and professional experience.',
        step1Title: 'Engineering foundations',
        step1Desc: 'Full-stack boundaries, data, architecture, reliability, security, and production operations.',
        step2Title: 'AI application foundations',
        step2Desc: 'Model interfaces, grounding, evaluation, production behavior, and operations.',
        step3Title: 'Specialist practice',
        step3Desc: 'Choose a work family, close its baseline gaps, and produce role-shaped evidence.',
        chooserEyebrow: 'Decision prompts',
        chooserTitle: 'Which work would you want to repeat every week?',
        prompt1Sub: 'Work beside users and carry one workflow into production?',
        prompt1Title: 'Customer AI Deployment',
        prompt2Sub: 'Make APIs, examples, and technical learning easier for developers?',
        prompt2Title: 'Developer Experience and Education',
        prompt3Sub: 'Build the data, retrieval, and quality systems behind AI features?',
        prompt3Title: 'AI Data Systems',
        prompt4Sub: 'Engineer tool-using loops, memory, orchestration, and runtime controls?',
        prompt4Title: 'Agent Systems Engineering',
        prompt5Sub: 'Ship model-powered product behavior from interface to production?',
        prompt5Title: 'LLM Product Engineering',
        prompt6Sub: 'Measure failure, gate releases, and operate AI systems reliably?',
        prompt6Title: 'AI Evaluation and Reliability',
        openGuide: 'Open guide',
        closeGuide: 'Close guide',
        workFamily: 'Work family',
        guidedRouteLabel: 'Guided route',
        baselineLabel: 'Baseline',
        whatYouOwn: 'What you would own',
        fitAndBoundary: 'Fit and boundary',
        portfolioProof: 'Portfolio proof',
        coverageAndGaps: 'Course coverage and gaps',
        goodFitIf: 'Good fit if:',
        boundaryLabel: 'Boundary:',
        stillEarnedLabel: 'Still earned elsewhere:',
        studySpecialist: 'Study specialist lessons',
        openFullPath: 'Open full path',
        openLesson: 'Open lesson',
        openRepLesson: 'Open representative lesson',
        backToFourDomains: 'Back to four domains',
        compareFourDomains: 'Compare the four domains',
        followConnected: 'Follow every connected competency.',
        unsureFooter: 'Unsure which direction fits? Compare the responsibilities and evidence, then build the shared foundations before specializing.',
        footerCopy: 'AI Engineering from Scratch · open source · free forever.',
        footerHome: 'Home',
        footerRoadmap: 'Curriculum roadmap',
        footerCatalog: 'Catalog',
        careers: {
          'forward-deployed-ai-engineer': {
            title: 'Customer AI Deployment',
            aliases: 'Forward-Deployed AI Engineer · Field AI Engineer · AI Solutions Engineer',
            mission: 'Turn a real customer workflow into a small, measurable AI system, then stay close enough to the rollout to learn where it breaks.',
            route: '12 specialist lessons · 865 minutes',
            baseline: 'Software delivery plus AI application fundamentals',
            own: [
              'Observe the workflow, users, exceptions, and hidden handoffs.',
              'Reduce the request to the smallest useful end-to-end slice.',
              'Integrate grounding, evaluation, and production controls.',
              'Run a measured pilot and turn feedback into the next system change.'
            ],
            fit: 'you like ambiguous user problems, fast technical iteration, and shared ownership after launch.',
            boundary: 'this is not sales engineering or generic consulting. The proof is a working, measured system that you can operate.',
            portfolio: 'Ship a workflow dossier, a grounded prototype, an evaluation set, and a pilot plan as one evidence bundle.',
            evidence: ['Named assumptions and the riskiest test', 'Measured task quality and failure cases', 'Rollout, rollback, and feedback ownership'],
            coverage: 'The route covers discovery, risk, RAG, evaluation, production, metrics, rollout, and feedback.',
            stillEarned: 'customer domain expertise, stakeholder trust, procurement constraints, and ownership under live production pressure.',
            footer: 'Start after the shared foundation check. Keep every lesson output as evidence for the final bundle.'
          },
          'ai-developer-relations-engineer': {
            title: 'Developer Experience and Education',
            aliases: 'AI Developer Relations Engineer · AI Developer Advocate · Developer Experience Engineer',
            mission: 'Make an AI capability understandable, runnable, and trustworthy for developers, then feed their friction back into the product.',
            route: '11 specialist lessons · 905 minutes',
            baseline: 'Software fundamentals, API use, and clear technical writing',
            own: [
              'Build integrations and examples that survive a clean setup.',
              'Explain API, tool, protocol, and skill contracts precisely.',
              'Reproduce developer friction instead of guessing at it.',
              'Turn support signals into documentation, tooling, and product feedback.'
            ],
            fit: 'you enjoy building, teaching, debugging with other developers, and making difficult systems legible.',
            boundary: 'this is not content-only marketing. Credibility comes from runnable technical work and accurate explanations.',
            portfolio: 'Publish a developer onboarding package with a working integration, examples, a reusable agent package, and a friction report.',
            evidence: ['Fresh-environment setup evidence', 'Positive, negative, and failure examples', 'Feedback linked to a concrete improvement'],
            coverage: 'The route covers APIs, tool contracts, MCP, Agent Skills, packaging, evaluation, and feedback.',
            stillEarned: 'live audience practice, community judgment, adoption analytics, editorial depth, and sustained developer support.',
            footer: 'Use the lessons to produce one coherent onboarding experience, not a pile of disconnected demos.'
          },
          'ai-data-engineer': {
            title: 'AI Data Systems',
            aliases: 'AI Data Engineer · Machine Learning Data Engineer · Retrieval Engineer',
            mission: 'Build the data and retrieval pipelines that let training, evaluation, and production AI behavior use trustworthy evidence.',
            route: '11 specialist lessons · 915 minutes',
            baseline: 'Python, data structures, statistics, and pipeline fundamentals',
            own: [
              'Ingest, transform, version, and validate training or retrieval data.',
              'Build embedding, indexing, retrieval, and evaluation pipelines.',
              'Define data quality checks and investigate silent drift.',
              'Expose lineage, freshness, cost, and runtime health.'
            ],
            fit: 'you enjoy pipelines, data quality, reproducibility, and debugging systems that fail far from the user interface.',
            boundary: 'this route focuses on AI data products. It does not replace the broader warehouse, database, and platform depth of data engineering.',
            portfolio: 'Ship a versioned document-to-retrieval pipeline with quality gates, evaluation data, and an operational report.',
            evidence: ['Reproducible ingestion and lineage', 'Retrieval quality and freshness measures', 'Failure recovery and observability evidence'],
            coverage: 'The route covers data management, features, pipelines, embeddings, context, RAG, evaluation, production, and observability.',
            stillEarned: 'advanced SQL, warehouse architecture, governance, privacy operations, and large-scale distributed data systems.',
            footer: 'Close the statistical and data-engineering baseline before treating the specialist sequence as ordered work.'
          },
          'agentic-ai-engineer': {
            title: 'Agent Systems Engineering',
            aliases: 'Agent Systems Engineer · Agentic AI Engineer · AI Agent Engineer',
            mission: 'Engineer the runtime around a tool-using model so context, memory, authority, orchestration, failure, and evidence remain explicit.',
            route: '14 specialist lessons · 865 minutes',
            baseline: 'LLM application foundations plus typed tool interfaces',
            own: [
              'Design tool contracts and the observe, decide, act loop.',
              'Control context, memory, state, and durable execution.',
              'Choose orchestration boundaries and termination policy.',
              'Threat-model authority and evaluate complete trajectories.',
              'Operate the runtime with traces and explicit failure controls.'
            ],
            fit: 'you enjoy runtime design, state machines, distributed coordination, safety boundaries, and difficult failure analysis.',
            boundary: 'this is systems engineering around model behavior, not a promise that adding an agent loop makes a product autonomous.',
            portfolio: 'Ship a bounded tool-using runtime with memory, orchestration, a threat model, trajectory evals, and a failure runbook.',
            evidence: ['Deterministic tool and state traces', 'Permission, sandbox, and injection controls', 'Termination, recovery, and evaluation evidence'],
            coverage: 'The route covers tools, MCP, loops, context, memory, graphs, orchestration, security, evaluation, runtimes, and observability.',
            stillEarned: 'provider-specific infrastructure, high-scale distributed operation, latency engineering, and production ownership with a team.',
            footer: 'Complete the LLM and tool-interface baseline first. Then treat every agent claim as something the runtime must prove.'
          },
          'applied-ai-engineer': {
            title: 'LLM Product Engineering',
            aliases: 'Applied AI Engineer · LLM Engineer · AI Product Engineer',
            mission: 'Turn model capability into useful product behavior that is grounded, evaluated, guarded, cost-aware, and recoverable in production.',
            route: '12 specialist lessons · 885 minutes',
            baseline: 'Software engineering plus LLM foundations',
            own: [
              'Design model-facing interfaces and structured contracts.',
              'Ground behavior with context, retrieval, and tools.',
              'Build task evaluations before optimizing the feature.',
              'Control safety, cost, latency, caching, and fallbacks.',
              'Release the complete feature with observable behavior.'
            ],
            fit: 'you want to connect product needs to model behavior and own the software around the model.',
            boundary: 'this is not foundation-model research or model training. The work begins where a model capability meets a real product constraint.',
            portfolio: 'Ship a grounded product feature with structured output, tools, an eval set, cost and latency budgets, and a guarded release.',
            evidence: ['Representative success and failure cases', 'Quality, cost, and latency tradeoffs', 'Fallback, release, and rollback evidence'],
            coverage: 'The route covers prompting, structured output, embeddings, context, RAG, tools, evaluation, cost, guardrails, production, gateways, and release.',
            stillEarned: 'product discovery, interaction design, real user research, domain regulation, and operating a feature under sustained traffic.',
            footer: 'Start only after the LLM foundation check. The portfolio proof is the integrated behavior, not twelve completed pages.'
          },
          'ai-evaluation-reliability-engineer': {
            title: 'AI Evaluation and Reliability',
            aliases: 'AI Evaluation Engineer · AI Reliability Engineer · Machine Learning Site Reliability Engineer',
            mission: 'Make model and agent behavior measurable, expose failure before release, and build operational controls for what still fails in production.',
            route: '12 specialist lessons · 750 minutes',
            baseline: 'Statistics, software testing, and production systems',
            own: [
              'Define evaluation sets, metrics, graders, and failure taxonomies.',
              'Instrument model, agent, and serving behavior.',
              'Build release gates, experiments, and regression detection.',
              'Test load, degradation, recovery, and incident response.',
              'Connect evidence to rollout and operational decisions.'
            ],
            fit: 'you enjoy statistics, adversarial testing, observability, release judgment, and learning from incidents.',
            boundary: 'this is broader than offline model accuracy. Reliability includes the application, runtime, infrastructure, and response process.',
            portfolio: 'Ship a behavioral evaluation harness connected to traces, a release gate, a load or failure experiment, and an incident runbook.',
            evidence: ['Versioned cases and metric rationale', 'Regression and rollout decisions', 'Observed recovery and residual risk'],
            coverage: 'The route covers model, LLM, and agent evaluation, observability, serving metrics, experiments, load, canary release, chaos, and SRE.',
            stillEarned: 'real on-call experience, organization-specific incident process, production traffic, compliance evidence, and cross-team release authority.',
            footer: 'Close the statistics and production baseline first. Treat every release gate as a decision backed by evidence.'
          }
        },
        domains: {
          'building-and-deploying': {
            number: 'Domain 01 · application systems',
            title: 'Building and Deploying AI Applications',
            copy: 'Move from the first model-facing interface to grounded behavior, evaluation, safeguards, and production operation. The application is the whole system around the model.',
            root: 'Build the complete AI application',
            meta: '12-lesson path · 780 minutes',
            nodes: [
              { index: '01 · interface', title: 'Model Interaction Contracts', desc: 'Turn intent into a bounded request with explicit inputs, outputs, and failure behavior.' },
              { index: '02 · contracts', title: 'Structured Generation Contracts', desc: 'Make generated data parseable, validated, and safe to pass into application code.' },
              { index: '03 · grounding', title: 'Evidence Representation', desc: 'Represent, retrieve, and place evidence where the model can use it.' },
              { index: '04 · retrieval', title: 'Retrieval and Freshness', desc: 'Build the ingestion, search, ranking, citation, and freshness loop around generation.' },
              { index: '05 · evidence', title: 'Behavioral Evaluation Gates', desc: 'Define acceptable behavior, collect cases, score outcomes, and gate regressions.' },
              { index: '06 · operation', title: 'Serving and Recovery', desc: 'Serve, observe, release, recover, and control cost under real traffic.' }
            ]
          },
          'software-fundamentals': {
            number: 'Domain 02 · engineering substrate',
            title: 'Software Engineering Fundamentals',
            copy: 'Coding agents reduce typing, not engineering judgment. Learn to steer tradeoffs across the application stack, data, architecture, security, reliability, and production operations.',
            root: 'Steer the software tradeoffs behind AI systems',
            meta: '13-lesson foundation path · 730 minutes',
            footerNote: 'Use the foundation path for sequence, or enter through the representative branch that matches the capability you need now.',
            nodes: [
              { index: '01 · application', title: 'End-to-End Application Delivery', desc: 'Connect request handling, streaming, persistence, fallbacks, health checks, and deployment into one working system.' },
              { index: '02 · data', title: 'Data Lifecycle and Storage', desc: 'Choose representations, validation, versioning, retention, and freshness from the access patterns the application needs.' },
              { index: '03 · architecture', title: 'System Architecture and Boundaries', desc: 'Design one explicit boundary: inputs, outputs, errors, permissions, and state before a capability enters a larger system.' },
              { index: '04 · assurance', title: 'Secure and Resilient Systems', desc: 'Audit secrets, permissions, dependencies, data handling, and release evidence before production.' },
              { index: '05 · operation', title: 'Production Scale and Service Ownership', desc: 'Define service objectives, watch health signals, prepare runbooks, and practice evidence-based incident response.' }
            ]
          },
          'coding-agents': {
            number: 'Domain 03 · agent-assisted engineering',
            title: 'Agent-Assisted Engineering',
            copy: 'A coding agent is useful when the task, context, tools, feedback, and stop condition form a dependable harness. Learn to shape that system around real repository work.',
            root: 'Turn model capability into dependable work',
            meta: '16-lesson path · 900 minutes',
            nodes: [
              { index: '01 · frame', title: 'Task Framing', desc: 'Turn a request into scope, constraints, permissions, evidence, and a stopping rule.' },
              { index: '02 · plan', title: 'Evidence-Based Planning', desc: 'Inspect the repository before proposing the smallest coherent change.' },
              { index: '03 · harness', title: 'Agent Workbench', desc: 'Engineer the loop, context boundary, tools, transcript, and termination policy.' },
              { index: '04 · context', title: 'Instructions and Memory', desc: 'Place durable guidance at the correct scope and keep runtime state observable.' },
              { index: '05 · feedback', title: 'Runtime Feedback', desc: 'Feed compiler, test, browser, and wire evidence back into the next decision.' },
              { index: '06 · verify', title: 'Verification and Review', desc: 'Prove the requested behavior independently of the agent\'s own completion claim.' },
              { index: '07 · delegate', title: 'Isolated Delegation', desc: 'Split bounded work across agents without sharing ambiguous ownership or state.' },
              { index: '08 · improve', title: 'Durable Improvement', desc: 'Convert corrections into tests, instructions, tooling, and reusable constraints.' }
            ]
          },
          'shaping-the-build': {
            number: 'Domain 04 · product judgment',
            title: 'Product Judgment and Delivery',
            copy: 'Before implementation, decide what outcome matters, what evidence supports the work, which risk deserves attention, and how you will know the change helped.',
            root: 'Choose the right build before producing output',
            meta: '8-lesson path · 550 minutes',
            nodes: [
              { index: '01 · outcome', title: 'Outcomes Before Output', desc: 'Define the changed state you want before discussing features or implementation.' },
              { index: '02 · observe', title: 'Workflow Discovery', desc: 'Study how the work happens now, including exceptions, handoffs, and hidden labor.' },
              { index: '03 · risk', title: 'Assumptions and Risk', desc: 'Expose what must be true and test the uncertainty that could invalidate the build.' },
              { index: '04 · slice', title: 'Testable Slices', desc: 'Choose the smallest end-to-end change that can produce decision-quality evidence.' },
              { index: '05 · specify', title: 'Executable Specifications', desc: 'Make constraints and acceptance observable without removing implementation judgment.' },
              { index: '06 · measure', title: 'Success Metrics', desc: 'Connect product outcomes to leading, guardrail, and operational measures.' },
              { index: '07 · stage', title: 'Release Strategy', desc: 'Match prototype, pilot, or production investment to the evidence you need next.' },
              { index: '08 · own', title: 'Feedback Ownership', desc: 'Assign who reads the signal, makes the decision, and changes the system.' }
            ]
          }
        }
      },
      glossary: {
        navContents: 'Contents',
        navCatalog: 'Catalog',
        navRoadmap: 'Roadmap',
        navGlossary: 'Glossary',
        navAbout: 'About',
        kicker: 'Reference ledger · curriculum v1.0',
        title: 'AI Engineering Glossary',
        deck1: 'Precise working definitions for the systems you build. Start with the meaning, then use the examples, distinctions, and lesson links to turn vocabulary into judgment.',
        deck2: 'Each entry separates a term\'s working definition from why it matters in an AI system, how it appears in practice, and what engineers commonly confuse it with. Search by term, alias, or idea; narrow the ledger by learning area; then follow the linked course lesson or primary source when you need the full derivation. Every entry has a stable fragment link, so you can cite one definition directly without sending someone through the entire index.',
        statTerms: 'Terms indexed',
        statCategories: 'Learning areas',
        statDeepLinks: 'Stable deep links',
        searchLabel: 'Search the ledger',
        searchPlaceholder: 'Term, alias, or idea',
        clearBtn: 'Clear',
        learningAreaLabel: 'Learning area',
        jumpToLetterLabel: 'Jump to letter',
        liveReferenceLabel: 'Live reference',
        resultsHeading: 'Reference entries',
        loadingEntries: 'Loading reference entries...',
        showMore: 'Show more',
        showAll: 'Show all',
        whyItMatters: 'Why it matters',
        inPractice: 'In practice',
        notToBeConfusedWith: 'Not to be confused with',
        relatedTerms: 'Related terms',
        courseLessons: 'Course lessons',
        primarySources: 'Primary sources',
        copyLink: 'Copy link',
        linkCopied: 'Copied',
        readAloud: 'Read aloud',
        readingAloud: 'Reading...',
        stopReading: 'Stop',
        emptyTitle: 'No glossary entries found',
        emptyCopy: 'Try a different search term or reset your filters.',
        resetFilters: 'Reset filters',
        workingDefinition: 'Working definition',
        whyCalled: 'Why it is called this',
        commonShortcut: 'Common shortcut',
        alsoCalled: 'Also called',
        distinctionsAndEvidence: 'Distinctions and evidence',
        allTerms: 'All terms'
      },
      about: {
        navContents: 'Contents',
        navCatalog: 'Catalog',
        navRoadmap: 'Roadmap',
        navGlossary: 'Glossary',
        navAbout: 'About',
        eyebrow: 'About',
        title: 'About this project',
        lede: 'AI Engineering from Scratch is a free, open-source curriculum that builds every core AI algorithm by hand. The curriculum spans <span id="aboutLessonCount">every published lesson</span> across 20 phases, from linear algebra to autonomous agents, in Python, TypeScript, Rust, and Julia.',
        whyTitle: 'Why it exists',
        whyP1: 'Most AI material teaches in scattered pieces. A paper here, a fine-tuning post there, a framework demo somewhere else. You can ship a chatbot without being able to explain its loss curve, or wire a tool to an agent without knowing what attention does inside the model calling it.',
        whyP2: 'This curriculum is the spine. Every algorithm gets written from raw math first, then run through the production library so you can see what the library was doing. By the time PyTorch shows up, you already know what it computes. Each lesson ends with a reusable artifact you keep: a prompt, a skill, an agent, or an MCP server.',
        howTitle: 'How it is made',
        howP1: 'The lessons are authored with AI assistance and reviewed by a human against primary sources. Where a lesson states a fact, it cites the original: an RFC, a spec, or a research paper, not a secondary summary. Corrections are welcome and tracked in the open on GitHub.',
        howP2: 'The site itself is deliberately plain: hand-written HTML, CSS, and vanilla JavaScript, no framework. A single build script (<code>site/build.js</code>) reads the lesson Markdown in the repository and generates the catalog, search index, sitemap, and <code>llms.txt</code> on every deploy. Public curriculum counts are read from that generated data. It is hosted on Vercel.',
        whoTitle: 'Who builds it',
        whoP1: 'Maintained by <a href="https://github.com/rohitg00" target="_blank" rel="noopener">Rohit Ghumare</a> and contributors. It is MIT-licensed and free forever. There is no token, no course upsell, and no gated content.',
        koreanTitle: 'Korean Edition & AX Academy',
        koreanP1: 'Localized and maintained for Korean developers by <strong>AX Academy</strong> to share practical, first-principles AI engineering knowledge with the community.',
        involvedTitle: 'Get involved',
        involved1: 'Read the source: <a href="https://github.com/rohitg00/ai-engineering-from-scratch" target="_blank" rel="noopener">github.com/rohitg00/ai-engineering-from-scratch</a>',
        involved2: 'Found an error or have a lesson idea? <a href="https://github.com/rohitg00/ai-engineering-from-scratch/issues/new/choose" target="_blank" rel="noopener">Open an issue</a>.',
        involved3: 'Start learning: <a href="catalog.html">browse the catalog</a> or <a href="prereqs.html">follow the roadmap</a>.'
      },
      catalog: {
        pageTitle: 'Lesson Catalog - AI Engineering from Scratch',
        skipLink: 'Skip to content',
        navContents: 'Contents',
        navCatalog: 'Catalog',
        navRoadmap: 'Roadmap',
        navGlossary: 'Glossary',
        navAbout: 'About',
        title: 'Lesson Catalog',
        subtitle: 'Every lesson across all 20 phases. Search, filter, sort.',
        searchPlaceholder: 'Search lessons...',
        searchAria: 'Search lessons',
        phaseFilterAria: 'Filter by phase',
        statusFilterAria: 'Filter by lesson status',
        allPhases: 'All Phases',
        allStatus: 'All Status',
        statusComplete: 'Complete',
        statusPlanned: 'Planned',
        loading: 'Loading lesson index...',
        noscript: 'JavaScript is needed for local filtering. Every lesson remains available in the <a href="https://github.com/rohitg00/ai-engineering-from-scratch#contents">GitHub curriculum index</a>.',
        colPhase: 'Phase',
        colLesson: 'Lesson',
        colType: 'Type',
        colLanguage: 'Language',
        colStatus: 'Status',
        showing: 'Showing {shown} of {matched} matching lessons. {total} total.',
        empty: 'No lessons match your filters.',
        loadError: 'The lesson index could not be loaded.',
        loadErrorHint: 'Reload the page or use the GitHub curriculum index.',
        phaseGroup: 'Phase {id}: {name}',
        lessonCount: '{n} lessons',
        remaining: '{n} matching lessons remain.',
        showMore: 'Show more',
        showNext: 'Show next {n}',
        showAll: 'Show all',
        typeLearn: 'Learn',
        typeBuild: 'Build',
        typeCapstone: 'Capstone',
        typeReference: 'Reference',
        footerCopy: 'AI Engineering from Scratch · open source · free forever.',
        footerHome: 'Home',
        footerGlossary: 'Glossary',
        footerReport: 'Report / Suggest'
      }
    }
  };

  root.AIFS_UI_I18N = UI_I18N;
})(typeof window !== 'undefined' ? window : this);

(function (root) {
  'use strict';

  var TRANSLATIONS_BASE = 'https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/translations/i18n/';
  var ATTRS = ['aria-label', 'title', 'placeholder'];
  var SKIP_TAGS = { SCRIPT: 1, STYLE: 1, CODE: 1, PRE: 1, KBD: 1, SAMP: 1, TEXTAREA: 1, NOSCRIPT: 1, svg: 1, SVG: 1, MATH: 1 };
  var SKIP_SELECTOR = '.lang-picker, .mermaid-render, .mermaid-modal-body, .quiz-question-text, .quiz-option-text, .quiz-explanation, .nav-title, .sidebar-lesson-link, .toc-nav, [data-i18n-skip]';
  var ARTICLE_SELECTOR = '.lesson-article';
  var ARTICLE_ALLOW_SELECTOR = '.lesson-action-panel, .lesson-action-path, .ai-panels, .quiz-section, .lesson-nav-bottom, .continue-callout, .cert-notice';
  var RTL = { ar: 1, he: 1, fa: 1, ur: 1 };

  var records = typeof WeakMap === 'function' ? new WeakMap() : null;
  var dictionaries = {};
  var pending = {};
  var active = 'en';
  var touched = false;
  var request = 0;
  var observer = null;

  function preload(lang, dict) {
    var table = dict && typeof dict === 'object' && dict.strings && typeof dict.strings === 'object' ? dict.strings : dict;
    if (table && typeof table === 'object') dictionaries[lang] = table;
    else delete dictionaries[lang];
  }

  function dictionaryFor(lang) {
    if (!lang || lang === 'en') return null;
    var dict = dictionaries[lang];
    return dict && typeof dict === 'object' ? dict : null;
  }

  function loadDictionary(lang, done) {
    if (!lang || lang === 'en' || Object.prototype.hasOwnProperty.call(dictionaries, lang)) {
      done(dictionaryFor(lang));
      return;
    }
    if (pending[lang]) {
      pending[lang].push(done);
      return;
    }
    pending[lang] = [done];
    root.fetch(TRANSLATIONS_BASE + encodeURIComponent(lang) + '/ui.json')
      .then(function (response) {
        if (!response.ok) throw new Error('missing');
        return response.json();
      })
      .then(function (json) { preload(lang, json); }, function () {})
      .then(function () {
        var callbacks = pending[lang] || [];
        delete pending[lang];
        for (var i = 0; i < callbacks.length; i++) callbacks[i](dictionaryFor(lang));
      });
  }

  function translateText(text, dict) {
    if (!dict) return text;
    var source = String(text);
    var lead = source.match(/^\s*/)[0];
    var core = source.slice(lead.length);
    var trail = core.match(/\s*$/)[0];
    core = core.slice(0, core.length - trail.length);
    if (!core) return source;
    var key = core.replace(/\s+/g, ' ');
    if (!Object.prototype.hasOwnProperty.call(dict, key)) return source;
    return lead + dict[key] + trail;
  }

  function matches(el, selector) {
    return !!(el && el.nodeType === 1 && typeof el.matches === 'function' && el.matches(selector));
  }

  function eligible(el) {
    var inArticle = false;
    var allowedInside = false;
    for (var node = el; node && node.nodeType === 1; node = node.parentNode) {
      if (SKIP_TAGS[node.nodeName] || matches(node, SKIP_SELECTOR)) return false;
      if (matches(node, ARTICLE_ALLOW_SELECTOR)) allowedInside = true;
      if (matches(node, ARTICLE_SELECTOR)) inArticle = true;
    }
    return !inArticle || allowedInside;
  }

  function record(node) {
    if (!records) return null;
    var rec = records.get(node);
    if (!rec) {
      rec = { text: null, attrs: {} };
      records.set(node, rec);
    }
    return rec;
  }

  function applyText(node, dict) {
    var rec = record(node);
    if (!rec) return;
    var current = node.nodeValue;
    if (!rec.text || current !== rec.text.out) rec.text = { orig: current, out: current };
    var out = dict ? translateText(rec.text.orig, dict) : rec.text.orig;
    if (out !== current) node.nodeValue = out;
    rec.text.out = out;
  }

  function applyAttr(el, name, dict) {
    if (!el.hasAttribute(name)) return;
    var rec = record(el);
    if (!rec) return;
    var current = el.getAttribute(name);
    var slot = rec.attrs[name];
    if (!slot || current !== slot.out) slot = rec.attrs[name] = { orig: current, out: current };
    var out = dict ? translateText(slot.orig, dict) : slot.orig;
    if (out !== current) el.setAttribute(name, out);
    slot.out = out;
  }

  function applyElement(el, dict) {
    if (!eligible(el)) return;
    for (var i = 0; i < ATTRS.length; i++) applyAttr(el, ATTRS[i], dict);
  }

  function applyNode(node, dict) {
    if (node.nodeType === 3) {
      if (eligible(node.parentNode)) applyText(node, dict);
    } else if (node.nodeType === 1) {
      applyElement(node, dict);
    }
  }

  function applyTree(rootNode, dict) {
    if (!rootNode) return;
    applyNode(rootNode, dict);
    if (rootNode.nodeType !== 1 && rootNode.nodeType !== 9 && rootNode.nodeType !== 11) return;
    var doc = rootNode.ownerDocument || rootNode;
    var walker = doc.createTreeWalker(rootNode, 5, null, false);
    var node;
    while ((node = walker.nextNode())) applyNode(node, dict);
  }

  function applyDir(lang) {
    if (typeof root.AIFS_applyLangDir === 'function') {
      root.AIFS_applyLangDir(lang);
      return;
    }
    root.document.documentElement.lang = lang;
    root.document.documentElement.dir = RTL[lang] ? 'rtl' : 'ltr';
  }

  function setLanguage(lang) {
    var sequence = ++request;
    loadDictionary(lang, function (dict) {
      if (sequence !== request) return;
      active = dict ? lang : 'en';
      if (!dict && !touched) return;
      touched = true;
      applyTree(root.document.body, dict);
      applyDir(active);
      observe();
    });
  }

  function observe() {
    if (observer || typeof MutationObserver !== 'function') return;
    observer = new MutationObserver(function (mutations) {
      var dict = dictionaryFor(active);
      if (!dict) return;
      for (var i = 0; i < mutations.length; i++) {
        var m = mutations[i];
        if (m.type === 'childList') {
          for (var j = 0; j < m.addedNodes.length; j++) applyTree(m.addedNodes[j], dict);
        } else if (m.type === 'characterData') {
          if (eligible(m.target.parentNode)) applyText(m.target, dict);
        } else if (m.type === 'attributes') {
          if (eligible(m.target)) applyAttr(m.target, m.attributeName, dict);
        }
      }
    });
    observer.observe(root.document.documentElement, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ATTRS
    });
  }

  function currentLang() {
    if (typeof root.AIFS_currentLang === 'function') return root.AIFS_currentLang();
    var fromQuery = '';
    try { fromQuery = new URLSearchParams(root.location.search).get('lang') || ''; } catch (_) {}
    if (fromQuery) return fromQuery;
    try { return root.localStorage.getItem('lang') || 'en'; } catch (_) { return 'en'; }
  }

  function start() {
    if (root.AIFS_UI_STRINGS && typeof root.AIFS_UI_STRINGS === 'object') {
      for (var lang in root.AIFS_UI_STRINGS) {
        if (Object.prototype.hasOwnProperty.call(root.AIFS_UI_STRINGS, lang)) preload(lang, root.AIFS_UI_STRINGS[lang]);
      }
    }
    root.document.addEventListener('aifs:lang', function (event) {
      setLanguage(event.detail && event.detail.lang);
    });
    if (root.document.readyState === 'loading') {
      root.document.addEventListener('DOMContentLoaded', function () { setLanguage(currentLang()); }, { once: true });
    } else {
      setLanguage(currentLang());
    }
  }

  var api = {
    translateText: translateText,
    dictionaryFor: dictionaryFor,
    loadDictionary: loadDictionary,
    preload: preload,
    currentLang: currentLang,
    setLanguage: setLanguage,
    TRANSLATIONS_BASE: TRANSLATIONS_BASE,
    ATTRS: ATTRS,
    SKIP_SELECTOR: SKIP_SELECTOR,
    ARTICLE_ALLOW_SELECTOR: ARTICLE_ALLOW_SELECTOR
  };
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.AIFSUiI18n = api;

  if (root.document && typeof root.document.createElement === 'function' && typeof root.fetch === 'function') start();
})(typeof window !== 'undefined' ? window : globalThis);
