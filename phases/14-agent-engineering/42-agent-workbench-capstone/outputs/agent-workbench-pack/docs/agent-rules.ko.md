# 에이전트 규칙

## startup/state-file-fresh
- 카테고리: startup
- 검사: state_file_fresh
에이전트는 모든 도구 호출 전에 agent_state.json을 읽어야 합니다.

## forbidden/no-out-of-scope-writes
- 카테고리: forbidden
- 검사: no_out_of_scope_writes
활성 작업의 범위 계약(scope contract) 밖의 파일을 절대 편집하지 마세요.

## done/tests-pass
- 카테고리: definition_of_done
- 검사: tests_pass
모든 수용 기준 명령이 0으로 종료될 때만 작업이 완료됩니다.

## uncertainty/open-question-note
- 카테고리: uncertainty
- 검사: opened_question_when_unsure
확신이 임계값 미만일 때는 추측하지 말고 질문 노트를 여세요.

## approval/new-dependency
- 카테고리: approval
- 검사: new_dependency_approved
런타임 의존성 추가는 명시적인 인간 승인이 필요합니다.
