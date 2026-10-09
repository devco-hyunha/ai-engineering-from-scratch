---
name: skill-end-to-end-safety-gate
description: 입력 감지자, 스트리밍 토큰 필터, 출력 분류기, 규칙 엔진을 조합하여 결정론적 집계 테이블과 요청별 추적을 포함하는 3단계 체크포인트 안전 게이트
version: 1.0.0
phase: 19단계
lesson: 87강
tags: [safety, harness, composition]
---

# 엔드투엔드 안전 게이트

## 수명주기

1. 생성 전 - 프롬프트에 대해 87강의 감지자를 실행합니다
   - 신뢰도 >= block_threshold인 경우: 거절 응답을 반환하고, 추적을 방출하며, 중단합니다
2. 생성 중 - 모델로부터 스트리밍하며, 두 청크를 버퍼링하고, 알려진 유해한 연속성을 스캔합니다
   - 일치하는 경우: 반복자를 종료하고, 추적을 표시하며, 중간 심각도로 취급합니다
3. 생성 후 - 조기 종료되지 않은 경우, 완료된 출력에 대해 85강의 분류기 라우터와 86강의 규칙 엔진을 실행합니다
4. 집계 - 생성 전, 생성 중, 생성 후 분류기, 생성 후 규칙의 최대 심각도를 취합니다
5. 적용 - 차단, 편집, 경고, 허용에 매핑합니다

## 집계 테이블

| 신호 상태 | 조치 |
|---|---|
| 높은 심각도 | 차단 |
| 중간 심각도 | 편집 |
| 낮은 심각도 | 경고 |
| 없음 | 허용 |

## 추적 구조

```text
RequestTrace
  request_id: str
  prompt: str
  pre_gen: { category, confidence, fired[] }
  during_gen: { terminated_early, matched_pattern, partial_chunks }
  post_gen: { classifier_action, classifier_severity, rules_max_severity, rules_violations[] } | null
  final_action: block | redact | warn | allow
  final_output: str
  latency_ms: float
```

## 산출물

`outputs/gate_trace.json`에는 요약과 요청별 추적 한 개가 포함되며, 50개의 분류 항목 픽스처와 10개의 무해한 프롬프트가 포함됩니다.
