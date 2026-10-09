# 작업 프레임: 가입 시 중복 이메일 주소 방지

상태: READY

## 저장소 사실
- 계정 쓰기는 AccountStore (`app/accounts.py:18`)를 사용합니다
- 중복 오류는 상태 코드 409 (`tests/test_accounts.py:44`)를 사용합니다

## 허용 경로
- `app/accounts.py`
- `tests/test_accounts.py`

## 금지 경로
- `migrations/**`
- `deploy/**`

## 수락 근거
- `python3 -m unittest tests.test_accounts`

## 미확인 사항
- 이메일 비교가 대소문자를 구분하지 않는지 여부
