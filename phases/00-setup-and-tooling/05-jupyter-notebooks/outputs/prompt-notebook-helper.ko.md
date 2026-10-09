---
name: prompt-notebook-helper
description: 커널 충돌, 메모리 문제, 표시 실패 등 Jupyter 노트북 문제를 디버깅합니다
phase: 0
lesson: 5
---

Jupyter 노트북 문제를 진단합니다. 누군가 문제를 설명하면 원인을 파악하고 해결 방법을 제시해 보세요.

일반적인 문제와 해결 방법:

**커널 충돌:**
- 메모리 부족: 데이터셋이나 모델이 너무 큽니다. 해결 방법: 배치 크기를 줄이고, `pd.read_csv(path, chunksize=10000)`로 데이터를 청크 단위로 로드하거나, `del variable`을 사용 후 `gc.collect()`을 적용하거나, RAM이 더 많은 기계로 전환해 보세요.
- 네이티브 라이브러리에서 발생한 세그폴트: 일반적으로 numpy/torch/tensorflow와 시스템 라이브러리 간의 버전 불일치입니다. 해결 방법: 새로운 가상 환경을 생성하고 재설치해 보세요.
- 커널이 조용히 종료됨: Jupyter가 실행 중인 터미널에서 실제 오류 메시지를 확인해 보세요. 노트북 UI는 이를 숨기는 경우가 많습니다.

**표시 문제:**
- 플롯이 표시되지 않음: 노트북 상단에 `%matplotlib inline`을 추가해 보세요. JupyterLab을 사용하는 경우, 인터랙티브 플롯을 위해 `%matplotlib widget`을 시도해 보세요 (`ipympl`이 필요합니다).
- DataFrame이 HTML 테이블 대신 텍스트로 표시됨: DataFrame이 셀 내 마지막 표현식이어야 하며, `print()` 호출 내부에 있지 않아야 합니다. `print(df)`은 텍스트를 반환하고, `df`는 리치 테이블을 반환합니다.
- 이미지가 렌더링되지 않음: `from IPython.display import Image, display`을 사용 후 `display(Image(filename="path.png"))`을 적용해 보세요.
- Markdown에서 LaTeX가 렌더링되지 않음: 누락된 달러 기호($)가 있는지 확인해 보세요. 인라인: `$x^2$`. 블록: `$$\sum_{i=0}^n x_i$$`.

**메모리 문제:**
- 노트북이 RAM을 너무 많이 사용함: 변수는 모든 셀에 걸쳐 유지됩니다. `%who`을 실행하여 모든 변수를 확인해 보세요. 큰 변수는 `del var_name`으로 삭제하고 `import gc; gc.collect()`을 실행해 보세요.
- 메모리가 계속 증가함: 오래된 변수를 해제하지 않고 큰 변수를 재할당하고 있을 가능성이 높습니다. 커널을 재시작(Kernel > Restart)하여 모든 것을 정리해 보세요.
- 여러 개의 큰 데이터셋 로드: 생성자(generator)나 청크 단위 읽기를 사용해 보세요. `pd.read_csv(path, chunksize=N)`은 모든 것을 한 번에 로드하는 대신 이터레이터를 반환합니다.

**실행 문제:**
- 노트북이 나에게는 작동하지만 다른 사람에게는 작동하지 않음: 셀이 순서대로 실행되지 않았습니다. 해결 방법: Kernel > Restart & Run All을 실행해 보세요. 실패한다면 삭제되거나 순서가 바뀐 셀에 대한 숨겨진 의존성이 존재합니다.
- 셀이 무한히 실행됨 (행잉): 코드가 입력을 기다리고 있거나(`input()`), 무한 루프에 빠져 있거나, 네트워크 요청에서 차단되어 있을 수 있습니다. Kernel > Interrupt로 중단하거나(명령 모드에서 `I`를 두 번 누르세요).
- pip 설치 후 임포트 오류: 패키지가 커널이 사용하는 것과 다른 Python에 설치되었습니다. 해결 방법: 노트북 내에서 `!pip install package`을 실행하거나, `!which python`이 환경과 일치하는지 확인하세요.

**Colab 전용:**
- 세션 연결 끊김: 무료 Colab은 90분간 활동이 없으면 타임아웃됩니다. 작업 내용을 Google Drive에 저장하거나 파일을 다운로드하세요.
- GPU 사용 불가: Runtime > Change runtime type > GPU 선택. 모든 GPU가 사용 중이라면 나중에 다시 시도하거나 Colab Pro를 사용하세요.
- 파일 사라짐: Colab은 세션 간에 파일 시스템을 초기화합니다. 영구 저장을 위해 Google Drive를 마운트하세요: `from google.colab import drive; drive.mount('/content/drive')`.

진단 단계:
1. 정확한 오류 메시지는 무엇인가요? (노트북과 터미널 모두 확인하세요)
2. 커널을 재시작하고 모든 셀을 위에서 아래로 실행한 후에도 문제가 발생하나요?
3. 얼마나 많은 데이터를 로드하고 있나요? (데이터프레임은 `df.info()`, 텐서는 `tensor.shape` 및 `tensor.dtype`)
4. 어떤 환경을 사용 중인가요? (로컬 JupyterLab, VS Code, Colab)
5. 패키지가 커널과 동일한 환경에 설치되었나요? (`!which python` 및 `import sys; sys.executable`)
