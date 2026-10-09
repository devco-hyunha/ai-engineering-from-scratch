# 데이터 관리

> 데이터는 연료입니다. 데이터를 어떻게 관리하느냐가 진행 속도를 결정합니다.

**유형:** Build
**언어:** Python
**선수 요건:** 0단계, 01강
**시간:** 약 45분

## 학습 목표

- Hugging Face `datasets` 라이브러리를 사용하여 데이터셋을 로드, 스트리밍 및 캐싱해 보세요
- CSV, JSON, Parquet, Arrow 형식 간에 변환하고 각 형식의 장단점을 설명해 보세요
- 고정된 랜덤 시드를 사용하여 재현 가능한 학습/검증/테스트 분할을 생성해 보세요
- `.gitignore`, Git LFS 또는 DVC를 사용하여 대형 모델 및 데이터셋 파일을 관리해 보세요

## 문제점

모든 AI 프로젝트는 데이터로 시작합니다. 데이터셋을 찾고, 다운로드하고, 형식 간에 변환하고, 학습 및 평가를 위해 분할하고, 실험이 재현 가능하도록 버전 관리해야 합니다. 매번 수동으로 이 작업을 수행하는 것은 느리고 오류가 발생하기 쉽습니다. 반복 가능한 워크플로우가 필요합니다.

## 개념

```mermaid
graph TD
    A["Hugging Face Hub"] --> B["datasets 라이브러리"]
    B --> C["로드 / 스트리밍"]
    C --> D["로컬 캐시<br/>~/.cache/huggingface/"]
    B --> E["형식 변환<br/>CSV, JSON, Parquet, Arrow"]
    E --> F["데이터 분할<br/>train / val / test"]
    F --> G["학습 파이프라인"]
```

Hugging Face `datasets` 라이브러리는 AI 작업을 위해 데이터를 로드하는 표준적인 방법입니다. 다운로드, 캐싱, 형식 변환 및 스트리밍을 기본으로 처리합니다.

```figure
s0-data-pipeline
```

## 구현하기

### 1단계: datasets 라이브러리 설치

```bash
pip install datasets huggingface_hub
```

### 2단계: 데이터셋 로드

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
print(dataset)
print(dataset["train"][0])
```

IMDB 영화 리뷰 데이터셋을 다운로드합니다. 첫 번째 다운로드 후에는 `~/.cache/huggingface/datasets/`의 캐시에서 로드합니다.

### 3단계: 대형 데이터셋 스트리밍

일부 데이터셋은 디스크에 담기에는 너무 큽니다. 스트리밍은 전체를 다운로드하지 않고 행 단위로 로드합니다.

```python
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break
```

스트리밍은 `IterableDataset`를 제공합니다. 도착하는 행을 처리합니다. 데이터셋 크기와 관계없이 메모리 사용량이 일정하게 유지됩니다.

### 4단계: 데이터셋 형식

`datasets` 라이브러리는 내부적으로 Apache Arrow를 사용합니다. 파이프라인의 필요에 따라 다른 형식으로 변환할 수 있습니다.

```python
dataset = load_dataset("stanfordnlp/imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")
```

형식 비교:

| 형식 | 크기 | 읽기 속도 | 최적 용도 |
|--------|------|-----------|----------|
| CSV | 큼 | 느림 | 인간 가독성, 스프레드시트 |
| JSON | 큼 | 느림 | API, 중첩 데이터 |
| Parquet | 작음 | 빠름 | 분석, 컬럼 단위 쿼리 |
| Arrow | 작음 | 가장 빠름 | 메모리 내 처리 (`datasets`가 내부적으로 사용) |

AI 작업에서는 Parquet이 가장 적합한 저장 형식입니다. Arrow는 메모리에서 작업할 때 사용하며, CSV와 JSON은 데이터 교환용입니다.

### 5단계: 데이터 분할

모든 ML 프로젝트는 세 가지 분할이 필요합니다:

- **학습(Train)**: 모델이 이 데이터에서 학습합니다 (일반적으로 80%)
- **검증(Validation)**: 학습 중 진행 상황을 확인합니다 (일반적으로 10%)
- **테스트(Test)**: 학습 완료 후 최종 평가를 수행합니다 (일반적으로 10%)

일부 데이터셋은 미리 분할되어 있습니다. 분할되지 않은 경우 직접 분할하세요:

```python
dataset = load_dataset("stanfordnlp/imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

재현성을 위해 항상 시드(seed)를 설정하세요. 동일한 시드는 매번 동일한 분할을 생성합니다.

### 6단계: 모델 다운로드 및 캐싱

모델은 용량이 큰 파일입니다. `huggingface_hub` 라이브러리가 다운로드와 캐싱을 처리합니다.

```python
from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Cached at: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Full model at: {model_dir}")
```

모델은 `~/.cache/huggingface/hub/`에 캐시됩니다. 한 번 다운로드되면 이후 실행에서는 즉시 로드됩니다.

### 7단계: 대용량 파일 처리

모델 가중치와 대용량 데이터셋은 git에 포함하지 않아야 합니다. 세 가지 옵션이 있습니다:

**옵션 A: .gitignore (가장 간단)**

```
*.bin
*.safetensors
*.pt
*.onnx
data/*.parquet
data/*.csv
models/
```

**옵션 B: Git LFS (git에서 대용량 파일 추적)**

```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

Git LFS는 저장소에 포인터를 저장하고 실제 파일은 별도 서버에 보관합니다. GitHub는 1 GB를 무료로 제공합니다.

**옵션 C: DVC (데이터 버전 관리)**

```bash
pip install dvc
dvc init
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"
```

DVC는 데이터를 가리키는 작은 `.dvc` 파일을 생성합니다. 데이터 자체는 S3, GCS 또는 다른 원격 저장소 백엔드에 resides합니다.

| 접근 방식 | 복잡도 | 최적 용도 |
|----------|-----------|----------|
| .gitignore | 낮음 | 개인 프로젝트, 재다운로드 가능한 데이터 |
| Git LFS | 중간 | git으로 모델 가중치를 공유하는 팀 |
| DVC | 높음 | 재현 가능한 실험, 대용량 데이터셋, 팀 |

이 과정에서는 `.gitignore`이면 충분합니다. 여러 기계에서 정확한 실험을 재현해야 할 때 DVC를 사용해 보세요.

### 8단계: 저장 패턴

**로컬 저장소**는 약 10 GB 미만의 데이터셋에 적합합니다. HF 캐시가 이를 자동으로 처리합니다.

**클라우드 저장소**는 더 큰 데이터셋이나 여러 기계에서 공유하는 경우 사용합니다:

```python
import os

local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"
```

DVC는 S3 및 GCS와 직접 연동됩니다:

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

이 과정에서는 로컬 저장소로 충분합니다. 원격 GPU 인스턴스에서 미세 조정(Fine-tuning)할 때 클라우드 저장소가 유용해집니다.

## 이 과정에서 사용된 데이터셋

| 데이터셋 | 강 | 크기 | 가르치는 내용 |
|---------|---------|------|----------------|
| IMDB | 토큰화, 분류 | 84 MB | 텍스트 분류 기초 |
| WikiText | 언어 모델링 | 181 MB | 다음 토큰 예측 |
| SQuAD | QA 시스템 | 35 MB | 질문 답변, 스팬 |
| Common Crawl (하위 집합) | 임베딩 | 가변 | 대규모 텍스트 처리 |
| MNIST | 비전 기초 | 21 MB | 이미지 분류 기초 |
| COCO (하위 집합) | 멀티모달 | 가변 | 이미지-텍스트 쌍 |

지금 이 모든 것을 다운로드할 필요는 없습니다. 각 강에서 필요한 것을 지정합니다.

## 사용하기

유틸리티 스크립트를 실행하여 모든 것이 정상적으로 작동하는지 확인하세요:

```bash
python code/data_utils.py
```

이 스크립트는 작은 데이터셋을 다운로드하고, 변환하고, 분할하고, 요약 정보를 출력합니다.

## 출시하기

이 강은 다음을 생성합니다:
- `code/data_utils.py` - 재사용 가능한 데이터 로딩 및 캐싱 유틸리티
- `outputs/prompt-data-helper.md` - 작업에 적합한 데이터셋을 찾기 위한 프롬프트

## 연습 문제

1. `mrpc` 구성으로 `glue` 데이터셋을 로드하고 첫 5개 예제를 확인하세요
2. `c4` 데이터셋을 스트리밍하고 10초 동안 처리할 수 있는 예제 수를 세어 보세요
3. 데이터셋을 Parquet으로 변환하고 CSV와 파일 크기를 비교하세요
4. 고정된 시드로 70/15/15의 학습/검증/테스트 분할을 생성하고 크기를 확인하세요

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 데이터셋 분할(Dataset Split) | "학습 데이터" | ML 수명 주기의 여러 단계에서 사용되는 명명된 하위 집합(train/val/test) |
| 스트리밍(Streaming) | "지연 로딩" | 전체 데이터셋을 다운로드하지 않고 원격 소스에서 데이터를 행 단위로 처리하는 방식 |
| Parquet | "압축된 CSV" | 분석 쿼리 및 저장 효율성을 위해 최적화된 열 기반 파일 형식 |
| Arrow | "빠른 데이터 프레임" | datasets 라이브러리가 제로 복사(zero-copy) 읽기를 위해 내부적으로 사용하는 메모리 내 열 기반 형식 |
| Git LFS | "대용량 파일을 위한 Git" | 대용량 파일을 Git 저장소 외부에 저장하면서 버전 관리에 포인터를 유지하는 확장 프로그램 |
| DVC | "데이터를 위한 Git" | 클라우드 스토리지와 통합되는 데이터셋 및 모델용 버전 관리 시스템 |
| 캐시(Cache) | "이미 다운로드됨" | 기본적으로 ~/.cache/huggingface/에 저장되는 이전에 가져온 데이터의 로컬 사본 |
