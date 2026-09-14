---
name: skill-ensemble-builder
description: 문제에 맞는 앙상블 방법을 고르고 설정한다
version: 1.0.0
phase: 2
lesson: 11
tags: [ensemble, bagging, boosting, random-forest, xgboost, stacking]
---

# 앙상블 방법 선택 가이드 (Ensemble Method Selection Guide)

앙상블은 여러 모델을 합쳐 단일 모델보다 나은 예측을 냅니다. 질문은 항상 같습니다. 어떤 종류의 앙상블을, 언제 쓸까?

## 결정 체크리스트

1. 현재 모델의 주요 문제는 무엇인가?
   - 높은 분산 (과적합): 배깅 사용 (랜덤 포레스트)
   - 높은 편향 (과소적합): 부스팅 사용 (그래디언트 부스팅, XGBoost)
   - 둘 다, 또는 최대 정확도가 필요: 스태킹 사용

2. 데이터가 얼마나 있는가?
   - 1,000행 미만: 랜덤 포레스트 (견고하고 설정을 망치기 어려움)
   - 1,000~100,000: XGBoost 또는 LightGBM (표 형식에 전반적으로 최선)
   - 100,000 초과: LightGBM (가장 빠른 그래디언트 부스팅, 대용량에 강함)

3. 튜닝에 얼마나 투자할 수 있는가?
   - 최소: 기본값 랜덤 포레스트 (거의 항상 동작)
   - 보통: learning_rate=0.1인 XGBoost, early stopping으로 n_estimators 튜닝
   - 최대: LightGBM 또는 XGBoost + 베이즈 하이퍼파라미터 탐색

4. 해석 가능성이 필요한가?
   - 예: 단일 결정 트리 또는 특성 중요도가 있는 작은 랜덤 포레스트
   - 부분: SHAP 값이 있는 그래디언트 부스팅
   - 아니오: 스태킹 또는 깊은 앙상블

5. 이상치가 많은 노이즈 데이터인가?
   - 예: 랜덤 포레스트 (배깅은 노이즈에 강함)
   - 아니오: 그래디언트 부스팅 (깨끗한 데이터에서 정확도를 더 밀어 올릴 수 있음)

## 방법별 사용 시점

**랜덤 포레스트 (배깅)**: 안전한 첫 선택. 부트스트랩 샘플로 많은 트리를 학습하고 평균합니다. 편향을 늘리지 않으면서 분산을 줄입니다. 중간 규모 데이터에서는 거의 과적합하지 않습니다. 튜닝은 최소: n_estimators=100-500, 나머지는 기본값.

**AdaBoost**: 샘플 재가중으로 하는 순차 부스팅. 단순한 베이스 학습기(결정 스텀프)와 잘 맞습니다. 오분류 점을 올려 가중하므로 이상치·노이즈 라벨에 민감합니다. 실무에서는 대부분 그래디언트 부스팅으로 대체되었습니다.

**그래디언트 부스팅**: 지금까지 앙상블의 잔차에 새 트리를 맞춥니다. 편향을 줄입니다. 표 형식 데이터에 가장 강력한 방법입니다. 튜닝 필요: learning_rate, n_estimators, max_depth, min_child_weight, subsample.

**XGBoost**: 정규화, 2차 최적화, 시스템 수준 가속이 있는 그래디언트 부스팅. 결측값을 네이티브로 처리합니다. Kaggle 대회와 표 형식 프로덕션 ML의 기본 선택입니다.

**LightGBM**: level-wise 대신 leaf-wise 성장의 그래디언트 부스팅. 큰 데이터셋에서 XGBoost보다 빠릅니다. 히스토그램 기반 분할을 씁니다. 행 5만 이상에 최선입니다.

**CatBoost**: 범주형 특성을 네이티브로 처리하는 그래디언트 부스팅. 원-핫이 필요 없습니다. 범주형이 많을 때 좋습니다.

**스태킹**: 서로 다른 여러 베이스 모델의 예측 위에 메타 학습기를 학습합니다. 절대 최고 정확도가 필요하고 계산 여유가 있을 때 쓰세요. 누수를 막으려면 베이스 예측은 항상 교차 검증으로 생성하세요.

**투표**: 가장 단순한 앙상블. 하드 투표(다수 클래스) 또는 소프트 투표(확률 평균). 메타 학습기 없이 서로 다른 모델 2~3개를 빠르게 합칩니다.

## 흔한 실수

- early stopping 없이 그래디언트 부스팅 사용 (라운드를 너무 많이 돌리면 과적합)
- learning_rate를 너무 높게 설정 (보통 0.3 초과면 불안정)
- 그래디언트 부스팅에서 max_depth를 튜닝하지 않음 (무제한·너무 깊은 기본값은 과적합)
- 같은 유형 모델만으로 스태킹 (다양성이 스태킹의 요점)
- 노이즈 데이터에 AdaBoost 사용 (이상치 가중치가 라운드마다 커짐)
- 랜덤 포레스트가 과소적합을 고칠 거라 기대 (분산만 줄이고 편향은 줄이지 않음)

## 방법별 튜닝 우선순위

**Random Forest:**
1. n_estimators: 100-500 (더 많다고 나빠지는 경우는 드물고, 느려질 뿐)
2. max_depth: None (끝까지 성장) 또는 속도를 위해 10-20으로 제한
3. max_features: 분류는 "sqrt", 회귀는 "log2" 또는 n/3

**XGBoost / LightGBM:**
1. learning_rate: 0.01-0.3 (트리를 더 돌릴 계산이 있으면 낮을수록 좋음)
2. n_estimators: 추측하지 말고 검증 세트에서 early stopping
3. max_depth: 3-8 (6부터)
4. min_child_weight / min_data_in_leaf: 1-20 (높을수록 과적합 방지)
5. subsample: 0.7-1.0
6. colsample_bytree: 0.7-1.0
7. reg_alpha (L1) and reg_lambda (L2): 0-10

## 빠른 참고

| Method | Reduces | Speed | Tuning effort | Best for |
|--------|---------|-------|--------------|----------|
| Random Forest | Variance | Fast | Low | Noisy data, quick baseline |
| AdaBoost | Bias | Fast | Low | Simple base learners, clean data |
| Gradient Boosting | Bias | Medium | High | Tabular data, competitions |
| XGBoost | Both | Fast | High | Production tabular ML |
| LightGBM | Both | Fastest | High | Large datasets (50k+ rows) |
| CatBoost | Both | Medium | Medium | Many categorical features |
| Stacking | Both | Slow | High | Maximum accuracy, diverse models |
| Voting | Variance | Fast | None | Quick combination of 2-3 models |
