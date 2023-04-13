# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score,\
                             confusion_matrix, recall_score,\
                             f1_score, roc_auc_score, \
                             precision_recall_curve, roc_curve
def get_eval_score(y_test, pred, pred_proba_c1=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    g = np.sqrt(precision*recall)
    auc = roc_auc_score(y_test, pred_proba_c1)
    print(f'오차행렬:\n {confusion}')
    print(f' 정확도:{accuracy:.4f}, 정밀도:{precision:.4f}', end=', ')
    print(f'재현율:{recall:.4f}\n F1:{f1:.4f}, G-measure:{g:.4f}', end=', ')
    print(f'AUC:{auc:.4f}')


def plot_importance(model, feature_names, top_n=None):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    sort_imp = importances.sort_values(ascending=False)
    if top_n:
        ftr_top = sort_imp[:top_n]
        top = ' Top '+ str(top_n)
    else:
        ftr_top = sort_imp
        top = ' All '
    plt.figure(figsize=(6,4))
    sns.barplot(x=ftr_top, y=ftr_top.index)
    plt.title('Feature Importances' + top)
    plt.show()


# #### 데이터 로딩

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=205)

# # 1. 보팅(Voting)

from sklearn.ensemble import VotingClassifier

# + active=""
# VotingClassifier(estimators, *, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False)
# -

# - estimators : 리스트 값으로 보팅에 사용될 여러 개의 classifier 객체
#     - 모델이름과 객체명을 튜플로 구성한 리스트
# - voting : 보팅 방식
#     - hard (default)
#     - soft

# #### soft 방식

# +
# 모델 객체 생성
lr_clf = LogisticRegression()
knn_clf = KNeighborsClassifier(n_neighbors=8)

estimators = [('LR', lr_clf), ('KNN', knn_clf)]
vo_clf = VotingClassifier(estimators=estimators, voting='soft')

# 학습
vo_clf.fit(X_train, y_train)

# 예측
pred = vo_clf.predict(X_test)
pred_prob = vo_clf.predict_proba(X_test)[:,1]

# 평가
get_eval_score(y_test, pred, pred_prob)
# -

# #### hard 방식

vo_clf_hard = VotingClassifier(estimators=estimators)
vo_clf_hard.fit(X_train, y_train)
pred_hard = vo_clf_hard.predict(X_test)
print(f'오차행렬:\n{confusion_matrix(y_test, pred_hard)}')
print(f'정확도:{accuracy_score(y_test, pred_hard):.4f}, 정밀도:{precision_score(y_test, pred_hard):.4f}, 재현율:{recall_score(y_test, pred_hard):.4f}, F1:{f1_score(y_test, pred_hard):.4f}')


# #### 참고. 개별 모델별 학습/예측/평가

# 학습/예측/평가 함수 정의
def fit_pred_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)[:,1]
    
    print(model.__class__.__name__)
    get_eval_score(y_test, pred, pred_prob)


fit_pred_eval(lr_clf, X_train, X_test, y_train, y_test)

fit_pred_eval(knn_clf, X_train, X_test, y_train, y_test)

# # 2. 배깅(Bagging)

# ## 랜덤 포레스트(Random Forest)

# ### 랜덤 포레스트에서의 부트스트래핑 샘플링 방식
#
# **부트스트래핑(bootstrapping) 분할 방식**
# - 개별 Classifier에게 데이터를 샘플링해서 추출하는 방식
# - 각 샘플링된 데이터 내에는 중복 데이터 포함
#
# **랜덤 포레스트 부트스트래핑 분할**
# - 개별적인 분류기의 기반 알고리즘은 결정 트리
# - 개별 트리가 학습하는 데이터 세트는 전체 데이터에서 일부가 중복되게 샘플링된 데이터 세트
# - Subset 데이터는 이러한 부트 스트래핑으로 데이터가 임의로 만들어짐
# - Subset 데이터 건수는 전체 데이터 건수와 동일하지만 개별 데이터가 중복되어 만들어짐
#
#
# - 예 : 원본 데이터 건수가 10개인 학습 데이터 세트

from sklearn.ensemble import RandomForestClassifier

# + active=""
# RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0,bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
# -

# - n_estimators : 결정트리의 개수
#     - 100 (default)
# - criterion : 트리 불순도 측정지표
#     - gini (default)
# - bootstrap : 트리 생성 시 bootstrap samples 이용 여부
#     - True (default)

rf_clf = RandomForestClassifier(random_state=205)
fit_pred_eval(rf_clf, X_train, X_test, y_train, y_test)

plot_importance(rf_clf, cancer.feature_names, 10)

# #### n_estimators = 200

rf_clf200 = RandomForestClassifier(n_estimators=200, random_state=205)
fit_pred_eval(rf_clf200, X_train, X_test, y_train, y_test)
plot_importance(rf_clf200, cancer.feature_names, 10)

# => n_estimators=100일 때보다 정확도가 올라감

# ### 교차검증 및 하이퍼파라미터 튜닝

from sklearn.model_selection import GridSearchCV

# #### 교차검증

# +
### RandomForestClassifier의 변수
params = {'n_estimators':[100],
          'max_depth':[8,16,24],
          'min_samples_split':[2,8,16],
          'min_samples_leaf':[1,6,12]}

rf_clf = RandomForestClassifier(random_state=205, n_jobs=-1)
### n_jobs=-1 : 내가 가진 CPU 몽땅 다 써라
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(X_train, y_train)
# -

# #### 하이퍼파라미터 찾기

pd.DataFrame(grid_cv.cv_results_).sort_values('rank_test_score').head(10)

print(f'최적의 하이퍼파라미터: {grid_cv.best_params_}')
print(f'최고 정확도: {grid_cv.best_score_:.4f}')


def best_model(model, X_test, y_test):
    best_rf_clf = model.best_estimator_
    pred = best_rf_clf.predict(X_test)
    pred_prob = best_rf_clf.predict_proba(X_test)[:,1]
    get_eval_score(y_test, pred, pred_prob)
    plot_importance(best_rf_clf, cancer.feature_names, 20)


best_model(grid_cv, X_test, y_test)

# ### 참고. 모델 선택/튜닝을 위해 사용하는 함수들

# - GridSearchCV() : 하이퍼파라미터의 값들을 지정하고 파라미터 조합에 대한 모델들을 교차검증하여 최적의 score를 내는 하이퍼파라미터를 찾음
#     - param_grid : 하이퍼파라미터 지정
#
#
# - RandomizedSearchCV() : 하이퍼파라미터의 값을 지정하여 그 중 랜덤하게 최적의 score를 찾아내는 하이퍼파라미터를 찾음
#     - param_distributions : 하이퍼파라미터 지정
#     - n_iter : 파라미터 검색 횟수 (지정 횟수만큼 조합을 반복하여 평가)
#
#
# - cross_validate() : 교차검증을 수행하는 함수
#     - scoring 매개변수
#         - 분류모델 : accuracy (default)
#         - 회귀모델 : r2 (default)
#         - 여러개 지정 가능
#     - cv : 교차검증 폴드 수 (default=5)나 splitter 객체 생성
#         - 회귀모델 : KFold
#         - 분류모델 : StratifiedKFold
#     - return_train_score
#         - False (default)
#         - True : 훈련데이터에 대한 점수 반환
#     - n_jobs : CPU 코어 수 지정
#         - 1 (default)
#         - -1 : 시스템의 모든 코어 사용

from sklearn.model_selection import RandomizedSearchCV, cross_validate

# #### cross_validate()를 사용한 교차검증

scores = cross_validate(rf_clf, X_train, y_train, scoring=['accuracy', 'roc_auc'], return_train_score=True)
scores

print(f'평균 정확도: {np.mean(scores["test_accuracy"]):.4f}')
print(f'평균 roc_auc: {np.mean(scores["test_roc_auc"]):.4f}')

# #### 모델 튜닝을 위한 RandomizedSearchCV()

# +
params = {'max_depth':range(5,20,1),
          'min_impurity_decrease':np.arange(0.0001, 0.001, 0.0001),
          'min_samples_split':range(2,100,10)}

rm_cv = RandomizedSearchCV(rf_clf, param_distributions=params, n_iter=100, n_jobs=-1, random_state=205)
rm_cv.fit(X_train, y_train)


# -

def param_cv_result(model, X_test, y_test):
    print(f'최적의 하이퍼파라미터: {model.best_params_}')
    print(f'최고 예측정확도: {model.best_score_:.4f}')
    print('---best_estimator---')
    best_rf_clf = model.best_estimator_
    pred = best_rf_clf.predict(X_test)
    pred_prob = best_rf_clf.predict_proba(X_test)[:,1]
    get_eval_score(y_test, pred, pred_prob)


param_cv_result(rm_cv, X_test, y_test)

# ## BaggingClassifier를 이용한 앙상블 학습
# - estimator : 결정트리 (default)

from sklearn.ensemble import BaggingClassifier

# +
params = {'n_estimators':range(10,60, 10),
          'random_state':[0,10,42,156],
          'max_samples':[0.1,0.25,0.5,0.75,1.0]}

bg_clf = BaggingClassifier(random_state=205)
grid_cv = GridSearchCV(bg_clf, param_grid=params, n_jobs=-1)
grid_cv.fit(X_train, y_train)
# -

param_cv_result(grid_cv, X_test, y_test)

# # 3. 부스팅

# ### AdaBoost

from sklearn.ensemble import AdaBoostClassifier

# - base_estimator : 예측할 모델
#     - DecisionTreeClassifier(max_depth=1) (default)
# - n_estimators : weak learner의 개수
#     - 50 (default)
# - learning_rate : 0~1 사이의 값 지정
#     - 1.0 (default)
#     - 너무 작은 값인 경우 최소점을 찾아서 성능이 높아지지만 학습시간이 오래걸림
#     - 너무 큰 값인 경우 최소점을 찾지 못해 예측성능이 떨어질 확률이 높음
#     - n_estimators와 상호보완

# +
ada_clf = AdaBoostClassifier(random_state=205)
ada_clf.fit(X_train, y_train)

scores = cross_validate(ada_clf, X_train, y_train, scoring=['accuracy', 'roc_auc'])
print(f"평균 정확도: {np.mean(scores['test_accuracy']):.4f}")
print(f"평균 roc_auc: {np.mean(scores['test_roc_auc']):.4f}")

# +
params = {'n_estimators':range(10,100,10),
          'learning_rate':[0.0001, 0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1],
          'random_state':[0, 42, 156]}

rm_cv = RandomizedSearchCV(ada_clf, param_distributions=params, n_iter=100, n_jobs=-1)
rm_cv.fit(X_train, y_train)
# -

param_cv_result(rm_cv, X_test, y_test)

plot_importance(ada_clf, cancer.feature_names, 10)

# ### GBM(Gradient Boosting Machine)

from sklearn.ensemble import GradientBoostingClassifier

# - loss : 경사하강법에서 사용할 비용 함수 지정
#     - log_loss (default)
# - n_estimators : weak learner 개수
#     - 100 (default)
# - learning_rate : GBM이 학습을 진행할 때마다 적용하는 학습률, 0~1 사이 값 지정, weak learner가 순차적으로 오류값을 보정해 나가는 데 적용하는 계수
#     - 0.1 (default)
# - subsample : weak learner가 학습에 사용하는 데이터의 샘플링 비율
#     - 1 (default) : 전체 학습 데이터를 기반으로 학습한다는 의미
#     - 과적합이 염려되는 경우 1보다 작은 값으로 설정

import time

# +
start_time = time.time()

gb_clf = GradientBoostingClassifier(random_state=205)
fit_pred_eval(gb_clf, X_train, X_test, y_train, y_test)
print(f'GBM 학습시간 : {time.time() - start_time:.2f}초')
# -

plot_importance(gb_clf, cancer.feature_names, 10)

# #### 교차검증 및 하이퍼 파라미터 튜닝

# +
params = {'n_estimators':[100,500],
          'subsample':[0.5,0.6,0.7,0.8,1.0],
          'max_depth':[6,8,10,12],
          'min_samples_split':[8,16,20],
          'min_samples_leaf':[6,8,12,18]}

start_time = time.time()

g_cv = GridSearchCV(gb_clf, param_grid=params, n_jobs=-1)
g_cv.fit(X_train, y_train)
param_cv_result(g_cv, X_test, y_test)
print(f'GBM 학습시간 : {time.time() - start_time:.2f}초')
# -

plot_importance(g_cv.best_estimator_, cancer.feature_names, 10)

# ### XGBoost(eXtra Gradient Boost)

# **XGBoost 장점**
# - 뛰어난 예측 성능
# - GBM 대비 빠른 수행 시간
# - 과적합 규제 
#     - 조기 중단(Early Stopping Rule)
#         - early_stoppings : 피용 평가 지표가 감소하지 않는 최대 반복 횟수
#         - eval_metric : 반복 수행시 사용하는 비용 평가 지표
#         - eval_set : 평가를 수행하는 별도의 검증 데이터 세트
# - 가지치기 (pruning)
# - 교차 검증 내장
# - 결측값 자체 처리

from xgboost import XGBClassifier

# +
xgb_clf = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, eval_metric='logloss')

fit_pred_eval(xgb_clf, X_train, X_test, y_train, y_test)
# -

plot_importance(xgb_clf, cancer.feature_names, 10)

# #### early_stopping = 10 으로 설정하고 재학습/예측/평가

# +
# 훈련/검증용 데이터 분할
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=205)

# 훈련
evals = [(X_tr, y_tr), (X_val, y_val)]
xgb_clf.fit(X_tr, y_tr, early_stopping_rounds=10, eval_set=evals, verbose=True)
### 평가지표인 log loss에서 같은 값이 50번 반복되면 중단
### verbose : 중간과정보여주기

# +
pred10 = xgb_clf.predict(X_test)
pred_proba10 = xgb_clf.predict_proba(X_test)[:,1]

get_eval_score(y_test, pred10, pred_proba10)
# -

# -> 전체로 돌린 xgboost보다 조기중단한 xgboost의 정확도가 떨어짐 (조기중단으로 인한 과소적합 가능성 있음)

plot_importance(xgb_clf, cancer.feature_names, 10)
