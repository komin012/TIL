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

# # Credit Card Fraud Detection
# - 캐글에서 데이터 다운 : https://www.kaggle.com/mlg-ulb/creditcardfraud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ## 1. 데이터 1차 가공 및 모델 학습/예측/평가

card = pd.read_csv('data/creditcard.csv')
card.head(3)

card.info()

card.describe()

# **creditcard 컬럼 정보**
#
# - Time : 데이터 생성 관련한 속성으로 분석에 큰 의미가 없음 => 삭제
# - Amount : 신용카드 트랜잭션 금액
# - Class : 레이블, 0-정상, 1-사기  => target

card.Class.value_counts()


# ### 전처리 함수 : get_preprocessed_df() 정의

def get_preprocessed_df(df=None):
    df_copy = df.copy()
    df_copy.drop('Time', axis=1, inplace=True)
    return df_copy


# ### 데이터 세트 분할 함수 : get_train_test_dataset() 정의

# +
from sklearn.model_selection import train_test_split

def get_train_test_dataset(df=None):
    df_copy = get_preprocessed_df(df)
    X = df_copy.drop('Class', axis=1)
    y = df_copy['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=205, stratify=y)
    ### stratify = y : 학습/테스트 데이터에도 y의 target 비율 반영
    return X_train, X_test, y_train, y_test


# -

X_train, X_test, y_train, y_test = get_train_test_dataset(card)

print('학습데이터 target 비율')
print(y_train.value_counts() / y_train.shape[0] * 100)

print('테스트데이터 target 비율')
print(y_test.value_counts() / y_test.shape[0] * 100)

# ### 평가 함수, 피처 중요도 시각화 함수 정의

# +
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

def plot_importances(model, feature_names, top_n=None):
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


# -

# ## LogisticRegression 적용

from sklearn.linear_model import LogisticRegression


# ### 모델 학습/예측/평가 함수 : fit_pred_eval()

def fit_pred_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)[:,1]
    
    print(model.__class__.__name__)
    get_eval_score(y_test, pred, pred_prob)


# +
lr_clf = LogisticRegression(max_iter=100)
### max_iter : 회귀계수 수렴에 반복 횟수 제한

fit_pred_eval(lr_clf, X_train, X_test, y_train, y_test)
# -

# => target데이터가 불균형이라 accuracy_score가 크게 나옴(accuracy_score의 단점)

# ## LightGBM 적용

from lightgbm import LGBMClassifier

# +
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
### boost_from_average=False : 불균형 데이터에 적용하면 효율성을 높일 수 있음

fit_pred_eval(lgbm_clf, X_train, X_test, y_train, y_test)
# -

# => 로지스틱 회귀보다 전반적으로 높은 점수를 보임

# ## 2. 데이터 분포도 변환 후 모델 학습/예측/평가

# ### 중요 feature의 분포도 확인

# #### Amout 피처 히스토그램

sns.histplot(card.Amount, bins=100, kde=True)
plt.show()

# ### ① Amount 피처에 standardscaler 적용하기
# - get_preprocessed_df() 수정

# +
from sklearn.preprocessing import StandardScaler

def get_preprocessed_df(df=None):
    df_copy = df.copy()
    df_copy.drop('Time', axis=1, inplace=True)
    
    # Amount 칼럼 표준화 데이터로 변경
    scaler = StandardScaler()
    amount = scaler.fit_transform(df_copy.Amount.values.reshape(-1,1))
    df_copy.insert(0, 'AmountScaled', amount)
    df_copy.drop('Amount', axis=1, inplace=True)
    return df_copy


# -

# #### standardscaler + LogisticRegression & lightGBM

# +
X_train, X_test, y_train, y_test = get_train_test_dataset(card)

# 로지스틱회귀 학습/예측/평가
print('standard scaler: 로지스틱회귀 학습/예측/평가 ------------')
lr_clf = LogisticRegression(max_iter=1000)
fit_pred_eval(lr_clf, X_train, X_test, y_train, y_test)

# LightGBM 학습/예측/평가
print('standardscaler: LightGBM 학습/예측/평가 ------------')
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
fit_pred_eval(lgbm_clf, X_train, X_test, y_train, y_test)
# -

# **LogisticRegression**
# - 원데이터
#     - 정확도:0.9990, 정밀도:0.8095, 재현율:0.5743, F1:0.6719, G-measure:0.6819, AUC:0.9327
# - standardscaler
#     - 정확도:0.9993, 정밀도:0.8818, 재현율:0.6554, F1:0.7519, G-measure:0.7602, AUC:0.9750
#
#
#
# **LGBMClassifier**
# - 원데이터
#     - 정확도:0.9996, 정밀도:0.9512, 재현율:0.7905, F1:0.8635, G-measure:0.8672, AUC:0.9858
# - standardscaler
#     - 정확도:0.9996, 정밀도:0.9587, 재현율:0.7838, F1:0.8625, G-measure:0.8668, AUC:0.9850

# ======
# - 로지스틱회귀 : 점수 좀 올라감
# - LGBM : 점수 좀 떨어짐

# #### 표준화된 Amount 시각화

card_scaled = get_preprocessed_df(card)
sns.histplot(card_scaled.AmountScaled, bins=100, kde=True)
plt.show()


# ### ② Amount 피처 로그변환하기
# - get_preprocessed_df() 수정

def get_preprocessed_df(df=None):
    df_copy = df.copy()
    df_copy.drop('Time', axis=1, inplace=True)
    
    # Amount 칼럼 로그변환
    amount = np.log1p(df_copy.Amount)
    df_copy.insert(0, 'Amount_log', amount)
    df_copy.drop('Amount', axis=1, inplace=True)
    return df_copy


# #### log + LogisticRegression & lightGBM

# +
X_train, X_test, y_train, y_test = get_train_test_dataset(card)

# 로지스틱회귀 학습/예측/평가
print('log: 로지스틱회귀 학습/예측/평가 ------------')
lr_clf = LogisticRegression(max_iter=1000)
fit_pred_eval(lr_clf, X_train, X_test, y_train, y_test)

# LightGBM 학습/예측/평가
print('log: LightGBM 학습/예측/평가 ------------')
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
fit_pred_eval(lgbm_clf, X_train, X_test, y_train, y_test)
# -

# **LogisticRegression**
# - 원데이터
#     - 정확도:0.9990, 정밀도:0.8095, 재현율:0.5743, F1:0.6719, G-measure:0.6819, AUC:0.9327
# - standardscaler
#     - 정확도:0.9993, 정밀도:0.8818, 재현율:0.6554, F1:0.7519, G-measure:0.7602, AUC:0.9750
# - log
#     - 정확도:0.9992, 정밀도:0.8739, 재현율:0.6554, F1:0.7490, G-measure:0.7568, AUC:0.9728
#
#
#
# **LGBMClassifier**
# - 원데이터
#     - 정확도:0.9996, 정밀도:0.9512, 재현율:0.7905, F1:0.8635, G-measure:0.8672, AUC:0.9858
# - standardscaler
#     - 정확도:0.9996, 정밀도:0.9587, 재현율:0.7838, F1:0.8625, G-measure:0.8668, AUC:0.9850
# - log
#     -  정확도:0.9996, 정밀도:0.9512, 재현율:0.7905, F1:0.8635, G-measure:0.8672, AUC:0.9858

# ========
# - 로지스틱회귀 : standardscaler보다 좀 떨어짐
# - LGBM : standardscaler 보다 좀 올라감

# #### 표준화된 Amount 시각화

card_log = get_preprocessed_df(card)
sns.histplot(card_log.Amount_log, bins=100, kde=True)
plt.show()

# ## 3. 이상치 데이터 제거 후 모델 학습/예측/평가

# ### 각 피처들 상관관계 시각화 : 히트맵

sns.heatmap(card.corr(),cmap='RdBu')
plt.show()

# ### 피처별 이상치 확인 시각화

# +
ftr_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

fig, axs = plt.subplots(figsize=(16, 24), ncols=4, nrows=7)
for i, f in enumerate(ftr_names):
    row = int(i/4)
    col = i%4
    sns.histplot(card[f], kde=True, ax=axs[row][col])

plt.tight_layout()
plt.show()
# -

# - Class = 1 (사기 거래)인 경우의 이상치

# +
card1 = card[card['Class']==1]
ftr_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

fig, axs = plt.subplots(figsize=(16, 24), ncols=4, nrows=7)
for i, f in enumerate(ftr_names):
    row = int(i/4)
    col = i%4
    sns.histplot(card[f], kde=True, ax=axs[row][col])

plt.tight_layout()
plt.show()
# -

# #### 피처 V1의 분포 : 사분위수

card.V1.describe()


# 평균은 0.000000000 인데 최대값은 2.45

# ### 이상치 필터링 함수 : get_outlier() 정의

def get_outlier(df=None, column=None, weight=1.5):
    df_c = df[column]
    q1 = np.percentile(df_c, 25)
    q3 = np.percentile(df_c, 75)
    iqr = q3-q1
    lw_hinge = q1 - iqr*weight
    up_hinge = q3 + iqr*weight    
    outlier_idx = df_c[(df_c < lw_hinge)|(df_c > up_hinge)].index
    return outlier_idx


# ### ③ Amount 피처 로그변환 + V1 피처 이상치 제거
# - get_preprocessed_df() 수정

def get_preprocessed_df(df=None):
    df_copy = df.copy()
    
    # Amount 칼럼 로그변환
    amount = np.log1p(df_copy.Amount)
    df_copy.insert(0, 'Amount_log', amount)
    df_copy.drop(['Amount', 'Time'], axis=1, inplace=True)
    
    # 이상치 제거
    out_idx = get_outlier(df_copy, 'V1')
    df_copy.drop(out_idx, axis=0, inplace=True)
    return df_copy


# #### log + LogisticRegression & lightGBM

# +
X_train, X_test, y_train, y_test = get_train_test_dataset(card)

# 로지스틱회귀 학습/예측/평가
print('log: 로지스틱회귀 학습/예측/평가 ------------')
lr_clf = LogisticRegression(max_iter=1000)
fit_pred_eval(lr_clf, X_train, X_test, y_train, y_test)

# LightGBM 학습/예측/평가
print('log: LightGBM 학습/예측/평가 ------------')
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
fit_pred_eval(lgbm_clf, X_train, X_test, y_train, y_test)
# -

# **LogisticRegression**
# - 원데이터
#     - 정확도:0.9990, 정밀도:0.8095, 재현율:0.5743, F1:0.6719, G-measure:0.6819, AUC:0.9327
# - standardscaler
#     - 정확도:0.9993, 정밀도:0.8818, 재현율:0.6554, F1:0.7519, G-measure:0.7602, AUC:0.9750
# - log
#     - 정확도:0.9992, 정밀도:0.8739, 재현율:0.6554, F1:0.7490, G-measure:0.7568, AUC:0.9728
# - 이상치제거+log
#     - 정확도:0.9995, 정밀도:0.8857, 재현율:0.6526, F1:0.7515, G-measure:0.7603, AUC:0.9932
#
#
#
# **LGBMClassifier**
# - 원데이터
#     - 정확도:0.9996, 정밀도:0.9512, 재현율:0.7905, F1:0.8635, G-measure:0.8672, AUC:0.9858
# - standardscaler
#     - 정확도:0.9996, 정밀도:0.9587, 재현율:0.7838, F1:0.8625, G-measure:0.8668, AUC:0.9850
# - log
#     -  정확도:0.9996, 정밀도:0.9512, 재현율:0.7905, F1:0.8635, G-measure:0.8672, AUC:0.9858
# - 이상치제거+log
#     - 정확도:0.9997, 정밀도:0.9600, 재현율:0.7579, F1:0.8471, G-measure:0.8530, AUC:0.9813

# ========
# - 로지스틱회귀 : 정밀도는 올라가고 재현율은 낮아짐
# - LGBM : 정밀도는 올라가고 재현율은 낮아짐

# #### 표준화된 Amount 시각화

card_outlier = get_preprocessed_df(card)
sns.histplot(card_log.Amount_log, bins=100, kde=True)
plt.show()

# ## 4. SMOTE 오버 샘플링 적용 후 모델 학습/예측/평가

# **SMOTE(Synthetic Minority Over-sample Techique)**
#
# - 적은 데이터 세트에 대해 k최근접 이웃으로 데이터 신규 증식 대상 설정해 오버샘플링
# - 기존데이터와 약간 차이나는 데이터 생성
# - 패키지 설치 : https://imbalanced-learn.org/stable/

from imblearn.over_sampling import SMOTE

## 로그변환만 한 데이터 분할 함수 적용
X_train, X_test, y_train, y_test = get_train_test_dataset(card)

# +
smote = SMOTE(random_state=205)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

print(f'SMOTE 적용 전 학습 데이터셋 -> {X_train.shape, y_train.shape}')
print(f'SMOTE 적용 후 학습 데이터셋 -> {X_train_over.shape, y_train_over.shape}')
# -

print(f'SMOTE 적용 전 Class 분포 \n{y_train.value_counts()}')
print(f'SMOTE 적용 후 Class 분포 \n {y_train_over.value_counts()}')

y_train.describe()

y_train_over.describe()

X_train.Amount_log.describe()

X_train_over.Amount_log.describe()

# => over샘플링을 하면 개수만 늘어날 뿐 4분위수는 비슷

# ### 오버샘플링된 데이터로 로지스틱 회귀 학습/예측/평가

# 로지스틱회귀 학습/예측/평가
print('## 오버샘플링 : 로지스틱회귀 학습/예측/평가----------')
lr_clf=LogisticRegression(max_iter=1000)
fit_pred_eval(lr_clf, X_train_over, X_test, y_train_over, y_test)


# **LogisticRegression**
# - 원데이터
#     - 정확도:0.9990, 정밀도:0.8095, 재현율:0.5743, F1:0.6719, G-measure:0.6819, AUC:0.9327
# - standardscaler
#     - 정확도:0.9993, 정밀도:0.8818, 재현율:0.6554, F1:0.7519, G-measure:0.7602, AUC:0.9750
# - log
#     - 정확도:0.9992, 정밀도:0.8739, 재현율:0.6554, F1:0.7490, G-measure:0.7568, AUC:0.9728
# - 이상치제거+log
#     - 정확도:0.9995, 정밀도:0.8857, 재현율:0.6526, F1:0.7515, G-measure:0.7603, AUC:0.9932
# - 오버샘플링
#     - 정확도:0.9733, 정밀도:0.0563, 재현율:0.9122, F1:0.1060, G-measure:0.2265, AUC:0.9791

# => 오버샘플링한 경우 정확도 떨어짐. 재현율은 많이 상승 but 정밀도가 너무 낮아서 의미가 없음

# #### Precision-Recall 곡선 시각화

def precision_recall_curve_plot(y_test, pred_proba_c1):
    prec,  rec, thres = precision_recall_curve(y_test, pred_proba_c1)  # precision, recall, thresholds
    plt.plot(thres, prec[:-1], label='precision')
    plt.plot(thres, rec[:-1], label='recall', ls='--')
    plt.legend()
    plt.xlabel('Threshold value')
    plt.ylabel('Precision & Recall value')
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.grid()
    plt.show()


pred_proba_c1 = lr_clf.predict_proba(X_test)[:,1]
precision_recall_curve_plot(y_test, pred_proba_c1)

# => 분류경계 임계값(threshold)이 0.99이하에서 재현율이 떨어지고 정밀도가 높아지는 상황. 재현율과 정밀도의 차이가 너무 심해 올바른 예측 모델을 생성했다고 보기 어려움.

# ### 오버샘플링된 데이터로 LightGBM 학습/예측/평가

# LightGBM 학습/예측/평가
print('## 오버샘플링 : LightGBM 학습/예측/평가-------------')
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
fit_pred_eval(lgbm_clf, X_train_over, X_test, y_train_over, y_test)

# **LGBMClassifier**
# - 원데이터
#     - 정확도:0.9996, 정밀도:0.9512, 재현율:0.7905, F1:0.8635, G-measure:0.8672, AUC:0.9858
# - standardscaler
#     - 정확도:0.9996, 정밀도:0.9587, 재현율:0.7838, F1:0.8625, G-measure:0.8668, AUC:0.9850
# - log
#     -  정확도:0.9996, 정밀도:0.9512, 재현율:0.7905, F1:0.8635, G-measure:0.8672, AUC:0.9858
# - 이상치제거+log
#     - 정확도:0.9997, 정밀도:0.9600, 재현율:0.7579, F1:0.8471, G-measure:0.8530, AUC:0.9813
# - 오버샘플링
#     - 정확도:0.9995, 정밀도:0.9037, 재현율:0.8243, F1:0.8622, G-measure:0.8631, AUC:0.9911

# => 오버샘플링한 경우 정확도 떨어짐, 정밀도도 떨어짐, 재현율은 상승

# #### Precision-Recall 곡선 시각화

pred_proba_c1 = lgbm_clf.predict_proba(X_test)[:,1]
precision_recall_curve_plot(y_test, pred_proba_c1)


