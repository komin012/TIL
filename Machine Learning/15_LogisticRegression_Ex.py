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

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# -

# # statsmodels의 Logit을 이용한 로지스틱 회귀 분서

# ## 예제데이터 : titanic

titanic = sns.load_dataset('titanic')
titanic.head()

titanic.info()

titanic_df = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
titanic_df.info()

# ### EDA & 전처리

titanic_df.hist(figsize=(8,6))
plt.tight_layout()
plt.show()

# 왜도
titanic_df[['age', 'fare']].skew()

# 결측치 확인 
titanic_df.isna().sum()

# 기술통계량
titanic_df.describe()

# #### 결측치 처리

titanic_df.dropna(inplace=True)
titanic_df.info()

titanic_df.hist(figsize=(8,6))
plt.tight_layout()
plt.show()

titanic_df[['age', 'fare']].skew()

# #### 로그변환

titanic_df['log_fare'] = np.log1p(titanic_df.fare)

titanic_df[['fare', 'log_fare']].hist()
plt.show()

titanic_df.log_fare.skew()

titanic_df.drop('fare', axis=1, inplace=True)

titanic_df.head()

# ### 원-핫 인코딩

# +
df_ohe = pd.get_dummies(titanic_df, columns=['pclass', 'sex'], drop_first=True)
## drop_first=True : 더미변수 중 첫번째 변수 칼럼은 삭제
##                   pclass(1,2,3) 중 1, 성별(여,남) 중 여

df_ohe.head()
# -

# ### statsmodels.Logit()으로 로지스틱 회귀분석

import statsmodels.api as sm

X_names = df_ohe.columns.difference(['survived'])
X = df_ohe[X_names]
y = df_ohe['survived']

# #### 상수항 추가

X_ = sm.add_constant(X, has_constant='add')
X_.head()

# #### 로지스틱 회귀모형 생성/적합

model = sm.Logit(y, X_)
result = model.fit()

# #### 회귀모형 적합 결과

result.summary()

# - LLR : p-value가 작으므로 귀무가설 기각 => 회귀계수가 0이 아닌게 있음. 회귀식 타당

# - 추정된 로지스틱 회귀식
#
# $$ logit(\frac{p(x)}{1-p(x)}) = 3.4785 - 0.0443age + 0.2215ln(fare) - 0.0984parch + ... - 0.4217sibsp $$
#
# $$p(x) = \frac{e^{3.4785 - 0.0443age + 0.2215ln(fare) - 0.0984parch + ... - 0.4217sibsp}}{1+e^{3.4785 - 0.0443age + 0.2215ln(fare) - 0.0984parch + ... - 0.4217sibsp}}$$

# - 회귀계수 추정값

result.params

# - 계수값에 대한 신뢰구간(2.5%, 97.5%)

result.conf_int()

# - Odds:exp(Logit)

np.exp(result.params)

# **로지스틱 회귀분석 결과 해석**
#
# - age는 값이 높아질수록 생존확률은 낮아짐
#     - Logit: age가 1살 증가할 때 생존일 Logit이 -0.0443 단위 증가
#     - Odds: age가 1살 증가할 때 생존할 확률이 0.9567배($e^{-0.0443}$) 증가
#
# - sex는 남성일 때 생존확률이 낮아짐
#     - Logit: sex_male이 1단위 증가할 때 생존일 Logit이 -2.622 단위 증가
#     - Odds: sex_male이 1단위 증가할 때 생존할 확률이 0.073배($e^{-2.622}$ 증가

# #### Logit 모델 성능 평가

from sklearn.preprocessing import Binarizer
from sklearn.metrics import confusion_matrix, accuracy_score

# +
pred_proba = result.predict().reshape(-1,1)  ## 예측 확률

bina = Binarizer(threshold=0.5)  
pred = bina.fit_transform(pred_proba)
## 0.5를 기준을 예측 확률이 그 이상이면 1/ 미만이면 0으로 나타냄 
pred[:10]
# -

confusion_matrix(y, pred)

accuracy_score(y,pred)

# #### 예측
# - Age가 10, 30, 50인 경우 생존확률 예측하기

# +
# Age 이외의 변수들 값은 평균값으로 설정

ages = [10, 30, 50]
means = X_.mean()

X_test = pd.DataFrame({'const':means.const,'age':ages, 'log_fare':means.log_fare,  'parch':means.parch, 
                          'pclass_2':means.pclass_2, 'pclass_3':means.pclass_3, 'sex_male':means.sex_male, 'sibsp':means.sibsp})
X_test
# -

result.predict(X_test)

X_test['predict'] = result.predict(X_test)
X_test

# => 나이가 많을수록 생존확률 떨어짐

# ### 사이킷런의 LogisticRegression을 이용한 분석

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

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


# -

# #### 전체데이터로 모델 학습/예측

# +
X_test = pd.DataFrame({'age':ages, 'log_fare':means.log_fare,  'parch':means.parch, 
                          'pclass_2':means.pclass_2, 'pclass_3':means.pclass_3, 'sex_male':means.sex_male, 'sibsp':means.sibsp})

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X,y)

pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,1]

pred, pred_proba
# -

lr_clf.intercept_, lr_clf.coef_

# #### 학습/테스트 데이터로 분할 후 모델 학습/예측

# +
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=205)

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,1]

get_eval_score(y_test, pred, pred_proba)
# -

lr_clf.intercept_, lr_clf.coef_

# #### LogisticRegression(penalty='none')
# - default) penalty='l2' (L2 규제)

# +
lr_clf_p = LogisticRegression(penalty='none')
lr_clf_p.fit(X_train, y_train)

pred = lr_clf_p.predict(X_test)
pred_proba = lr_clf_p.predict_proba(X_test)[:,1]

get_eval_score(y_test, pred, pred_proba)
# -

lr_clf_p.intercept_, lr_clf_p.coef_

# #### 교차검증

lr_clf_cv = LogisticRegression(solver='liblinear')
scores = cross_val_score(lr_clf_cv, X, y)
print(scores)
print(np.mean(scores))

# #### 로지스틱회귀에서 규제를 적용한 최적의 모델 검증

from sklearn.model_selection import GridSearchCV

# +
lr_clf_g = LogisticRegression()

params = {'penalty':['l2', 'l1'],
         'C':[0.01, 0.1, 1, 5, 10]} ## C: 규제강도 조절, alpha의 역수, C값이 작을수록 규제강도 큼
grid_clf = GridSearchCV(lr_clf_g, param_grid=params, cv=3, scoring='accuracy')
grid_clf.fit(X_train, y_train)
# -

# - 최적의 하이퍼파라미터 성능

print(grid_clf.best_params_)
print(grid_clf.best_score_)

# - 최적의 하이퍼파라미터로 학습한 분류기 성능

# +
best_clf = grid_clf.best_estimator_
best_pred = best_clf.predict(X_test)
best_pred_proba = best_clf.predict_proba(X_test)[:,1]

get_eval_score(y_test, best_pred, best_pred_proba)
# -


