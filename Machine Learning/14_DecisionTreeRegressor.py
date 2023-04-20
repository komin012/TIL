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

# # 회귀 트리(Regression Tree)
# - 트리 기반의 회귀 알고리즘
# - 트리 생성 방식은 분류 트리와 같지만 리프 노드에 속하는 데이터 값의 평균값을 구해 회귀 예측값으로 결정한다는 점이 다름

# ## 예제. 보스턴 주택 가격 예측

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# +
boston = load_boston()
X = boston.data
y = boston.target

bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDF['price'] = boston.target
bostonDF.head()


# -

# ### 교차검증 및 예측 성능 출력 함수 작성

def model_cv_training(model, X, y, cv=5, scoring='neg_mean_squared_error', alpha=None):
    neg_mse = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    rmse = np.sqrt(neg_mse*(-1))
    avg_rmse = np.mean(rmse)
    
    if alpha:
        text = 'alpha=' + str(alpha)
    else:
        text = ''
    print(f'## {model.__class__.__name__} {text} ##')
    print(f'RMSE: {np.round(rmse,3)}')
    print(f'AVG_RMSE: {avg_rmse:.3f}')
    return rmse, avg_rmse


# ### 모델1. 랜덤포레스트

rf = RandomForestRegressor(n_estimators=1000, random_state=205)
rf_score = model_cv_training(rf, X, y)

# ### 모델2. 다양한 유형의 회귀트리 모델 이용
# - 의사결정트리(DecisionTreeRegressor)
# - 배깅
#     - RandomForestRegressor
# - 부스팅
#     - GradientBoostingRegressor
#     - XGBRegressor
#     - LGBMRegressor

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# +
dt = DecisionTreeRegressor(max_depth=4, random_state=205)
rf = RandomForestRegressor(n_estimators=1000, random_state=205)
gb = GradientBoostingRegressor(n_estimators=1000, random_state=205)
xgb = XGBRegressor(n_estimators=1000)
lgbm = LGBMRegressor(n_estimators=1000)

for m in [dt, rf,gb, xgb, lgbm]:
    print(f'{m.__class__.__name__}')
    model_cv_training(m, X, y)
# -

# ### 회귀트리의 피처별 중요도
# - Regressor는 coef_가 없음
# - 대신 feature_importances_를 이용해 피처별 중요도 파악

# +
rf = RandomForestRegressor(n_estimators=1000, random_state=205)
rf.fit(X,y)

ftr_import = pd.Series(rf.feature_importances_, index=boston.feature_names)
ftr_import = ftr_import.sort_values(ascending=False)
# -

sns.barplot(ftr_import,ftr_import.index)
plt.show()

sns.regplot(bostonDF.RM, bostonDF.price)
plt.show()

# ### 모델3. RM변수와 price변수만 선택하여 단순회귀분석

# - 독립변수 : 'RM' (거주할 수 있는 방의 수)
# - 종속변수 : 'PRICE' (주택 가격)
# - 데이터 개수 : 100
# - 모델 : 
#     - LinearRegression
#     - DecisionTreeRegressor(max_depth=2)
#     - DecisionTreeRegressor(max_depth=4)
#     - DecisionTreeRegressor(max_depth=7)
# - 테스트 데이터셋
#     - X_test = np.arange(4.5, 8.5, 0.04).reshape(-1,1)

# #### 데이터셋 생성

boston_sample = bostonDF[['RM', 'price']].sample(100, random_state=205)
sns.regplot(boston_sample.RM, boston_sample.price)
plt.show()

X = np.array(boston_sample.RM).reshape(-1,1)
y = np.array(boston_sample.price).reshape(-1,1)
X, y

# #### 회귀트리로 주택가격 예측

from sklearn.linear_model import LinearRegression

# +
X_test = np.arange(4.5, 8.5, 0.04).reshape(-1,1)

lr = LinearRegression()
dt_2 = DecisionTreeRegressor(max_depth=2, random_state=205)
dt_4 = DecisionTreeRegressor(max_depth=4, random_state=205)
dt_7 = DecisionTreeRegressor(max_depth=7, random_state=205)

lr.fit(X, y)
dt_2.fit(X, y)
dt_4.fit(X, y)
dt_7.fit(X, y)

lr_pred = lr.predict(X_test)
dt_2_pred = dt_2.predict(X_test)
dt_4_pred = dt_4.predict(X_test)
dt_7_pred = dt_7.predict(X_test)
# -

# #### 예측한 회귀직선 시각화

# +
fig, axs = plt.subplots(nrows=2, ncols=2)

axs[0][0].set_title('LinearRegression')
axs[0][0].scatter(X, y)
axs[0][0].plot(X_test, lr_pred, color='orange')

axs[0][1].set_title('DecisionTreeRegressor(max_depth=2)')
axs[0][1].scatter(X, y)
axs[0][1].plot(X_test, dt_2_pred, color='orange')

axs[1][0].set_title('DecisionTreeRegressor(max_depth=4)')
axs[1][0].scatter(X, y)
axs[1][0].plot(X_test, dt_4_pred, color='orange')

axs[1][1].set_title('DecisionTreeRegressor(max_depth=7)')
axs[1][1].scatter(X, y)
axs[1][1].plot(X_test, dt_7_pred, color='orange')

plt.tight_layout()
plt.show()
