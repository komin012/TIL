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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# -

# # 선형 모델을 위한 데이터 변환
# - 중요 피처들이나 타깃 값의 분포도가 심하게 비대칭인 경우

# **피처(X) 변환
# 1. 표준화 : StandardScaler 클래스
# 2. 정규화 : MinMaxScaler 클래스
# 3. 표준화/정규화를 수행한 데이터세트에 다시 다항특성을 적용하여 변환
#     - 표준화/정규화를 해도 예측성능이 없는 경우
# 4. 비대칭분포(오른쪽으로 꼬리가 긴 분포)의 경우 로그 변환
#
# **타깃(y) 변환**<br>
# :주로 비대칭인 경우 로그 변환

# ## 보스톤 주택가격 예측

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score

# +
boston = load_boston()

X = boston.data
y = boston.target

bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDF['price'] = boston.target
# -

# ### 주요 피처들의 분포

bostonDF.hist(figsize=(12,12))
plt.show()


# ### 규제가 있는 회귀모델 적용 함수

# +
def cv_training(model, X, y, cv=5, scoring='neg_mean_squared_error', alpha=None):
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

# X, y : DataFrame / Series 형식
def get_linear_reg_eval(model_name, params=None, X=None, y=None, return_coeff=True):
    coeff_df = pd.DataFrame()
    for param in params:
        if model_name == 'Ridge':
            model = Ridge(alpha=param)
        elif model_name == 'Lasso':
            model = Lasso(alpha=param)
        elif model_name == 'ElasticNet':
            model = ElasticNet(alpha=param)
        
        # 교차검증
        print(f'## {model_name}(alpha={param}) 교차검증 결과: ')
        cv_training(model, X, y, alpha=params)
        print('------------')
    
        # 회귀계수
        model.fit(X,y)
        if return_coeff:
            coeff = pd.Series(model.coef_, index=X.columns)
            coeff_df['alpha='+str(param)] = coeff
    return coeff_df


# -

# ### 데이터 변환을 위한 함수 정의 get_scaled_data()

# +
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

def get_scaled_data(method=None, p_degree=None, input_data=None):
    if method == 'Standard':
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax':
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log':
        scaled_data = np.log1p(input_data)
    else:
        scaled_data =input_data
    
    if p_degree:
        poly = PolynomialFeatures(degree=p_degree, include_bias=False)
        scaled_data = poly.fit_transform(scaled_data)
    
    return scaled_data


# -

# ### 회귀모델 적용
# - Ridge
# - alphas = [0.1, 1, 10, 100]

alphas = [0.1, 1, 10, 100]
methods = [(None, None), ('Standard',None), ('Standard', 2), ('MinMax', None), ('MinMax', 2), ('Log', None), ('Log', 2)]
for method, degree in methods:
    scaled_data = get_scaled_data(method, p_degree=degree, input_data=X)
    
    print(f'*** scaled_method:{method}, degree:{degree}***')
    get_linear_reg_eval('Ridge', alphas, scaled_data, y, return_coeff=False)

# **Avg_RMSE**<br>
#
# | 스케일방식 | a=0.1 | a=1 | a=10 | a=100 |
# |------------|--------|-------|------|-----|
# |원본데이터|5.788|5.653|5.518|5.330|
# |Standard|5.826|5.803|5.637|5.421|
# |Standard+2차다항식|8.827|6.871|5.485|4.634|
# |MinMax|5.764|5.465|5.754|7.635|
# |MinMax+2차다항식|5.298|4.323|5.185|6.538|
# |Log|4.770|4.676|4.836|6.241|
# |Log+2차다항식|9.547|5.847|4.270|4.559|

# => 피처들의 분포가 비대칭인게 많아서 Log 변환이 가장 좋아 보임
