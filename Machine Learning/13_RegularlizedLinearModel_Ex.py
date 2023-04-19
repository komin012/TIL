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

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# ### 데이터 준비

boston = load_boston()

print(boston.DESCR)

# #### 변수설명
# - CRIM: 지역별 범죄 발생률
# - ZN: 25,000평방피트를 초과하는 거주 지역의 비율
# - INDUS: 비상업 지역 넓이 비율
# - CHAS: 찰스강에 대한 더미 변수(강의 경계에 위치한 경우는 1, 아니면 0)
# - NOX: 일산화질소 농도
# - RM: 거주할 수 있는 방 개수
# - AGE: 1940년 이전에 건축된 소유 주택의 비율
# - DIS: 5개 주요 고용센터까지의 가중 거리
# - RAD: 고속도로 접근 용이도
# - TAX: 10,000달러당 재산세율
# - PTRATIO: 지역의 교사와 학생 수 비율
# - B: 지역의 흑인 거주 비율
# - LSTAT: 하위 계층의 비율
# - MEDV: 본인 소유의 주택 가격(중앙값)

bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDF['price'] = boston.target
bostonDF.head(3)

bostonDF.describe()

bostonDF.hist(figsize=(12,12))
plt.tight_layout()
plt.show()

# +
skew = []
kurt = []
for ftr in bostonDF.columns:
    skew.append(bostonDF[ftr].skew())
    kurt.append(bostonDF[ftr].kurt())

describeDF = bostonDF.describe().T
describeDF['skewness'] = skew
describeDF['kurtosis'] = kurt
describeDF[['skewness', 'kurtosis']]

# +
feature_names = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT'] ## 더미변수인 'CHAS' 제외

fig, axs = plt.subplots(figsize=(16,8), nrows=3, ncols=4)
for i, ftr in enumerate(feature_names):
    row = i//4
    col = i%4
    sns.regplot(bostonDF[ftr], bostonDF.price, ax=axs[row][col])
plt.tight_layout()
plt.show()


# -

def corr_heatmap(df):
    corr_df = df.corr()
    cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
    plt.figure(figsize=(12,12))
    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_df, annot=True, cmap=cmap, mask=mask, linewidths=0.5, fmt='.2f', annot_kws={'size':8})
    plt.show()


corr_heatmap(bostonDF)

# +
from scipy import stats

corrs = []
features = bostonDF.columns
print('## Price와의 상관계수 ##')
for ftr in features:
    stat, p = stats.pearsonr(bostonDF.price, bostonDF[ftr])
    print(f'Price와 {ftr}의 상관계수={stat}, p-value={p}')
    corrs.append(np.round(stat,3))
    
corrDF = pd.DataFrame(corrs, index=features, columns=['Correlation'])
corrDF.sort_values('Correlation', ascending=False)
# -

# ### 일반 선형 모델 : LinearRegression()

from sklearn.linear_model import LinearRegression


# +
# 회귀계수 시각화
def reg_coeff_plot(model, names):
    coeff = pd.Series(model.coef_, index=names)
    coeff = coeff.sort_values(ascending=False)
    print('회귀계수')
    print(coeff)
    sns.barplot(coeff, coeff.index)
    plt.show()
    
# 잔차 구하기
def get_residuals(y_test, y_pred, top_n=10):
    df = pd.DataFrame(y_test.values, columns=['real'])
    df['pred'] = np.round(y_pred)
    df['abs(resid)'] = np.abs(df['real']-df['pred'])
    print(df.sort_values('abs(resid)', ascending=False)[:top_n])

# 성능평가
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score

def eval_regr_score(y, pred, is_expm1=False):
    rmse = np.sqrt(mean_squared_error(y,pred))
    mae = mean_absolute_error(y,pred)
    r2 = r2_score(y, pred)
    
    if is_expm1:
        rmsle = mean_squared_log_error(y,pred)
        print(f'RMASLE: {rmsle:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, r2-score: {r2:.3f}')
    else:
        print(f'RMSE: {rmse:.3f}, MAE: {mae:.3f}, r2-score: {r2:.3f}')

# 모델 학습/예측/평가
def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if is_expm1:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)
        text = 'log 변환'
    else:
        text = ''
    
    print(f'## {model.__class__.__name__} {text} ##')
    eval_regr_score(y_test, y_pred, is_expm1=is_expm1)


# +
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=205)

lr = LinearRegression()
get_model_predict(lr, X_train, X_test, y_train, y_test)
# -

reg_coeff_plot(lr, boston.feature_names)

# #### LinearRegression 교차검증

from sklearn.model_selection import cross_val_score

# +
neg_mse = cross_val_score(lr, boston.data, boston.target, scoring='neg_mean_squared_error')
rmse = np.sqrt(neg_mse*(-1))
avg_rmse = np.mean(rmse)

print(f'RMSE: {np.round(rmse,3)}\nAVG_RMSE: {avg_rmse:.3f}')
# -

# ## 1. 릿지 회귀모델

# #### 릿지 회귀모델 정의

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=10)  ## default) alpha=1
get_model_predict(ridge, X_train, X_test, y_train, y_test)

# #### 릿지 회귀모델 회귀계수 시각화

print('## alpha=10인 경우 ##')
reg_coeff_plot(ridge, boston.feature_names)


# -> 릿지를 사용해서 회귀계수가 줄어든 거 확인 가능

# #### 교차 검증으로 모델 평가

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


ridge = Ridge(alpha=10)
cv_training(ridge, boston.data, boston.target, alpha=10)

# - LinearRegression CV
#     - RMSE: [3.53  5.104 5.751 8.987 5.772]
#     - AVG_RMSE: 5.829
# - Ridge(alpha=10) CV
#     - RMSE: [3.38  4.929 5.305 8.637 5.34 ]
#     - AVG_RMSE: 5.518

# ### Ridge에서 alpha를 다르게 준 경우

# alpha = [0, 0.1, 1, 10, 100]

# +
## alpha = 0 : 규제X => linear regression과 동일 

alpha = [0, 0.1, 1, 10, 100]
for a in alpha:
    ridge = Ridge(alpha=a)
    cv_training(ridge, boston.data, boston.target, alpha=a)
# -

# #### 각 alpha에 따른 회귀계수 시각화

# +
coef_df = pd.DataFrame(index=boston.feature_names)
# 회귀계수 저장
for a in alpha:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    coef_df['alpha='+str(a)] = ridge.coef_

coef_df
# -

# => ridge 모델은 회귀계수를 0으로 만들진 않음

# +
# 회귀계수 시각화
fig, axs = plt.subplots(figsize=(14,6), nrows=1, ncols=5)
for i, a in enumerate(coef_df.columns):
    axs[i].set_xlim(coef_df.min().min(), coef_df.max().max())
    sns.barplot(coef_df[a], coef_df.index, ax=axs[i])

plt.tight_layout()
plt.show()
# -

# ### 규제가 있는 회귀모델 적용 함수 get_linear_reg_eval() 함수 정의
# - 매개변수로 Ridge / Lasso / ElasticNet을 받아 해당 규제모델을 학습/예측성능 출력

from sklearn.linear_model import Ridge, Lasso, ElasticNet
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


# +
X = bostonDF.drop('price', axis=1)
y = bostonDF.price

get_linear_reg_eval('Ridge', [5,10,20], X, y)
# -

# ## 2. 라쏘 회귀모델

coef_df = get_linear_reg_eval('Lasso', [0.07,0.1,0.5,1,3], X, y)
coef_df


# lasso 모델에서 'NOX'의 회귀계수는 0이 됨

# #### alpha에 따른 회귀계수 값 시각화

def coeffs_plot(coef_df):
    ncols = coef_df.shape[1]
    coef_max = coef_df.max().max()
    coef_min = coef_df.min().min()
    
    fig, axs = plt.subplots(figsize=(14,6), ncols=ncols)
    for i, a in enumerate(coef_df.columns):
        axs[i].set_xlim(coef_min, coef_max)
        sort_coef = coef_df[a].sort_values(ascending=False)
        sns.barplot(sort_coef, sort_coef.index, ax=axs[i])
        
    plt.tight_layout()
    plt.show()


coeffs_plot(coef_df)

# => Lasso 모델은 alpha 값이 커지면서 회귀계수의 값이 0이 되는 피처가 많아짐

# ## 3. 엘라스틱넷 회귀 모델

elastic_alpha = [0.07, 0.2, 0.5, 1, 3]
coef_elastic = get_linear_reg_eval('ElasticNet', elastic_alpha, X, y)

coef_elastic

coeffs_plot(coef_elastic)
