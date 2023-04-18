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

# # House Sales in King County, USA
# - 2014년 5월 ~ 2015년 5월까지 King County 주택가격
# - https://www.kaggle.com/harlfoxem/housesalesprediction

house = pd.read_csv('data/kc_house_data.csv')
house.head(3)

houseDF = house.copy()
houseDF

# ### 전처리

houseDF['soldYear'] = houseDF['date'].str[:4].astype('int64')
houseDF['sold-built_years'] = houseDF['soldYear'] - houseDF['yr_built']
houseDF.drop(['id', 'date'], axis=1, inplace=True)
houseDF

# ### 변수별 히스토그램

houseDF.hist(figsize=(10,10))
plt.tight_layout()
plt.show()

descDF = houseDF.describe().T
descDF

# +
skew = []
kurt = []
for ftr in houseDF.columns:
    skew.append(houseDF[ftr].skew())
    kurt.append(houseDF[ftr].kurt())
    
descDF['skew'] = skew  ## 왜도: 0에 가까울수록 좌우대칭 / 0보다 크면 오른쪽으로 꼬리가 김
descDF['kurt'] = kurt ## 첨도: 클수록 뾰족
descDF[['skew', 'kurt']]


# -

# ### 상관관계 히트맵

def corr_heatmap(df):
    corr_df = df.corr()
    cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
    plt.figure(figsize=(12,12))
    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_df, annot=True, cmap=cmap, mask=mask, linewidths=0.5, fmt='.2f', annot_kws={'size':8})
    plt.show()


corr_heatmap(houseDF)

# #### 주택가격과 상관관계

# +
from scipy import stats

corrs = []
features = houseDF.columns
for ftr in features:
    stat, p = stats.pearsonr(houseDF.price, houseDF[ftr])
    corrs.append(np.round(stat, 3))

corrDF = pd.DataFrame(corrs, index=features, columns=['Correlation'])
corrDF.sort_values('Correlation', ascending=False)
# -

# #### 주요 피처와 주택가격과의 관계(산점도, regplot)

# +
ftr_names = ['sqft_living', 'sqft_above', 'sqft_living15', 'bathrooms', 'sqft_basement', 'bedrooms']

fig, axs = plt.subplots(figsize=(16,8), ncols=3, nrows=2)
for i, ftr in enumerate(ftr_names):
    row = i//3  # 몫
    col = i%3   # 나머지
    sns.regplot(houseDF[ftr], houseDF['price'], ax=axs[row][col])
    
plt.tight_layout()
plt.show()
# -

# **주택가격 예측에 영향이 작은 변수 삭제**
# - price와 상관관계가 매우 약한(corr<0.2) 변수들
# - 'sqft_living'과 상관관계가 강한 변수들
# - 주소관련 변수들
# - 'sold-built_years'를 도출한 변수들

del_ftr = ['yr_renovated','sqft_lot','sqft_lot15','condition',
           'sqft_living15','sqft_above',
           'long', 'zipcode', 'lat',
           'yr_built','soldYear']
houseDF.drop(del_ftr, axis=1, inplace=True)
houseDF.info()

# ### 선형 회귀 모델 학습/예측/평가

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score

# +
X = houseDF.drop('price', axis=1)
y = houseDF.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=205)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE={mse:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, r2-score={r2:.3f}')  # 주택 가격 단위가 높아서 mse가 크게 나옴
# -

# #### 회귀모델 절편, 회귀계수

print(f'{lr.intercept_:.3f}')
print(f'{np.round(lr.coef_,3)}')

coeff = pd.Series(lr.coef_, index=X.columns)
coeff = coeff.sort_values(ascending=False)
coeff

sns.barplot(coeff, coeff.index)
plt.show()


def reg_coffe_plot(model, names):
    coeff = pd.Series(model.coef_, index=names)
    coeff = coeff.sort_values(ascending=False)
    print('회귀계수')
    print(coeff)
    sns.barplot(coeff, coeff.index)
    plt.show()


reg_coffe_plot(lr, X.columns)


# ### 실제값과 예측값의 차이

def get_residuals(y_test, y_pred, top_n=10):
    df = pd.DataFrame(y_test.values, columns=['real'])
    df['pred'] = np.round(y_pred)
    df['abs(resid)'] = np.abs(df['real']-df['pred'])
    print(df.sort_values('abs(resid)', ascending=False)[:top_n])


get_residuals(y_test, y_pred)

# ## 교차검증으로 MSE와 RMSE 측정

# #### cross_val_score() 이용

from sklearn.model_selection import cross_val_score

lr = LinearRegression()
neg_mse = cross_val_score(lr, X, y, cv=5, scoring='neg_mean_squared_error')
## 값이 클수록 지표가 좋은거라고 맞춰주기 위해 neg(-1) 붙임
mse = neg_mse*(-1)
rmse = np.sqrt(mse)
avg_rmse = np.mean(rmse)
print(f'Negative MSE: \n{np.round(neg_mse, 2)}')
print(f'RMSE: \n{np.round(rmse, 2)}')
print(f'평균 RMSE: {avg_rmse:.3f}')

# ------

# ## 주택가격 price 로그변환하여 회귀분석 진행

fig, axs = plt.subplots(figsize=(10,4), ncols=3)
sns.histplot(houseDF.price, ax=axs[0])
sns.boxplot(houseDF.price, ax=axs[1])
sns.distplot(np.log1p(houseDF.price), ax=axs[2])
plt.show()


# #### 회귀모델 성능 평가를 위한 함수 작성
# - RMSE, RMSLE, MAE, R2 계산

def eval_regr(y, pred):
    rmse = np.sqrt(mean_squared_error(y,pred))
    rmsle = mean_squared_log_error(y,pred)
    mae = mean_absolute_error(y,pred)
    r2 = r2_score(y, pred)
    
    print(f'RMASLE: {rmsle:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, r2-score: {r2:.3f}')


# #### 로그변환된 주택가격 회귀모델 학습 및 평가

# +
X = houseDF.drop('price', axis=1)
y = np.log1p(houseDF.price)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=205)

lr_log = LinearRegression()
lr_log.fit(X_train, y_train)
y_pred = lr_log.predict(X_test)

y_test_exp = np.expm1(y_test) # 로그변환된걸 다시 지수변환
y_pred_exp = np.expm1(y_pred)
eval_regr(y_test_exp, y_pred_exp)
# -

# #### 잔차

get_residuals(y_test_exp, y_pred_exp)

# #### 회귀계수

reg_coffe_plot(lr_log, X.columns)


# ### 회귀모델 학습 및 평가 수치 반환 함수 작성

# +
def eval_regr_score(y, pred, is_expm1=False):
    rmse = np.sqrt(mean_squared_error(y,pred))
    mae = mean_absolute_error(y,pred)
    r2 = r2_score(y, pred)
    
    if is_expm1:
        rmsle = mean_squared_log_error(y,pred)
        print(f'RMASLE: {rmsle:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, r2-score: {r2:.3f}')
    else:
        print(f'RMSE: {rmse:.3f}, MAE: {mae:.3f}, r2-score: {r2:.3f}')
        
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
X = houseDF.drop('price', axis=1)

# 원데이터로 회귀분석
y = houseDF.price
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=205)
lr = LinearRegression()
get_model_predict(lr, X_train, X_test, y_train, y_test)

# 로그변환데이터로 회귀분석
y = np.log1p(houseDF.price)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=205)
lr = LinearRegression()
get_model_predict(lr, X_train, X_test, y_train, y_test, is_expm1=True)
# -

# ----

# ## 범주형 변수 원핫인코딩 후 회귀분석 진행

X_ohe = pd.get_dummies(X, columns=['waterfront','view'])
X_ohe

# #### 로그변환, 원핫인코딩한 데이터로 회귀모델 예측

# +
y = np.log1p(houseDF.price)

X_train, X_test, y_train, y_test = train_test_split(X_ohe,y, test_size=0.2, random_state=205)

lr = LinearRegression()
get_model_predict(lr, X_train, X_test, y_train, y_test, is_expm1=True)
# -

reg_coffe_plot(lr, X_ohe.columns)


