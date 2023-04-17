# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# ## 데이터 전처리

df = pd.read_csv('data/kc_house_data.csv')
df.head(3)

# ### 변수설명
# #### 독립변수(X)
# - id : 고유한 id 값
# - date: 집이 매각된 날짜
# - bedrooms: 침실 수 
# - bathrooms: 욕실 수
# - sqft_living: 집의 평방 피트
# - sqft_lot: 부지의 평방 피트
# - floors: 집의 총 층 수
# - waterfront: 물가가 보이는 집
# - condition: 상태가 얼마나 좋은 지 여부
# - grade: 주택에 부여되는 등급
# - sqft_above: 지하실을 제외한 집의 평방 피트
# - sqft_basement: 지하실의 평방 피트
# - yr_built: 지어진 연도
# - yr_renovated: 리모델링된 연도
# - lat: 위도 좌표
# - long: 경도 좌표
# - sqft_living15: 2015년 당시 거실 면적(일부 개조를 의미하고, 부지 면적에 영향을 미칠 수도 있고 아닐 수도 있음)
# - sqft_lot15: 2015년 당시 부지 면적(일부 개조를 의미함)
#
# #### 종속변수(y)
# - price: 주택 가격

# ### 결측치 확인

df.isnull().sum()

# ## 탐색적 데이터 분석(EDA)

df.describe()

# ### 종속변수(price) 확인

df.price.describe()

df.price.hist()

# 왜도 : 0에 가까우면 정규분포 형태, 0보다 크면 오른쪽으로 꼬리가 긴 형태, 0보다 작으면 왼쪽으로 꼬리가 김
df.price.skew()

# #### => price 로그변환

np.log1p(df.price).hist()

np.log1p(df.price).skew()

# ### date 데이터 처리

df['date2'] = df.date.str[:4].astype('int64')
df.date2

# ### sold-built_years 변수 추가
# - date2(매각년도) - yr_built(지어진 년도)

df['sold-built_years'] = df['date2'] - df['yr_built']
df['sold-built_years']

df['sold-built_years'].hist(bins=20)

# ## 시각화를 통해 데이터 특징 파악

# ### 분석에 필요한 변수들만 선택

# id랑 주소관련 컬럼 제외
ftr_names = df.columns.difference(['id', 'date', 'lat', 'long', 'zipcode']) ## 빼고 가져오기
df_reg = df[ftr_names]
df_reg.columns

# ### 변수별 분포(히스토그램)

df_reg.hist(figsize=(22,18), density=True)
plt.tight_layout()
plt.show()

# ### 산점도
# - price와 선형관계가 있을 거 같은 변수 몇개만 그려보기

ftr_import = ['price', 'bedrooms', 'sqft_living', 'sqft_lot', 'yr_built', 'sold-built_years', 'view']
df_import = df_reg[ftr_import]

sns.pairplot(df_import, diag_kind='kde')
plt.show()

sns.pairplot(df_import, diag_kind='kde', corner=True)
plt.show()

# ### 히트맵 - 상관관계 분석

df_corr = df_import.corr()
df_corr

cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
cmap

sns.heatmap(df_corr, annot=True, fmt='.2f', annot_kws={'size':10}, cmap=cmap)
plt.show()

n = df_corr.shape[1]  # 열 개수
mask = np.zeros((n,n), dtype=np.bool)
mask

mask[np.triu_indices_from(mask)] = True # 대각행렬의 상단만 가져오기
mask

sns.heatmap(df_corr, annot=True, fmt='.2f', annot_kws={'size':10}, cmap=cmap, mask=mask)
plt.show()

# => price와 상관관계가 강한 독립변수 : sqft_living (집크기)

df_corr.price.sort_values(ascending=False)

# #### 전체변수들에 대한 상관계수

df.drop('id', axis=1, inplace=True)
df_corr_all = df.corr()

mask = np.zeros_like(df_corr_all, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,12))
sns.heatmap(df_corr_all, annot=True, fmt='.2f', annot_kws={'size':10}, cmap=cmap, mask=mask)
plt.show()

plt.figure(figsize=(12,6))
sns.distplot(df['sqft_living'], hist=True, norm_hist=False, kde=False, label='sqft_living')
sns.distplot(df['sqft_living15'], hist=True, norm_hist=False, kde=False, label='sqft_living15')
sns.distplot(df['sqft_above'], hist=True, norm_hist=False, kde=False, label='sqft_above')
plt.legend()
plt.show()

# => 세 변수(sqft_living, sqft_above, sqft_living15)는 비슷하여 다중공선성이 발생할 가능성이 높음

# ## statsmodels를 이용한 회귀분석

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ### 단순 선형 회귀
# - 독립변수(X) : sqft_lving
# - 종속변수(y) : price

# #### 시각화

sns.jointplot(x='sqft_living', y='price', data=df_reg, kind='reg')
plt.show()

# #### OLS model fit

X = df_reg['sqft_living']
y = df_reg['price']

# 방법1
X = sm.add_constant(X, has_constant='add') # 앞 열에 상수열 추가
X

# +
# 단순회귀모델 객체 생성
model = sm.OLS(y, X)

# 훈련
result = model.fit()

# 모델 결과
result.summary()
# -

# - (3) $R^2$ : 모델 설명력, 클수록 좋음
#     - 독립변수가 늘어나면 따라서 증가하는 단점 있음
# - (3) $Adj. R^2$ : $R^2$의 단점 보완
# - (1) H0 : 모델이 유의미하지 않다
#     - F값의 p-value가 작으므로 귀무가설 기각 => 모델 유의미
# - AIC : 모델 성능 지표, 작을수록 좋음
# - BIC : AIC 살짝 수정한거
#
# - (2) H0: $b_0 = 0$ 
#     - const의 p-value가 작으므로 귀무가설 기각 => $b_0$ ≠ 0
# - (2) H0: $b_1 = 0$ 
#     - sqft_living의 p-value가 작으므로 귀무가설 기각 => $b_1$ ≠ 0
#
# - (4) Omnibus : 클수록 정규성 따름
#     - H0: 정규분포를 따른다.
# - (4) Durbin-Watson : 1.5~2.5이면 독립성 만족, 0 또는 4에 가까울수록 독립성 불만족
# - Skew: 왜도, 0에 가까울수록 좌우대칭
# - Kurtosis: 첨도, 0에 가까울수록 정규분포 형태
# - (4) Jarque-Bera : 클수록 정규성 만족
# - Cond. No. : 10 이상일 경우 다중공선성 문제있음
#     - 변수가 하나일 땐 의미x

# +
# 방법2
# 객체 생성
model2 = ols('price ~ sqft_living', data=df_reg)

# 학습
result2 = model2.fit()

# 결과
result2.summary()
# -

# #### 추정된 절편과 회귀계수

result.params

# #### 추정된 회귀식

print(f'y = {result.params.const:.4f} + {result.params.sqft_living:.4f}X')

# #### 잔차 확인

result.resid

# #### 예측값($\hat{y})$

result.fittedvalues

# #### 예측값과 잔차의 산점도

plt.scatter(result.fittedvalues, result.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# ### 회귀모형의 가정진다
# - 모형의 선형성
# - 잔차의 정규성
# - 잔차의 등분산성
# - 잔차의 독립성

# #### 모형의 선형성
# - 독립변수와 종속변수의 산점도, 상관계수
# - 해결방안 : 변수변환, 비선형 모형 적합

# #### 잔차의 정규성
# - 잔차가 정규분포를 따르는지 검정
# - Q-Q plot: y=x 직선 위에 놓이는지 확인
# - 첨도, 왜도, 정규성 검정(Shapiro-Wilk / KS test)
# - 해결방안 : 변수변환, 새로운 변수 추가, 모형 수정

qqplot = sm.qqplot(result.resid, line='s')


def normal_test(data):
    stat, p = stats.shapiro(data)
    print(f'Shapiro 검정통계량 = {stat:.4f}, p-value={p:.4f}')
    
    if p<0.05:
        print('정규성을 만족하지 않음')
    else:
        print('정규성을 만족함')


normal_test(result.resid)


# => 정규성 불만족

# #### 잔차의 등분산성
# - 잔차의 분산이 동일한지 검정
# - 예측값과 잔차에 대한 산점도를 확인
# - 어떤 패턴을 띠지 않아야 함
# - 해결방안 : 변수변환, 가중회귀분석

def draw_resid_plot(fitted, resid):
    plt.figure(figsize=(6,6))
    plt.scatter(fitted, resid)
    xmin = np.floor(plt.xlim()[0])
    xmax = np.round(plt.xlim()[1])
    plt.hlines(0, xmin, xmax, colors='gray', ls='--')
    plt.title('Fitted values vs Residuals')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.show()


draw_resid_plot(result.fittedvalues, result.resid)

# => 나팔 모양으로 나타나는 이분산<br>
# => 종속변수를 로그변환해야 함

# #### 잔차의 독립성
# - 잔차들이 상관성이 있는지 확인
# - 잔차를 시계열로 가정하여 자기상관계수를 구하여 진단
# - 더빗왓슨 검정: 0이면 양의 자기상관, 2이면 독립, 4이상이면 음의 자기상관
# - 잔차그래프, ACF그래프
# - 해결방안 : ARIMA와 같은 시계열 모형 이용

result.resid.plot()
plt.show()

plt.figure(figsize=(8,8))
sm.graphics.tsa.plot_acf(result.resid)
plt.show()

# - Durbin-Watson 통계량 = 1.983

# => 독립성 만족

# ----

# ### 문제.
# price를 log변환한 뒤 단순회귀분석 수행

log_y = np.log1p(y)

model3 = sm.OLS(log_y,X)
result3 = model3.fit()
result3.summary()

# #### 정규성

qqplot = sm.qqplot(result3.resid, line='s')

# - Omnibus: p-value가 유의수준보다 크므로 귀무가설 기각하지 않음
# - Jarque-Bera (JB):	p-value가 유의수준보다 크므로 귀무가설 기각하지 않음<br>
# => 정규성을 따름<br>
# ※ Shapiro 검정은 크기가 큰 데이터에는 맞지 않음

# #### 독립성

result.resid.plot()
plt.show()

sm.graphics.tsa.plot_acf(result.resid)
plt.show()

# - Durbin-Watson: 1.979

# => 독립성 만족

# #### 등분산성

draw_resid_plot(result3.fittedvalues, result3.resid)

# => 잔차 그래프에서 특정한 패턴이 보이지 않으므로 등분산성 만족

# ----

# # 다중 선형 회귀
# 1. 모든 독립변수(id 제외)를 사용해서 다중선형회귀분석
# 2. 선별한 독립변수들(bedrooms, sqft_living, waterfront, view, sold-built_years)로 다중선형회귀분석

# ## 1. 모든 독립변수를 사용하여 회귀분석 수행

df.columns

X = df.drop(['date', 'price'], axis=1)
X = sm.add_constant(X, has_constant='add')
y = df.price

# ### 다중 회귀 모델 적합

model_all = sm.OLS(y,X)
result_all = model_all.fit()
result_all.summary()

# ### 다중공선성(Multicollinearity)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# +
X_ = X.drop('const', axis=1)

df_vif = pd.DataFrame()
df_vif['features'] = X_.columns
df_vif['VIF'] = np.round([variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])], 3)
df_vif
# -

# - vif > 10 : 다중공선성 O
# - vif > 5 : 주의 요함

# => 독립변수 정보 중 상관관계를 가지는 변수들이 많아 다중공선성 발생

# ## 2. 선별된 독립변수들을 사용한 다중 선형회귀

# - 종속변수(price) : 로그변환
# - 독립변수
#     - bedrooms(연속형)
#     - sqft_living(연속형)
#     - waterfront(범주형)
#     - view(범주형)
#     - sold-built_years: date-yr_built

X = df[['bedrooms','sqft_living','waterfront','view','sold-built_years']]
X.head()

X = sm.add_constant(X, has_constant='add')
y = np.log1p(df['price'])

model4 = sm.OLS(y, X)
result4 = model4.fit()
result4.summary()

# - Omnibus, Jarque-Bera : p-value가 작으므로 귀무가설 기각=> 정규성 만족X
# - Durbin-Watson: 2에 가까우므로 독립성 만족

# ### 다중공선성 확인

# +
X_ = X.drop('const', axis=1)

df_vif = pd.DataFrame()
df_vif['features'] = X_.columns
df_vif['VIF'] = np.round([variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])],3)
df_vif
# -

# ### 회귀모형 가정 진단

# #### 정규성검정

qqplot = sm.qqplot(result4.resid, line='s')

sns.histplot(result4.resid)
plt.show()

# #### 독립성 검정

result4.resid.plot()
plt.show()

sm.graphics.tsa.plot_acf(result4.resid)
plt.show()

# #### 등분산성

draw_resid_plot(result4.fittedvalues, result4.resid)
