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
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# # 다항 회귀 (Polynomial Regression)
# - 일차식이 아닌 2차, 3차식 등으로 표현되는 회귀식

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)  ## 2차식
X = np.arange(6).reshape(3,2)  ## [x1, x2]
X

poly.fit_transform(X)

# $1+X_1+X_2+X_1^2+X_1X_2+X_2^2$<br>

X2 = np.arange(9).reshape(3,3)
X2

poly.fit_transform(X2)

# $1+X_1+X_2+X_3+X_1^2+X_1X_2+X_1X_3+X_2^2+X_2X_3+X_3^2$

poly3 = PolynomialFeatures(degree=3) ## 3차식
poly3.fit_transform(X)


# $1+X_1+X_2+X_1^2+X_1X_2+X_2^2+X_1^3+X_1^2X_2+X_1X_2^2+X_2^3$<br>

# ### 3차 다항식 결정값을 구하는 함수 polynomial_func(x) 생성
# $y=1+2x_1+3x_1^2+4x_2^3$

def polynomial_func(X):
    y = 1 + 2*X[:,0] + 3*X[:,0]**2 + 4*X[:,1]**3
    print(f'x1 = {X[:,0]}')
    print(f'x2 = {X[:,1]}')
    return y 


X = np.arange(4).reshape(2,2)
X

y = polynomial_func(X)
print(f'삼차다항식 결정값: {y}')

# ### 3차 다항식 계수의 피처값과 3차 다항식 결정값으로 학습

from sklearn.linear_model import LinearRegression

X = np.arange(4).reshape(2,2)
print(f'X:\n{X}')
poly_ = PolynomialFeatures(degree=3).fit_transform(X)
print(f'3차 다항식 계수(features):\n{poly_}')

model = LinearRegression()
model.fit(poly_, y)
print(f'다항식 회귀계수(coefficients):\n{np.round(model.coef_,2)}')

# ## 파이프라인(Pipeline)을 이용한 3차 다항회귀 학습
# 연속된 변환을 순서대로 처리할 수 있도록 도와주는 클래스

from sklearn.pipeline import Pipeline

X = np.arange(4).reshape(2,2)
y = polynomial_func(X)
y

# +
model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression())])

models = model.fit(X,y)
models.named_steps['linear'].coef_
# -

# ## 가상 데이터 생성 및 다항 회귀 추정

# #### 가상 데이터 생성

n = 100
X = 6*np.random.rand(n, 1)-3 ## 6*(0~1)-3 => 0~3 사이 데이터 100개
y = 0.5*X**2 + X + 2 + np.random.randn(n,1)

# #### scatterplot

plt.scatter(X,y)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# - 비선형적으로 분포하고 있는 데이터에 단순히 직선으로 예측하는 것은 맞지 않음 <br>
# => 다항식 사용

import statsmodels.api as sm

model = sm.OLS(y, X)
result = model.fit()
result.summary()


## 잔차그래프 그리기
def draw_resid_plot(fitted, resid):
    plt.figure(figsize=(6,10))
    plt.scatter(fitted, resid)
    xmin = np.floor(plt.xlim()[0])
    xmax = np.round(plt.xlim()[1])
    plt.hlines(0, xmin, xmax, colors='gray', ls='--')
    plt.title('Fitted Values vs Residuals')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()


draw_resid_plot(result.fittedvalues, result.resid)

# => 2차식으로 ㄱㄱ

# #### 다항식으로 변환

poly = PolynomialFeatures(degree=2)
Z = poly.fit_transform(X)

# #### LinearRegression 적용

lr = LinearRegression()
lr.fit(Z, y)
print(f'X절편:{lr.intercept_[0]:.3f}, \n회귀계수:{np.round(lr.coef_, 3)}')

# #### 다항 회귀식 시각화

# +
X_new = np.linspace(-3,3,100).reshape(100,1)
X_new_poly = poly.transform(X_new)
y_new = lr.predict(X_new_poly)

plt.scatter(X,y)
plt.plot(X_new, y_new, 'r')
plt.show()
