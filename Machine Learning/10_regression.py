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

# # 비용(cost)최소화하는 회귀모델 구하기

# ## 예제 데이터 생성
# - $y = 6 + 4x + noise$

X = np.random.rand(100,1) * 2
y = 6 + 4*X + np.random.randn(100,1) ## 정규난수(randn) 노이즈 생성

# 데이터 산점도
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.show()


# ## 1. 경사하강법

# #### 경사하강법의 일반적인 프로세스
# 1. $w_{0}, w_{1}$을 임의의 값으로 설정하고 첫 비용함수 값 계산
# 2. $w_{1}$를 $w_{1}+η\frac{2}{N}\sum_{i=1}^{N}x_{i}(실제값_{i}-예측값_{i})$으로, $w_{0}$를 $w_{0}+η\frac{2}{N}\sum_{i=1}^{N}(실제값_{i}-예측값_{i})$으로 업데이트한 후 다시 비용함수 값을 계산
# 3. 비용함수의 값이 감소했으면 다시 2번 과정을 반복하고 더 이상 비용함수의 값이 감소하지 않으면 그때의 $w_{1}, w_{0}$를 구하고 반복을 중지

# **$w_{0}$와 $w_{1}$의 값 업데이트 함수 정의**
# - $y = w_0 + w_{1}X$

def get_weight_updates(w1, w0, X, y, learning_rate=0.1):
    N = len(y)
    
    ## w1_update, w0_update를 각각 w1, w0의 shape과 동일한 크리를 가진 0 값으로 초기화
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    
    ## 예측 배열 계산하고 실제 값과의 차이 계산
    y_pred = w0 + np.dot(X, w1.T)
    diff = y - y_pred
    
    ## w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성 (상수 1)
    w0_factors = np.ones((N,1))
    
    ## w1_update와 w0_update 계산
    w1_update = -(2/N)*learning_rate*np.dot(X.T, diff)
    w0_update = -(2/N)*learning_rate*np.dot(w0_factors.T, diff) ## -2/N*learning_rate*(1*diff)
    
    return w1_update, w0_update


# - 임의의 초기값을 주어 첫번째 w0, w1 계산

# +
w0 = np.zeros((1,1))
w1 = np.zeros((1,1))

w1_update, w0_update = get_weight_updates(w1, w0, X, y)
w1_update, w0_update
# -

w1_update, w0_update = get_weight_updates(w1_update, w0_update, X, y)
w1_update, w0_update


# **$w_1$과 $w_0$를 반복적으로 업데이트하는 함수 정의**

def gradient_descent_steps(X, y, iters=100, verbose=False):
    ## w0와 w1은 0로 초기화
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    
    ## 가중치 업데이트
    for i in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
        w1 = w1 - w1_update  ###(업데이트 값이 음수이므로 빼야 더해지는 거임
        w0 = w0 - w0_update
        
        if verbose:
            if i%10 == 0:  ### 10번째마다 출력
                print(f'iter={i+1}: w1={w1[0,0]:.3f}, w0={w0[0,0]:.3f}')
    return w1, w0


# **예측 오차 비용 계산 함수 정의**

def get_cost(y, y_pred):
    N = len(y)
    cost = np.sum(np.square(y-y_pred))/N   ## 오차 제곱합
    return cost


# #### 경사하강법 실행

# +
w1, w0 = gradient_descent_steps(X, y, iters=100, verbose=True)
y_pred = w0[0,0] + w1[0,0]*X

print(f'w1={w1[0,0]:.3f}, w0={w0[0,0]:.3f}')
print(f'경사하강법에 의한 비용 : {get_cost(y, y_pred):.4f}')
# -

# #### 경사하강법으로 구한 선형회귀식 시각화

y0 = 6+4*X


def draw_reg_plot(X, y, y0, y_pred):
    plt.scatter(X,y)
    plt.plot(X,y0,'k',label='y=6+4X')
    plt.plot(X,y_pred, 'r', label='predict')
    plt.legend()
    plt.show()


draw_reg_plot(X,y,y0,y_pred)

# #### 반복횟수 변화에 따른 회귀계수 추정과 비용함수 및 회귀직선 시각화

for iter_n in range(100, 2100, 100):
    if iter_n == 100:
        w1, w0 = gradient_descent_steps(X,y,iters=iter_n)
        y_pred = w0[0,0] + w1[0,0]*X
        cost = get_cost(y, y_pred)
    else:
        pre_cost = get_cost(y, y_pred)
        w1, w0 = gradient_descent_steps(X,y,iters=iter_n)
        y_pred = w0[0,0] + w1[0,0]*X
        cost = get_cost(y, y_pred)
        if np.abs(pre_cost - cost) < 0.001:
            break
    
    print(f'반복횟수 = {iter_n}')
    print(f'w1={w1[0,0]:.3f}, w0={w0[0,0]:.3f}')
    print(f'경사하강법에 의한 비용 : {cost:.4f}')
    draw_reg_plot(X, y, y0, y_pred)

# 경사하강법에 의해 추정된 회귀계수와 예측값
y_pred_GD, w1_GD, w0_GD = y_pred, w1, w0


# ## 2. 미니배치 확률적 경사하강법(Stochastic Gradient Descent)

# **미니 배치 확률적 경사하강법 함수 정의**
# - batch_size 만큼 데이터를 랜덤하게 추출하여 이를 기반으로 w1_update, w0_update 계산

def stochastic_gradient_descent_steps(X, y, batch_size=100, iters=100, verbose=False):
    ## w0와 w1은 0로 초기화
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    prev_cost = 100000
    
    ## 가중치 업데이트
    for i in range(iters):
        np.random.seed(i)
        # 데이터 인덱스 섞음
        stoch_rnd_idx = np.random.permutation(X.shape[0],) ## 주어진 데이터의 경우를 랜덤하게 나열(셔플과 비슷)
        sample_X = X[stoch_rnd_idx][:batch_size]
        sample_y = y[stoch_rnd_idx][:batch_size]
        w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01)
        
        w1 = w1 - w1_update  ###(업데이트 값이 음수이므로 빼야 더해지는 거임
        w0 = w0 - w0_update
        
        if verbose:
            if i%10 == 0:  ### 10번째마다 출력
                print(f'iter={i+1}: w1={w1[0,0]:.3f}, w0={w0[0,0]:.3f}')
    return w1, w0


# #### 미니배치 경사하강법을 이용해 반복횟수 변화에 따른 회귀계수 추정과 비용함수 및 회귀직선 시각화

for iter_n in range(100, 2100, 100):
    if iter_n == 100:
        w1, w0 = stochastic_gradient_descent_steps(X,y,batch_size=1000,iters=iter_n)
        y_pred = w0[0,0] + w1[0,0]*X
        cost = get_cost(y, y_pred)
    else:
        pre_cost = get_cost(y, y_pred)
        w1, w0 = stochastic_gradient_descent_steps(X,y,batch_size=1000,iters=iter_n)
        y_pred = w0[0,0] + w1[0,0]*X
        cost = get_cost(y, y_pred)
        if np.abs(pre_cost - cost) < 0.001:
            break
    
    print(f'반복횟수 = {iter_n}')
    print(f'w1={w1[0,0]:.3f}, w0={w0[0,0]:.3f}')
    print(f'경사하강법에 의한 비용 : {cost:.4f}')
    draw_reg_plot(X, y, y0, y_pred)

# 확률적 경사하강법에 의해 추정된 회귀계수와 예측값
y_pred_SGD, w1_SGD, w0_SGD = y_pred, w1, w0

# ## 3. OLS 정규방정식을 이용한 회귀계수 추정

# $\hat{\theta} = (X^TX)^{-1}X^Ty$

X_con = np.c_[np.ones((100,1)), X] ## 계산하기 편하게 변환
theta_hat = np.linalg.inv(X_con.T.dot(X_con)).dot(X_con.T).dot(y)
theta_hat

# $\hat{y} = X\hat\theta$

# +
X_new = np.array([[0],[2]]) ## x=2 (임의의 x 정해보기)
## |0|
## |2|
X_new_con = np.c_[np.ones((2,1)), X_new]
## |1 0|
## |1 2|

y_pred_OLS = X_new_con.dot(theta_hat)
y_pred_OLS
# -

# #### 정규방정식을 사용한 회귀모델 시각화

y_pred_OLS = X_con.dot(theta_hat)
draw_reg_plot(X, y, y0, y_pred_OLS)

# #### OLS, 경사하강법, 확률적 경사하강법 비교

plt.scatter(X, y)
plt.plot(X, y0, label='y=6+4X')
plt.plot(X, y_pred_GD, label='GD')
plt.plot(X, y_pred_SGD, label='SGD')
plt.plot(X, y_pred_OLS, label='OLS')
plt.legend()
plt.show()

print('y = 6 + 4X')
print(f'경사하강법 : y = {w0_GD[0,0]:.3f} + {w1_GD[0,0]:.3f}X')
print(f'확률적경사하강법 : y = {w0_SGD[0,0]:.3f} + {w1_SGD[0,0]:.3f}X')
print(f'OLS : y = {theta_hat[0,0]:.3f} + {theta_hat[1,0]:.3f}X')

# ## 4. 사이킷런의 LinearRegression 클래스 사용

from sklearn.linear_model import LinearRegression

# +
# 객체 생성
lr = LinearRegression()

# 훈련
lr.fit(X, y)
# -

# #### X=1.5248 값 예측

pred = lr.predict(np.array([[1.5248]]))
pred

# ### 회귀계수 coef_ 속성

# X 계수
lr.coef_

# ### 절편 intercept_ 속성

# y 절편
lr.intercept_

# ### 회귀모델 시각화

y_pred_LR = lr.predict(X)
draw_reg_plot(X, y, y0, y_pred_LR)

# #### 경사하강법, OLS정규방정식,  사이킷런 비교

plt.scatter(X, y)
plt.plot(X, y0, label='y=6+4X')
plt.plot(X, y_pred_GD, label='GD')
plt.plot(X, y_pred_SGD, label='SGD')
plt.plot(X, y_pred_OLS, label='OLS')
plt.plot(X, y_pred_LR, label='LR')
plt.legend()
plt.show()

# ### Scikit-Learn의 SGDRegressor를 이용한 분석

from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(max_iter=2000, tol=1e-3) ## 중지 기준 10에 -3제곱
sgd.fit(X,y)
y_pred_SGD_sci = sgd.predict(X)

draw_reg_plot(X, y, y0, y_pred_SGD_sci)

sgd.coef_

sgd.intercept_

print('y = 6 + 4X')
print(f'경사하강법 : y = {w0_GD[0,0]:.3f} + {w1_GD[0,0]:.3f}X')
print(f'확률적경사하강법 : y = {w0_SGD[0,0]:.3f} + {w1_SGD[0,0]:.3f}X')
print(f'OLS : y = {theta_hat[0,0]:.3f} + {theta_hat[1,0]:.3f}X')
print(f'LinearRegression : y = {lr.intercept_[0]:.3f} + {lr.coef_[0,0]:.3f}X')
print(f'SGDRegressor : y = {sgd.intercept_[0]:.3f} + {sgd.coef_[0]:.3f}X')
