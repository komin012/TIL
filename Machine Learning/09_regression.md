# 회귀분석
- 분류와의 차이점
    - 분류: category 반환
    - 회귀: 숫자값 반환
        - ex: 아파트 값 예측, 매출 증가액 예측

- 유형
    - 선형/비선형 : 회귀계수 결합에 따라
    - 단순/다중 : 독립변수 개수에 따라

- 종류
    - 단순회귀분석
        - 하나의 독립변수로 하나의 종속변수를 설명하는 모형
        - ex. 아버지의 키로 한 자녀의 키를 설명
    - 다중회귀분석
        - 두 개 이상의 독립변수로 하나의 종속변수를 설명하는 모형
        - ex. 아버지와 어머니의 키로 한 자녀의 키를 설명
    - 다항회귀분석
        - 독립변수와 종속변수의 관계를 2차 이상의 함수로 설명
    - 다변량회귀분석
        - 두 개 이상의 종속변수를 사용하는 모형

## 선형 회귀(Linear Regression)
- 모델 함수식을 통해 독립변수 정보에 따른 종속변수의 값을 예측
<br>

**여러개의 독립변수와 한 개의 종속변수 간의 상관관계를 모델링하는 기법**
- $Y_{i} = β_{0} + β_{1}X_{2i} + β_{2}X_{3i} + … + + β_{n}X_{ni}$
    - 독립변수($X_{i}$)
    - 종속변수($Y$)
    - 회귀계수($β_{i}$)
<br>

**실제 값과 예측값의 차이를 최소화하는 직선형 회귀선을 최적화하는 방식**
- 오차 제곱합 최소화
- $e_{i} = y_{i} - \hat{y}_{i}$
- $min \sum{e^2_{i}}$

### 주요 가정
1. 선형성
2. 정규성
3. 등분산성
4. 독립성<br>
$ε_{i}\sim N(0,σ^2)$

### 대표적인 선형회귀모델
- 일반선형회귀 : 규제를 적용하지 않은 모델로 예측값과 실제값의 RSS를 최소화할 수 있도록 최적화
- 릿지(Ridge) : 선형회귀에 L2규제를 추가한 회귀 모델
- 라쏘(Lasso) : 선형 회귀에 L1 규제를 적용한 모델
- 엘라스틱넷(ElasticNet) : L2, L1 규제를 함께 결합한 모델
- 로지스틱회귀(Logistic Regression) : 분류에서 사용되는 선형 모델<br>
※규제 : 일반적인 선형 회귀의 과적합 문제를 해결하기 위해 회귀계수에 패널티 값을 적용하는 것

## 최적의 회귀 모델
- 실제값과 모델 사이의 오류값(잔차)이 최소가 되는 모델
- 오류 값 합이 최소가 될 수 있는 최적의 회귀계수를 찾는 것
- $RSS(w_{0},w_{1}) = \frac{1}{N} \sum_{i=1}^{N} (y_{i} - (w_{0}+w_{1}×x_{i}))^2$<br>
    - 회귀계수(β) 대신 가중치($w$) 사용
    - 머신러닝에서는 비용함수(cost function)라고 하며, 손실함수(loss function)라고도 함

## 비용 최소화하는 회귀모델 구하기

### 1. 경사하강법
비용함수가 최소가 되는 지점의 $w$를 계산
- 2차 함수의 미분값이 1차함수의 기울기가 가장 최소일 때
- 최초의 $w$에서부터 미분을 적용한 뒤 미분값이 계속 감소하는 방향으로 순차적으로 업데이트
- 더 이상 미분된 1차함수의 기울기가 감소하지 않는 지점을 비용함수의 최소인 지점으로 간주하고 $w$ 반환<br>

$\frac{\partial R(w)}{\partial w_{1}} = \frac{2}{N}\sum_{i=1}^{N}-x_{i}×(y_{i}-(w_{0}+w_{1}x_{i})) = -\frac{2}{N}\sum_{i=1}^{N}x_{i}×(실제값_{i}-예측값_{i})$<br>
$\frac{\partial R(w)}{\partial w_{0}} = \frac{2}{N}\sum_{i=1}^{N}-(y_{i}-(w_{0}+w_{1}x_{i})) = -\frac{2}{N}\sum_{i=1}^{N}(실제값_{i}-예측값_{i})$
<br>

#### 경사하강법의 일반적인 프로세스
1. $w_{0}, w_{1}$을 임의의 값으로 설정하고 첫 비용함수 값 계산
2. $w_{1}$를 $w_{1}+η\frac{2}{N}\sum_{i=1}^{N}x_{i}×(실제값_{i}-예측값_{i})$으로, $w_{0}$를 $w_{0}+η\frac{2}{N}\sum_{i=1}^{N}(실제값_{i}-예측값_{i})$으로 업데이트한 후 다시 비용함수 값을 계산
3. 비용함수의 값이 감소했으면 다시 2번 과정을 반복하고 더 이상 비용함수의 값이 감소하지 않으면 그때의 $w_{1}, w_{0}$를 구하고 반복을 중지

### 2. 미니배치 확률적 경사 하강법(Stochastic Gradient Descent)
- 일부 데이터만을 이용해 $w$가 업데이트되는 값을 계산
- 경사하강법에 비해 빠른 속도를 보장

### 3. OLS 정규방정식을 이용하여 비용최소화하는 회귀계수 추정
$(\hat{θ}) = (X^TX)^{-1}X^Ty$<br>
$\hat{y} = X\hat{θ}$