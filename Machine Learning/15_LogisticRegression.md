# 로지스틱 회귀
- 독립변수의 선형결합을 이용해 사건의 발생 가능성 예측
- 종속변수가 두개의 범주를 갖는 범주형인 경우
- 최대우도추정법(MLE)을 사용하여 회귀계수 추정

![logistic regression](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb6RYtA%2FbtqACP40kYg%2FyWYC5rbnWuDwCvbh09zyBK%2Fimg.png)
https://ebbnflow.tistory.com/129

## 로지스틱 회귀 모형

#### 독립변수가 한 개인 경우
$$ln(\frac{p(y=1|x)}{1-p(y=1|x)})=\beta_0+\beta_1x$$
- 좌변 : 로그 오즈(log odds) 또는 **로짓(logit)**
- $x$가 한 단위 증가할 때 로그-오즈가 $\beta$만큼 증가하고, 오즈는 $e^{\beta_1}$배 만큼 증가

### 로지스틱 함수
$$p(x) = \frac{e^{\beta_0+\beta_{1}x}}{1+e^{\beta_0+\beta_{1}x}}$$
- $x$는 $-\infty$와  $\infty$ 사이의 값
- 결과는 0과 1 사이의 값

### 오즈(odds)
$$\frac{p(y=1|x)}{1-p(y=1|x)} = e^{\beta_0+\beta_1x}$$
- 성공확률을 실패확률로 나눈 비
- $\frac{p(y=1|x)}{1-p(y=1|x)}$: 오즈, 0과 $\infty$ 사이의 값

### 회귀계수 추정을 위한 우도 함수
$$l(\beta_0, \beta_1)=\prod_{i:y_i=1}p(x_i)\prod_{i':y_i'=0}(1-p(x_i'))$$
- 추정치 $\beta_0, \beta_1$은 우도함수를 최대화하도록 선택됨

### 회귀계수에 대한 가설검정
- 회귀계수 $\beta_i$에 대한 검정은 와드(Ward)검정 통계량을 이용한 카이제곱검정을 수행
- $H_0$: 개별 회귀계수 $\beta_i$가 0이다.

### 모형에 대한 가설검정
- 모형에 대한 검정은 우도비 검정을 이용
    - 절편($\beta_0$)만 포함한 모형과 적합한 모형을 비교해 절편만 포함한 모형보다 얼마나 유의미한지 비교
- $H_0$: 모형에 포함된 모든 회귀계수는 0이다.($\beta_1=...=\beta_n=0$)

### 다중 로지스틱 회귀 모형
$$ln(\frac{p(y=1|x)}{1-p(y=1|x)}) = \beta_0+\beta_1x_1+...+\beta_nx_n$$


### 로지스틱 회귀모형 특징
- 회귀모형과 오차항에 대한 정규성, 등분산성, 선형성 등의 가정이 없음
- 모형 적합 방법 특성상 소표본인 경우 적합이 잘 되지 않을 수 있음
- 독립변수 간 척도(scale) 차이가 큰 경우 모형 적합이 잘 되지 않을 수 있음
- 모형적합에 주의할 점은 적합에 이용되는 데이터의 반응변수 비율이 치우친 경우 모형 적합이 잘 되지 않을 수 있음

### 로지스틱 회귀 변환과정
① 승산비(오즈비)**(0~$\infty$)**
$$Odds=\frac{p}{1-p}, where p= \beta_0+\beta_1x_1+...+\beta_nx_n$$

② 로짓함수 **($-\infty \sim \infty$)**
- 오즈비를 로그 변환
$$log(Odds) = log(\frac{p}{1-p})$$

③ 역함수 **(0~1)**
- 로짓합수의 역함수
$$logistic(x)=\frac{e^x}{1+e^x}$$