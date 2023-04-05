# 머신러닝 개요

### 머신러닝 워크플로우
1. 데이터 수집
2. 데이터 전처리
3. 데이터 분할 (훈련용-검증용-테스트용)
4. 데이터 훈련
5. 데이터 검증 및 테스트

---

# 분류
- **feature** : 독립변수와 같은 역할
- **label** : 종속변수와 같은 역할

## 1️⃣ kNN 알고리즘
- 분류와 회귀 모두 활용
    - 분류 : k개의 최근접 이웃들의 class들 중 다수결 결과로 class 예측
    - 회귀 : k개 최근접 이웃들이 가지고 있는 값의 평균을 결과값으로 예측
- 비모수 방식
- k : 대개 홀수
    - k가 작으면 과대적합
    - k가 크면 과소적합

- 코드
    ```python
    from sklearn.neighbors import KNeighborsClassfier
    
    ## 객체 생성
    kn = KNeighborsClassfier()

    ## 훈련
    kn.fit(X, y)

    ## 평가 -> 정확도
    kn.score(X,y)

    ## 예측
    kn.predict([[x,y]])
    ```
    - k 지정
    ```python
    kn = KNeighborsClassfier(n_neighbors=k)
    ```
---
<br>

## 2️⃣ 데이터 분할
- **홀드-아웃** : 데이터가 충분히 클때
- **교차검증(CV)** : 충분히 큰 데이터가 없을 때
- **Stratified Sampling** : 필요에 따라

### 1. 홀드-아웃 방식
- Train / Test
- Train / Validation / Test
- 7:3 ~ 9:1 널리 사용
- 데이터셋이 편향되게 분할하면 안됨
    - ex. train set에는 'A'데이터만 들어가고 test set에는 'B'데이터만 들어가는 경우 제대로된 성능이 나오지 않음
    - 데이터를 섞어서 (shuffle) 분할해야 함
- 코드
```python
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, shuffle=True)
### test_size 기본값 25%
```

#### 데이터 표준화
- features 간의 숫자 차이가 나는 경우 숫자가 큰 특성의 영향이 커지므로 표준화하여 맞추어야 함.
```python
mean = np.mean(train_X, axis=0)
std = np.std(train_X, axis=0)
train_scaled = (train_X - mean) / std

## 표준화 데이터로 훈련
kn.fit(train_scaled, train_y)

## test 데이터도 표준화 (train 데이터의 평균, 표준편차로)
test_scaled = (test_X - mean) / std

kn.score(test_scaled, test_y)
```