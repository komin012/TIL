## 데이터 인코딩

### 1️⃣ 레이블 인코딩
- 문자열 데이터를 숫자로 코드화 (범주형 자료의 수치화)
- 회귀모델에는 사용 불가
- 코드
```python
from sklearn.preprocessing import LabelEncoder

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

# 인코딩된 값을 다시 디코딩
encoder.inverse_transform([0,1,2])
```
<br>

### 2️⃣ 원-핫 인코딩
- 고유 값에 해당하는 컬럼만 1을 표시하고 나머지 컬럼에는 0을 표시
- 범주형 변수를 독립변수로 갖는 회귀분석의 경우 범주형 변수를 dummy변수로 변환
- 코드
```python
from sklearn.preprocessing import OneHotEncoder

## 입력값을 2차원 데이터로 변환해야 함
items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']
items = np.array(items).reshape(-1,1)
encoder = OneHotEncoder()
encoder.fit(items)
labels = encoder.transform(items)

# pandas로 더미변수 만들기
items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']
df = pd.DataFrame({'items' : items})
df.concat(pd.get_dummies(df),axis=1)
```

## 피처 스케일링
- 서로 다른 변수의 값 범위를 일정한 수준으로 맞춤
- 학습 데이터와 테스트 데이터 스케일링 변환 시 학습데이터 스케일러 기준에 따라야 함

### Z-scaling (표준화)
- 정규분포로 변환
- 좌우대칭 파악 가능
- 코드
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_df = scaler.transform(df)
```

### min-max
- 0과 1 사이 값으로 변환
- 음수값이 있으면 -1~1사이의 값으로 변환
- 코드
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)
scaled = scaler.transform(df)
```
