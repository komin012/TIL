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

# # 타이타닉 생존자 예측
# - 캐글에서 제공하는 타이타닉 탑승자 데이터 이용
# - https://www.kaggle.com/c/titanic/data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ## 1. 데이터 파악

titanic = pd.read_csv('data/titanic/train.csv')
titanic.head()

titanic.info()

titanic.describe()

## object 변수까지 모두 나타내기
titanic.describe(include='all')

# ## 2. 데이터 전처리

# ### 1) 결측치 파악

titanic.isnull().sum()

# ### 2) 결측치 처리

# #### Age

titanic['Age'].fillna(np.mean(titanic['Age']), inplace=True)

titanic['Age'].isnull().sum()

# #### Cabin

titanic['Cabin'].unique()

titanic['Cabin'].fillna('N', inplace=True)

titanic['Cabin'] = titanic['Cabin'].str[:1]

titanic['Cabin'].unique()

# #### Embarked

titanic['Embarked'].unique()

titanic['Embarked'].fillna('N', inplace=True)

titanic['Embarked'].unique()

# ### 3) 성별에 따른 생존자 수 분포 (EDA)

titanic.groupby(['Sex','Survived']).Survived.count()

# ### 3-1) 검정
# - 성별에 따른 생존자 분포 검정

# H0: 성별에 따라 생존 여부 분포가 같다.
# - 범주형 변수 분석 -> 카이제곱 검정
# - 동질성 검정

# +
from scipy import stats

titanic_ss = titanic.groupby(['Sex','Survived']).Survived.count()
titanic_ss = titanic_ss.unstack()

stat, p, df, f_exp = stats.chi2_contingency(titanic_ss)
print(f'검정통계량:{stat}, p-value:{p}, 자유도:{df}')
print(f'기대도수:\n{f_exp}')
# -

# => 유의수준 5%하에서 p값이 0.05보다 작기 때문에 귀무가설 기각 <br>
# => 성별에 따른 생존자 수 분포가 같다고 할 수 없다.

# ### 3-2) 시각화

sns.barplot(data=titanic, x='Sex', y='Survived')
plt.show()

# ### 4) 수치형 변수의 구간화

# #### 연령대에 따른 생존자수 분포

labels = ['baby','child','teenager','student','young adult','adult','elderly']
titanic['age_cat'] = pd.cut(titanic['Age'],
                           [0,5,12,18,25,35,60,100],
                           labels=labels)
titanic.head()

titanic['age_cat'].value_counts().sort_index()

# ### 4-1) 검정
# - 범주형 변수 검정 -> 카이제곱 검정
# - 분포가 동일한지 검정 -> 동질성 검정

# 연령대에 따른 생존
titanic_as = titanic.groupby(['age_cat','Survived']).Survived.count()
titanic_as = titanic_as.unstack()
titanic_as

stat, p, df, exp_f = stats.chi2_contingency(titanic_as)
print(f'검정통계량:{stat}, p-value:{p}, 자유도:{df}')
print(f'기대도수:\n{exp_f}')

# => 유의수준 5%하에서 p값이 0.05보다 작기 때문에 귀무가설 기각 <br>
# => 연령대별로 생존자 수 분포가 같다고 할 수 없다.

# ### 4-2) 시각화

sns.barplot(data=titanic, x='age_cat', y='Survived')
plt.show()

## 연령대별 성별 생존비율
sns.barplot(data=titanic, x='age_cat', y='Survived', hue='Sex')
plt.show()

# ### 5) 인코딩
# - Sex, Cabin, Embarked, age_cat

# #### LabelEncoder

titanic_le = titanic.copy()

from sklearn.preprocessing import LabelEncoder

# +
labels = ['Sex', 'Cabin', 'Embarked', 'age_cat']
for label in labels:
    encoder = LabelEncoder()
    titanic_le[label] = encoder.fit_transform(titanic_le[label])

titanic_le.head()
# -

# #### OneHotEncoder

titanic_ohe = titanic.copy()

from sklearn.preprocessing import OneHotEncoder

# +
labels = ['Sex', 'Cabin', 'Embarked', 'age_cat']
for label in labels:
    encoder = OneHotEncoder(sparse=False)
    ohe = encoder.fit_transform(titanic_ohe[[label]])
    label_df = pd.DataFrame(ohe,columns=[label+'_'+cat for cat in encoder.categories_[0]])
    titanic_ohe = pd.concat([titanic_ohe.drop(label, axis=1), label_df], axis=1)

titanic_ohe
# -

titanic_ohe.info()

# ### 6) 열 삭제
# - PassengerId, Name, Ticket

titanic_le.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)
titanic_ohe.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)

# ## 3. 생존자 예측 모델링

# ### 1) 데이터 분할

# +
## 피처와 라벨 지정
X_le = titanic_le.drop('Survived', axis=1)
y_le = titanic_le['Survived']

X_ohe = titanic_ohe.drop('Survived', axis=1)
y_ohe = titanic_ohe['Survived']

# +
from sklearn.model_selection import train_test_split

X_train_le, X_test_le, y_train_le, y_test_le = train_test_split(X_le, y_le, test_size=0.2, random_state=123)
X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_ohe, y_ohe, test_size=0.2, random_state=123)
# -

# ### 2) 학습 / 예측 / 평가
# - DecisionTreeClassifier
# - RandomForestClassifier
# - LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# +
# 모델 객체 생성
dt = DecisionTreeClassifier(random_state=123)
rf = RandomForestClassifier(random_state=123)
lr = LogisticRegression(random_state=123)

# 학습
dt.fit(X_train_le, y_train_le)
rf.fit(X_train_le, y_train_le)
lr.fit(X_train_le, y_train_le)

# 예측
dt_pred_le = dt.predict(X_test_le)
rf_pred_le = rf.predict(X_test_le)
lr_pred_le = lr.predict(X_test_le)

# 평가
dt_acc_le = accuracy_score(y_test_le, dt_pred_le)
rf_acc_le = accuracy_score(y_test_le, rf_pred_le)
lr_acc_le = accuracy_score(y_test_le, lr_pred_le)

print('Label Encoding')
print(f'DecisionTreeCalssifier 정확도 : {dt_acc_le}')
print(f'RandomForestCalssifier 정확도 : {rf_acc_le}')
print(f'LogisticRegression 정확도 : {lr_acc_le}')
# -

# => RandomForest 정확도가 가장 높음 <br>
# ** Label Encoding은 회귀모델(LogisticRegression)에 적합X

# +
# 모델 객체 생성
dt = DecisionTreeClassifier(random_state=123)
rf = RandomForestClassifier(random_state=123)
lr = LogisticRegression(random_state=123)

# 학습
dt.fit(X_train_ohe, y_train_ohe)
rf.fit(X_train_ohe, y_train_ohe)
lr.fit(X_train_ohe, y_train_ohe)

# 예측
dt_pred_ohe = dt.predict(X_test_ohe)
rf_pred_ohe = rf.predict(X_test_ohe)
lr_pred_ohe = lr.predict(X_test_ohe)

# 평가
dt_acc_ohe = accuracy_score(y_test_ohe, dt_pred_ohe)
rf_acc_ohe = accuracy_score(y_test_ohe, rf_pred_ohe)
lr_acc_ohe = accuracy_score(y_test_ohe, lr_pred_ohe)
print('One-Hot Encoding')
print(f'DecisionTreeCalssifier 정확도 : {dt_acc_ohe}')
print(f'RandomForestCalssifier 정확도 : {rf_acc_ohe}')
print(f'LogisticRegression 정확도 : {lr_acc_ohe}')
# -

# => RandomForest의 정확도가 가장 높음

# ### 3) 교차검증을 통한 모델 성능 향상

# #### cross_val_score()
# - stratified kfold 적용

from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, X_train_ohe, y_train_ohe, scoring='accuracy')
print(scores)
print(np.mean(scores))

# GridSearchCV()

from sklearn.model_selection import GridSearchCV

# +
params = {'max_depth':[2,3,5,10],
          'min_samples_split':[2,3,5],
         'min_samples_leaf':[1,5,8]}
## min_samples_split : 노드를 분할하기 위한 최소한의 샘플 데이터 수
##                     default=2, 작게 설정할 수록 과적합 가능성 증가
## min_samples_leaf : 말단 노드가 되기 위한 최소한의 샘플 데이터 수

grid = GridSearchCV(rf, param_grid=params, scoring='accuracy')
grid.fit(X_train_ohe, y_train_ohe)
print(f'GridSearchCV 최적 하이퍼파라미터 : {grid.best_params_}')
print(f'GridSearchCV 최고 정확도 : {grid.best_score_}')
# -

pd.DataFrame(grid.cv_results_)

# 최적의 파라미터로 학습된 estimator로 예측/평가
best = grid.best_estimator_
best_pred = best.predict(X_test_ohe)
accuracy_score(y_test_ohe, best_pred)


