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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# -

# # 예제1. iris data
# - 4개의 속성을 2개의 PCA 차원으로 압축

from sklearn.datasets import load_iris

iris = load_iris()
iris.feature_names

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_df = pd.DataFrame(iris.data, columns=col_names)
iris_df['target'] = iris.target
iris_df

# #### sepal_length, sepal_width 두 개의 속성으로 데이터 산포 시각화

sns.scatterplot(data=iris_df, x='sepal_length', y='sepal_width', hue='target')
plt.legend(['setosa', 'versicolor', 'virginica'])
plt.show()

# +
# matplotlib으로 산점도 그리기
for i in range(3):
    plt.scatter(iris_df[iris_df.target==i].sepal_length, iris_df[iris_df.target==i].sepal_width, label=iris_df.columns[i])

plt.legend()
plt.show()
# -

sns.pairplot(iris_df)
plt.show()

# ## PCA로 4개 속성을 2개로 압축하여 분포 시각화
# - PCA는 여러 속성의 값을 연산해야 하므로 속성의 스케일에 영향을 받음
# - 따라서 PCA 적용 전 개별 속성 스케일링 변환 작업 필요
#     - StandardScaler 이용

# #### 평균이 0, 분산이 1인 정규 분포로 원본 데이터를 변환

from sklearn.preprocessing import StandardScaler

iris_scaled = StandardScaler().fit_transform(iris.data)
iris_scaled[:10]

# 평균 0
np.mean(iris_scaled, axis=0)

# 표준편차 1
np.std(iris_scaled)

iris_scaled.shape

# ### PCA 변환

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_scaled)
iris_pca.shape

iris_pca_df = pd.DataFrame(iris_pca, columns=['Comp1', 'Comp2'])
iris_pca_df['target'] = iris.target
iris_pca_df

# #### PCA로 차원 축소된 피처들로 데이터 산포도 시각화

sns.scatterplot(data=iris_pca_df, x='Comp1', y='Comp2', hue='target')
plt.legend(['setosa', 'versicolor', 'virginica'])
plt.show()

# [PCA 결과]
# - 주성분 1(Comp1)축을 기반으로 setosa 품종은 명확하게 구분됨
# - versicolor와 virginica의 경우 겹치는 부분이 있으나 이 두 품종도 주성분 1(Comp1)로 구분 가능
#
# => 주성분 1이 원본데이터의 변동성을 가장 잘 반영한 축임

# #### 각 PCA Component별 변동성 비율
# - PCA Component별로 원본 데이터의 변동성 반영 정도 확인
# - PCA 객체의 explained_variance_ratio_ 속성
#     - 전체 변동성에서 개별 PCA Component별로 차지하는 변동성 비율 제공

pca.explained_variance_ratio_

# - 주성분 1은 전체 변동성의 약 73%
# - 주성분 2는 전체 변동성의 약 23%
# - 두 성분의 변동성은 전체의 약 96%

# ### 원본 데이터와 PCA 변환된 데이터 기반에서 예측 성능 비교

# - estimator: RandomForestClassifier
# - cross_val_score()이용
#     - cv=3, 정확도로 결과 비교

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# +
# 원본데이터
model = RandomForestClassifier(random_state=205)

scores = cross_val_score(model, iris.data, iris.target, cv=3)
print(scores)
print(np.mean(scores))

# +
# PCA 변환
model = RandomForestClassifier(random_state=205)

scores = cross_val_score(model, iris_pca, iris.target, cv=3)
print(scores)
print(np.mean(scores))
# -

# ----

# # 예제2. 신용카드 고객 데이터
# - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

card_df = pd.read_excel('data/pca_credit_card.xls', sheet_name='Data', header=1)
card_df.head()

card_df.set_index('ID', inplace=True)
card_df.head()

card_df.info()

# #### 변수설명
#
# - LIMIT_BAL: Amount of the given credit (NT dollar)
#     - it includes both the individual consumer credit and his/her family (supplementary) credit.
#
# - SEX: 1 = male; 2 = female
#
# - EDUCATION: 1 = graduate school; 2 = university; 3 = high school; 4 = others
#
# - MARRIAGE: 1 = married; 2 = single; 3 = others
#
# - AGE: year
#
# - PAY_0 ~ PAY_6: 과거 월별 상환 내역(2005년 4월(April)에서 9월(September))
#     - 상환상태
#         - -1 : pay duly, 1 : 1개월 연체, 2: 2개월 연체, 8: 8개월 연체, 9: 9개월 이상 연체
#     - PAY_0 : 2005년 9월 상환 상태 
#     - PAY_2 : 2005년 8월 상환 상태
#     - PAY_3 : 2005년 7월 상환 상태
#     - ...
#     - PAY_6 : 2005년 4월 상환 상태
#
# - BILL_AMT1 ~ BILL_AMT6: 청구 금액(NT dollar)
#     - BILL_AMT1 : 2005년 9월(September) 청구 금액
#     - BILL_AMT2 : 2005년 8월(August) 청구 금액
#     - ... 
#     - BILL_AMT6 : 2005년 4월(April) 청구 금액
#
# - PAY_AMT1 ~ PAY_AMT6: 지불 금액(NT dollar)
#     - PAY_AMT1 : 2005년 9월(September) 지불 금액
#     - PAY_AMT2 : 2005년 8월(August) 청구 금액
#     - ... 
#     - PAY_AMT6 : 2005년 4월(April) 청구 금액

# **Target**
# - default payment next month : 다음달 연체 여부
#     - 연체 : 1
#     - 정상 납부 : 0

# #### 컬럼명 변경

card_df.rename(columns={'PAY_0':'PAY_1', 'default payment next month':'default'}, inplace=True)
card_df.info()

# #### 데이터 조작
# - default컬럼을 y변수로 별도 저장
# - default컬럼을 제외한 피처 데이터는 별도의 DataFrame으로 생성

X = card_df[card_df.columns.difference(['default'])]
y = card_df.default

#타겟 변수 비율
y.value_counts()/y.value_counts().sum()*100


# #### 피처간 상관도 시각화

# 상관관계 히트맵 함수
def corr_heatmap(df):
    corr_df = df.corr()
    cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
    plt.figure(figsize=(12,12))
    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_df, annot=True, cmap=cmap, mask=mask, linewidths=0.5, fmt='.2f', annot_kws={'size':8})
    plt.show()


corr_heatmap(X)

# [결과] 
# - BILL_AMT1 ~ BILL_AMT6 : 상관계수가 대부분 0.9 이상으로 매우 강함
# - PAY_1 ~ PAY_6 : 대부분 0.6 이상으로 양의 상관관계가 강함

# ### 상관도가 높은 피처들의 PCA 변환 후 변동성 확인

# +
# BILL_AMT1 ~ BILL_AMT6 차원 축소
bill_colnames = ['BILL_AMT'+str(i) for i in range(1,7)]
scaler = StandardScaler()
billScaled = scaler.fit_transform(X[bill_colnames])

pca = PCA(n_components=2)
bill_scaled = pca.fit_transform(billScaled)
pca.explained_variance_ratio_
# -

# => 2개의 주성분만으로도 6개 피처의 변동성을 95% 이상 설명 가능
# - 첫번째 주성분이 90%의 변동성을 수용할 정도로 BILL_AMT 피처들의 상관도가 높음을 나타냄

# +
# PAY_1 ~ PAY_6 차원 축소
pay_colnames = ['PAY_'+str(i) for i in range(1,7)]
scaler = StandardScaler()
payScaled = scaler.fit_transform(X[pay_colnames])

pca = PCA(n_components=2)
pay_scaled = pca.fit_transform(payScaled)
pca.explained_variance_ratio_
# -

# ### 모든 피처에 대하여 주성분분석을 통해 반환

pca_full = PCA()
x_pca = pca_full.fit_transform(X)
var_ratio = pca_full.explained_variance_ratio_
var_ratio_s = pd.Series(np.round(var_ratio*100,3), index=['comp'+str(i) for i in range(23)],
                        name='pca_full_variance')
var_ratio_s

# ### 분류 예측 성능 비교

# - 원본 데이터 세트

model = RandomForestClassifier(random_state=205)
scores = cross_val_score(model, X, y, cv=3)
print(scores)
print(np.mean(scores))

# - PCA 변환 데이터 세트

for c in [2, 6, 10, 13]:
    pca = PCA(n_components=c)
    X_pca = pca.fit_transform(X)

    model = RandomForestClassifier(random_state=205)
    scores = cross_val_score(model, X_pca, y, cv=3)
    print(f'n_components={c}')
    print(scores)
    print(np.mean(scores))

# #### 2개의 주성분으로 압축한 신용카드 데이터 시각화

# +
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_pca_df = pd.DataFrame(X_pca, columns=['Comp1','Comp2'])
X_pca_df['target'] = y.values
X_pca_df.head()
# -

sns.scatterplot(data=X_pca_df, x='Comp1', y='Comp2', hue='target')
plt.show()

# #### 데이터를 표준화한 뒤 랜덤포레스트로 분류 성능 예측

# +
X_scaled = StandardScaler().fit_transform(X)

model = RandomForestClassifier(random_state=205)
scores = cross_val_score(model, X_scaled, y, cv=3)
print(scores)
print(np.mean(scores))
# -

# #### 데이터를 표준화한 뒤 PCA적용한 데이터로 분류 성능 예측

pca_full = PCA()
x_scaled_pca = pca_full.fit_transform(X_scaled)
var_ratio = pca_full.explained_variance_ratio_
var_ratio_df = pd.DataFrame(np.round(var_ratio*100,3), index=['comp'+str(i+1) for i in range(23)], columns=['variance'])
var_ratio_df

# +
# 90% 설명력을 갖는 comp 개수 찾기

var_ratio_df['cum_ratio'] = np.cumsum(var_ratio_df) ## 누적합
var_ratio_df

# +
pca = PCA(n_components=19)
X_pca = pca.fit_transform(X_scaled)

model = RandomForestClassifier(random_state=205)
scores = cross_val_score(model, X_pca, y, cv=3)
print(scores)
print(np.mean(scores))
# -

# #### 주성분별 분산 반영비 시각화

plt.plot(var_ratio_df.index, var_ratio_df.variance, 'bo-')
plt.xticks(rotation=70)
plt.xlabel('n_components')
plt.ylabel('Variance_ratio')
plt.show()

# => 변동비율이 많이 줄어든 지점의 성분 수를 사용

# ## 예제3. 이미지 압축


