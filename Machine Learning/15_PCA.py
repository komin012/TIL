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
