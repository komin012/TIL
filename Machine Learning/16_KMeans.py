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
import matplotlib.pyplot as plt
import seaborn as sns

# # K-Means

# **알고리즘**
# 1. k개 중심점을 임의로 배치
#     - 무작위로 k개 샘플을 선택하여 중심점으로 결정
# 2. 모든 자료와 k개의 중심점과 거리를 계산하여 가장 가까운 중심점의 군집으로 할당
# 3. 군집의 중심(평균)을 구함
# 4. 정지 규칙에 이를 때까지 2~3단계를 반복
#     - 군집의 변화가 없을때
#     - 중심점의 이동이 임계값 이하일 때

# ## 사이킷런의 군집 알고리즘 클래스 KMeans

# ```python
# KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
#        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, 
#        n_jobs=1, algorithm='auto')
# ```
#
# - 기본적으로 최적화된 알고리즘을 적용
# - K-평균 초기화 파라미터 중 가장 중요한 파라미터
#     - n_clutsers : 클러스터 개수(클러스터 중심점의 개수)
#     - init : 초기에 클러스터 중심점 좌표를 설정할 방식
#         - 디폴트는 'k-means++'
#         - 임의로 중심점을 설정할 경우 'random'
#     - max_iter : 최대 반복횟수
#         - 이 횟수 이전에 모든 데이터의 중심점 이동이 없으면 종료

# ## 예제1. 시뮬레이션 데이터 군집화
# : 군집이 5개인 시뮬레이터 데이터를 생성하여 K-Means 군집화 결과 비교

# ### 군집화를 위한 데이터 생성
# - make_blobs()/make_classification()
#     - 여러 클래스에 해당하는 데이터셋 생성
#     - 하나의 클래스에 여러 개의 군집이 분포될 수 있게 생성
#     - 군집과 분류를 위한 테스트 데이터 생성을 위해 사용
#     -  두 api의 차이점
#         - make_blobs() : 개별 군집의 중심점과 표준편차 제어 기능이 추가되어 있음
#         - make_classification(): 노이즈를 포함한 데이터를 만드는데 유용
# - make_circle()/make_moon()
#     - 중심 기반의 군집화로 해결하기 어려운 데이터 셋 생성

# #### make_blobs()로 데이터 생성
# - 5개의 클러스터 중심을 갖는 2000개 데이터

from sklearn.datasets import make_blobs

# +
# 군집의 중심 좌표
blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8,2.8], [-2.8,1.3]])

# 군집 표준편차
blob_stds = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, n_features=2, centers=blob_centers, cluster_std=blob_stds, random_state=205)
X.shape
# -

np.unique(y)

# #### 산점도

plt.figure(figsize=(8,4))
plt.scatter(X[:,0], X[:,1], s=1, c=y)
plt.scatter(blob_centers[:,0], blob_centers[:,1], marker='s', c='b')
plt.xlabel('ftr1')
plt.ylabel('ftr2')
plt.show()

# ### K-Means로 군집화

from sklearn.cluster import KMeans

k=5
kmeans = KMeans(n_clusters=k, random_state=205)
y_pred = kmeans.fit_predict(X)

# +
data = pd.DataFrame(X, columns=['ftr1', 'ftr2'])
data['y'] = y
data['y_pred'] = y_pred

data[:10]  ## y에서 지정한 라벨 인덱스와 y_pred에서 지정한 라벨 인덱스의 순서가 다름
# -

data.groupby(['y','y_pred']).ftr1.count()

# #### labels_ 
# : 각 샘플에 할당된 레이블 (클러스터 인덱스)

kmeans.labels_

# #### cluster_centers_
# :KMeans의 centriods(중심점) 정보

kmeans.cluster_centers_

blob_centers

# #### n_iter_
# :cluster center 결정을 위한 반복 횟수

kmeans.n_iter_

# #### 군집화 결과 시각화

# +
kcenters = kmeans.cluster_centers_

fig, axs = plt.subplots(figsize=(8,4), ncols=2)
axs[0].scatter(X[:,0], X[:,1], s=1, c=y, label='origin')
axs[1].scatter(X[:,0], X[:,1], s=1, c=y_pred, label='kmeans')
axs[0].scatter(blob_centers[:,0], blob_centers[:,1],marker='s', c='r', label='origin_center')
axs[1].scatter(kcenters[:,0], kcenters[:,1], marker='^', c='b', label='kmeans_center')
plt.show()
# -

# ## 예제2. iris dataset 군집화

# +
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
# -

# #### K-Means 수행

kmeans = KMeans(n_clusters=3, random_state=205)  ## 품종 3개
kmeans.fit(X)
kmeans.labels_

iris_df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
iris_df['target'] = y
iris_df['cluster'] = kmeans.labels_
iris_df.head()

iris_df.groupby(['target', 'cluster']).sepal_length.count()

# - 0: setosa, 1: versicolor, 2: virginica
# - 분류 타깃이 0인 경우 모두 1번 클러스터로 군집화
# - 타깃이 1,2인 경우 각가 2개 14개는 다른 그룹으로 군집화

# ### PCA로 차원축소하고 군집화
# - iris 데이터에 4개의 피처들을 pca 적용하여 2차원으로 축소한 뒤 군집화 수행

# #### PCA로 차원축소

from sklearn.decomposition import PCA

# +
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(X)

iris_df['pca1'] = iris_pca[:,0]
iris_df['pca2'] = iris_pca[:,1]
iris_df.head()
# -

# #### 차원축소 데이터 시각화

sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='target')
plt.show()

# #### k-means한 결과를 차원축소한 데이터에 적용하여 시각화

sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='cluster')
plt.show()

# +
fig, axs = plt.subplots(figsize=(10,4), ncols=2)
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='target', ax=axs[0])
axs[0].set_title('target')
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='cluster', ax=axs[1])
axs[1].set_title('clustered')

plt.tight_layout()
plt.show()
# -

# #### 차원축소한 데이터를 K-Means한 결과 시각화

kmeans_pca = KMeans(n_clusters=3, random_state=205)
kmeans_pca.fit(iris_pca)
kmeans_pca.labels_

iris_df['pca_cluster'] = kmeans_pca.labels_
iris_df.groupby(['target', 'pca_cluster']).sepal_length.count()

# +
fig, axs = plt.subplots(figsize=(14,4), ncols=3)
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='target', ax=axs[0])
axs[0].set_title('target')
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='cluster', ax=axs[1])
axs[1].set_title('clustered')
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='pca_cluster', ax=axs[2])
axs[2].set_title('pca -> clustered')

plt.tight_layout()
plt.show()
