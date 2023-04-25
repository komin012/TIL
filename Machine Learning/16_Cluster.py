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

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# # K-평균 최적화

# ## 1. 중심점(centroid) 초기화

# **방법1. 다른 군집 알고리즘을 먼저 실행하여 근사한 센트로이드 위치를 결정**
# - KMeans클래스의 ```init``` 매개변수에 센트로이드 리스트를 담은 np.array를 지정하고 ```n_init```을 1로 설정
#
# **방법2. 랜덤 초기화를 다르게 하여 여러 번 알고리즘을 실행하고 가장 좋은 솔루션 선택**
# - 랜덤 초기화 횟수 지정
#     - KMeans 클래스의 매개변수 ```n_init``` 조절
#     - ```n_init``` 기본값은 10
#     
# **방법3. K-Means++ 초기화 알고리즘 사용**
# - KMeans 클래스 ```init```매개 변수의 기본값으로 ```K-means++```를 사용하고 있음
# - 이 방식을 사용하고 싶지 않은 경우 ```init``` 매개변수를 ```'random'```으로 지정

# ### 중심점 초기화(init) 지정

# #### 가상데이터 생성

# +
# 군집의 중심 좌표
blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8,2.8], [-2.8,1.3]])

# 군집 표준편차
blob_stds = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, n_features=2, centers=blob_centers, cluster_std=blob_stds, random_state=205)
X.shape


# -

# centers: [('label1', center_pos1), ('label2', center_pos2)]
def draw_cluster(X,y, centers=None, title=None):
    markers = ['sb', '^r', 'og']
    plt.figure(figsize=(8,4))
    plt.scatter(X[:,0], X[:,1], s=1, c=y)
    if centers:
        for i, center_label in enumerate(centers):
            label, center = center_label
            plt.scatter(center[:,0], center[:,1], marker=markers[i][0], c=markers[i][1], label=label)
        plt.legend()
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2', rotation=0)
    plt.show()


draw_cluster(X,y,[('origin cluster', blob_centers)], title='Sample Clusters')

# #### 초기화 방법1. 다른 군집 알고리즘으로 근사한 중심점을 초기 센트로이드로 사용

# +
good_init = np.array([[-3,3], [-3,2], [-3,1], [-1,2], [0,2]])

k=5
kmeans_goodinit = KMeans(n_clusters=k, init=good_init, n_init=1)
kmeans_goodinit.fit(X)
kg_centers = kmeans_goodinit.cluster_centers_
print(f'## 다른 군집알고리즘으로 얻어진 중심점으로 초기화한 경우 ##\n클러스터 중심점 좌표들:\n{np.round(kg_centers,3)}')
draw_cluster(X, y, centers=[('kmeans_result', kg_centers), ('good_init', good_init), ('origin', blob_centers)])
# -

# #### 초기화 방법2. 랜덤 초기화를 다르게 하여 여러번 알고리즘을 실행하고 가장 좋은 솔루션 선택

# - max_iter 변화시키면서 군집 수행

# +
# pip install -U threadpoolctl
# -

max_iters = [1,2,3,4,5]
for iter_n in max_iters:
    kmeans = KMeans(n_clusters=k, init='random', n_init=1, algorithm='full', max_iter=iter_n, random_state=205)
    ### algorithm = 'full' : 속도개선 알고리즘 적용
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    print(f'max_iter={iter_n}일 때 클러스터 중심좌표\n{np.round(centers,3)}')
    title = f'init="random", max_iter={iter_n}, algorithm="full"'
    draw_cluster(X, y, centers=[(f'max_iter={iter_n}', centers), ('origin', blob_centers)],
                 title=title)   


# #### 초기화방법3. K-Means++ 초기화 알고리즘 사용

kmeans = KMeans(n_clusters=k, init='k-means++', random_state=205)
kmeans.fit(X)
kp_centers = kmeans.cluster_centers_
draw_cluster(X, y, centers=[('k-means++',kp_centers),('origin',blob_centers)], title='init="k=means++"')

# ## 2. K-평균 최적 솔루션 성능 지표
#
# **이너셔가 가장 작은 모델이 최적**
# - 이너셔(inertia): 각 샘플과 가까운 센트로이드 사이의 평균 제곱 거리
# - KMeans 클래스의 ```inertia_```인스턴스 변수에 이너셔 값을 저장

# #### 이너셔 비교

kmeans_goodinit.inertia_

# +
good_init = np.array([[-3,3], [-3,2], [-3,1], [-1,2], [0,2]])

k=5
inertia={}
for method in ['init_value', 'random', 'k-means++']:
    if method == 'init_value':
        kmeans = KMeans(n_clusters=k, init=good_init, n_init=1)
        kmeans.fit(X)
        inertia[method] = kmeans.inertia_
    elif method == 'random':
        for iter_n in range(1,7):
            kmeans = KMeans(n_clusters=k, init='random', n_init=1, algorithm='full', max_iter=iter_n, random_state=205)
            kmeans.fit(X)
            inertia[f'random: max_iter={iter_n}'] = kmeans.inertia_
    elif method=='k-means++':
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=205)
        kmeans.fit(X)
        inertia[method] = kmeans.inertia_
    
inertia
# -

# #### score() 메서드
# - 이너셔의 음수값을 반환
#     - 사이킷런은 큰 값이 좋은 값이라는 규칙에 따라 이너셔의 음수값을 반환

kmeans.score(X)

# ### 참고. 미니배치 K-평균
# - 전체 데이터셋을 사용해 반복하지 않고 각 반복마다 미니배치를 사용하여 센트로이드를 조금씩 이동
# - 알고리즘 속도를 높임
# - 메모리에 들어가지 않는 대량의 데이터셋에 군집 알고리즘을 적용할 수 있음
#
# - 일반 K-평균 알고리즘 보다 훨씬 빠르지만 이너셔는 일반적으로 조금 더 나쁨

# +
from sklearn.cluster import MiniBatchKMeans

mini_kmeans = MiniBatchKMeans()
mini_kmeans.fit(X)
mini_kmeans.inertia_
# -

# ### 3. 최적의 클러스터 개수 찾기

# #### 방법1. 이너셔가 급격히 작아지는 지점(elbow)의 K 선택

kmeans_per_k = [KMeans(n_clusters=k, random_state=205).fit(X) for k in range(1,11)]
inertias = [model.inertia_ for model in kmeans_per_k]
inertias_s = pd.Series(inertias, index=['k='+str(k) for k in range(1,11)])
inertias_s

# #### 그래프 그리기

plt.plot(inertias_s.index, inertias_s, 'bo-')
plt.ylabel('inertia')
plt.xlabel('k')
plt.show()

plt.plot(inertias_s.index, inertias_s, 'bo-')
plt.ylabel('inertia')
plt.xlabel('k')
plt.annotate('Elbow', xy=(3,inertias_s[3]), xytext=(0.45, 0.35), textcoords='figure fraction',
            fontsize=12, arrowprops=dict(facecolor='black', shrink=0.1))
## xytext=(가로 퍼센트, 세로 퍼센트) : 텍스트 위치
plt.show()

# - 클러스터가 늘어날수록 각 샘플은 가까운 센트로이드에 더 가깝게 되어 이너셔는 더 작아짐
#
# - 엘보(elbow) : 클러스터 개수 k가 커질 때 이너셔가 꺽이는 지점
#     
#     => k=4 (k=2라고 생각할 수 있지만 k=1->k=3의 각도보다 k=3->k=5 각도가 더커보이기 때문쓰)

# #### 방법2. 실루엣 점수(Silhouette score)
# : 모든 샘플에 대한 실루엣 계수 평균
#
# 실루엣계수 = $\frac{b-a}{max(a,b)}$
# - a: 동일한 클러스터에 있는 다른 샘플까지 평균 거리
#     - 클러스터 내부의 평균 거리
# - b: 가장 가까운 클러스터의 샘플까지 평균 거리
#     - 가장 가까운 클러스터는 자신이 속한 클러스터는 제외하고 b가 최소인 클러스터
# - -1 ~ +1
#     - +1에 가까우면 자신의 클러스터 안에 잘 속해 있고 다른 클러스터와는 멀리 떨어져 있음을 나타냄
#     - 0에 가까우면 클러스터 경계에 위치함을 나타냄
#     - -1에 가까우면 이 샘플이 잘못된 클러스터에 할당되었음을 의미

# - silhouette_score() 함수

from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=k, random_state=205).fit(X)
silhouette_score(X, kmeans.labels_)

# - 클러스터 개수에 따른 실루엣 점수 비교

kmeans_per_k

sil_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]] ## n_clusters >= 2인 것부터
## 실루엣점수는 다른 클러스터와의 거리도 알아야 하므로 클러스터 개수가 최소 2개이상이어야 함
sil_scores_s = pd.Series(sil_scores, index=['k='+str(i) for i in range(2,11)])
sil_scores_s

plt.plot(sil_scores_s.index, sil_scores_s, 'bo-')
plt.ylabel('Silhouette score')
plt.xlabel('$k$')
plt.title('Silhouette Scores with k')
plt.annotate('maximum Silhouette score', xy=(2,sil_scores_s[2]), xytext=(0.42, 0.8), textcoords='figure fraction', 
             fontsize=14, arrowprops=dict(facecolor='black', shrink=0.1))
plt.show()

# - 실루엣 스코어가 가장 높은 k찾기

sil_scores_s.index[np.argmax(sil_scores)]


# - 실루엣 다이어그램
#     - 모든 샘플의 실루엣 계수를 할당된 클러스터와 계수값으로 정렬하여 그린 그래프
#     - 클러스터마다 칼 모양의 그래프가 그려짐
#     - 그래프의 높이는 클러스터가 포함하고 있는 샘플의 개수
#     - 그래프의 너비는 클러스터에 포함된 샘플의 정렬된 실루엣 계수를 나타냄
#         - 너비가 넓을수록 좋음

# (참고. 실루엣 다이어그램 그리는 함수)<br>
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

# +
### 여러개의 클러스터링 갯수를 List로 입력 받아 
### 각각의 실루엣 계수를 면적으로 시각화한 함수 작성

def visualize_silhouette(cluster_lists, X_features): 
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 
    # 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서
    # 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 
        # 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")


# -

visualize_silhouette([2,3,4,5,6,7], X)

# cluster=6 : 클러스터 2가 평균실루엣 스코어보다 낮으므로 부적정
# cluster=7 : 위와 같은 이유로 부적정<br>
# => 실루엣 계수가 높은 cluster 4 or 5가 적정

# +
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1)

visualize_silhouette([2,3,4,5,6], X)

# +
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data

visualize_silhouette([2,3,4,5,6], X)

# +
# iris 데이터에 pca를 적용하고 kmeans를 적용
from sklearn.decomposition import PCA

iris_pca = PCA(n_components=2).fit_transform(iris.data)

visualize_silhouette([2,3,4,5,6], iris_pca)
# -

# ---

# # 계층적 군집분석(Hierachical Clustering)

from sklearn.cluster import AgglomerativeClustering

# ```python
# AgglomerativeClustering(n_clusters=2, *, affinity='deprecated', metric=None, memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False)
# ```

agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward', compute_distances=True)
agg = agg.fit(iris.data)
agg.labels_

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['hier_cluster'] = agg.labels_
iris_df.head()

fig, axs = plt.subplots(figsize=(10,5), ncols=2)
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='target', ax=axs[0])
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='hier_cluster', ax=axs[1])
plt.show()

fig, axs = plt.subplots(figsize=(10,5), ncols=2)
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='target', ax=axs[0])
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='hier_cluster', ax=axs[1])
plt.show()

# +
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# -

plot_dendrogram(agg,truncate_mode="level", p=3)

# # 평균 이동(Mean Shift) 방법

# ```python
# MeanShift(*, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300)
# ```
#
# - bandwidth를 작게하면 지나치게 세분화된 군집을 찾음
# - bandwidth가 크면 적정한 군집을 찾지 못할 수 있음

from sklearn.cluster import MeanShift

for band in range(1,6):
    meanshift = MeanShift(bandwidth=band)
    labels = meanshift.fit_predict(iris.data)
    print(f'bandwidth={band}일 때\ncluster labels: {np.unique(labels)}')

# +
from sklearn.cluster import estimate_bandwidth

best_band = estimate_bandwidth(iris.data)
print(f'최적의 bandwidth = {best_band:.3f}')
# -

meanshift = MeanShift(bandwidth=best_band)
labels = meanshift.fit_predict(iris.data)
iris_df['mean_shift'] = labels
print(f'bandwidth={best_band:.3f}일 때\ncluster labels:{np.unique(labels)}')

# +
iris_df['mean_shift'] = labels
fig, axs = plt.subplots(figsize=(10,5), ncols=2)
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='target', ax=axs[0])
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='mean_shift', ax=axs[1])

plt.show()
# -

# # Gaussian Mixture Model(GMM)

# ```python
# GaussianMixture(n_components=1, *, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
# ```

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=205)
labels = gmm.fit_predict(iris.data)
iris_df['gmm_cluster'] = labels

# +
fig, axs = plt.subplots(figsize=(10,5), ncols=2)
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='target', ax=axs[0])
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='gmm_cluster', ax=axs[1])

plt.show()
# -

# # DBSCAN(Density Based Spatial Clustering of Application with Noise)
#
# ```python
# DBSCAN(eps=0.5, *, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
# ```

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.6, min_samples=8, metric='euclidean')
labels = dbscan.fit_predict(iris.data)
iris_df['dbscan_cluster'] = labels

# +
fig, axs = plt.subplots(figsize=(10,5), ncols=2)
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='target', ax=axs[0])
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='dbscan_cluster', ax=axs[1])

plt.show()
# -

# - pca한 데이터로 클러스터 적용

dbscan = DBSCAN(eps=0.6, min_samples=8, metric='euclidean')
labels = dbscan.fit_predict(iris_pca)
iris_df['pca_dbscan_cluster'] = labels

# +
fig, axs = plt.subplots(figsize=(10,5), ncols=2)
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='target', ax=axs[0])
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='pca_dbscan_cluster', ax=axs[1])

plt.show()
