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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# # 군집화 실습: 고객 세그멘테이션
#
# **군집화 기준**
# - RFM 기법
#     - Recency(R): 가장 최근 상품 구입 일에서 오늘까지의 기간
#     - Frequency(F): 상품 구매 횟수
#     - Monetary Value(M): 총 구매 금액
#     
# ## 예제 데이터 : Online Retail Data Set
# https://archive.ics.uci.edu/ml/datasets/online+retail
#
# **변수**
# - InvoiceNo    : 주문번호 'C'로 시작하는 것은 취소 주문 
# - StockCode    : 제품 코드(Item Code)
# - Description  : 제품 설명        
# - Quantity     : 주문 제품 건수         
# - InvoiceDate  : 주문 일자
# - UnitPrice    : 제품 단가       
# - CustomerID   : 고객 번호       
# - Country      : 주문고객의국적

retail = pd.read_excel('data/OnlineRetail.xlsx')
retail.head(3)

retail.info()

retail.shape

retail.isnull().sum()

# => 고객 아이디는 상당히 중요한 변수이기 때문에 결측값이 많다고 아예 삭제할 수 없음. 그렇기 때문에 고객 아이디가 없는 행을 삭제해 결측치 처리

retail[['Quantity', 'UnitPrice']].describe()

# => 제품 건수와 제품 단가에서 음수가 나옴.

# #### 결측치와 이상치 처리

# +
retail = retail[retail.Quantity>0]
retail = retail[retail.Quantity>0]
retail = retail[retail.CustomerID.notnull()]

retail.shape
# -

retail.isnull().sum()

retail[['Quantity', 'UnitPrice']].describe()

# #### Country 변수 빈도

retail.Country.value_counts()

retail.Country.unique().shape

# - 최종 분석데이터로 가장 주문 빈도가 높은 United Kingdom의 자료만 선택

retail_UK = retail[retail.Country == 'United Kingdom']
retail_UK.shape

# ## RFM 기반 데이터 가공

# #### 주문금액 변수 sale_amount 추가
# : $Quantity × UnitPrice$

retail_UK['sale_amount'] = retail_UK.Quantity*retail_UK.UnitPrice

# #### CustomerID 자료형을 정수형으로 변환

retail_UK['CustomerID'] = retail_UK['CustomerID'].astype(int)
retail_UK.dtypes

# #### Top5 주문건수와 주문금액을 가진 고객 출력

# 주문건수 top5
retail_UK.CustomerID.value_counts().head(5)

# 주문금액 top5
retail_UK.groupby('CustomerID').sale_amount.sum().sort_values(ascending=False)[:5]

# #### 주문번호+상품코드 기준으로 정렬

retail_UK.groupby(['InvoiceNo','StockCode']).InvoiceNo.count()

# ### 고객 기준의 R,F,M 변수를 갖는 데이터 생성
# - 주문번호+상품코드 기준의 데이터를 개별 고객 기준의 데이터로 Groupby 수행
#
# #### R, F, M 변수 생성
# - Recency: 가장 최근 상품 구입일에서 오늘까지의 기간
#     - 'CustomerID'별로 글룹화하여 'Invoice Date' 중 가장 최근 주문 일자 사용
# - Frequency: 고객별 주문건수
#     - 'CustomerID'별로 그룹화하여 'InvoiceNo' 개수를 계산
# - Monetary: 총 구매금액
#     - 'CustomerID'별로 그룹화하여 'sale_amount'합계를 사용

retail_UK.columns

retail_UK.InvoiceDate.max(), retail_UK.InvoiceDate.min()

# +
agg_params = {'InvoiceDate':'max',  ## 가장 최근 주문
              'InvoiceNo':'count',  ## 주문 건수
              'sale_amount':'sum'}  ## 주문 금액 합계

retail_UK_rfm = retail_UK.groupby('CustomerID').agg(agg_params)
retail_UK_rfm
# -

# #### 컬럼 이름 변경

retail_UK_rfm.rename(columns={'InvoiceDate':'Recency', 'InvoiceNo':'Frequency','sale_amount':'Monetary'}, inplace=True)
retail_UK_rfm.head()

retail_UK_rfm.reset_index(inplace=True)
retail_UK_rfm

# #### Recency 변수 가공
# - 오늘 날짜를 기준으로 최근 주문 일자를 뺌
# - 오늘 날짜: 2011.12.10
#     - 온라인 판매 데이터: 2010.12.01~2011.12.09
# - 2022.12.10에서 최근 주문일자를 빼고 일자 데이터만 추출

# +
import datetime as dt

retail_UK_rfm.Recency = dt.datetime(2011,12,10) - retail_UK_rfm.Recency
retail_UK_rfm.Recency = retail_UK_rfm.Recency.apply(lambda x: x.days+1)
retail_UK_rfm.head()
# -

# ## RFM 기반 고객 세그먼테이션

# #### R, F, M 변수 히스토그램

# +
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.hist(retail_UK_rfm.Recency)
plt.title('Recency hist.')

plt.subplot(1,3,2)
plt.hist(retail_UK_rfm.Frequency)
plt.title('Frequency hist.')

plt.subplot(1,3,3)
plt.hist(retail_UK_rfm.Monetary)
plt.title('Monetary hist.')

plt.tight_layout()
plt.show()
# -

# #### R,F,M 기술통계량

retail_UK_rfm.iloc[:, 1:].describe()

# ### R,F,M 변수 표준화

# +
from sklearn.preprocessing import StandardScaler

ftr_names = retail_UK_rfm.columns.difference(['CustomerID'])
X = retail_UK_rfm[ftr_names]
X_scaled = StandardScaler().fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=ftr_names)
X_scaled_df.describe()
# -

# ## K-평균 군집분석

# #### 표준화된 R,F,M 변수로 구성된 데이터를 기반으로 K-평균 군집분석 수행

# +
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

kmeans = KMeans(n_clusters=3, random_state=205)
labels = kmeans.fit_predict(X_scaled)
# -

retail_UK_rfm['cluster'] = labels

silhouette_score(X_scaled, labels)


# #### 적정한 군집 개수 결정

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

### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 클러스터링 결과를 시각화 
def visualize_kmeans_plot_multi(cluster_lists, X_features):
    
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성 
    n_cols = len(cluster_lists)
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 입력 데이터의 FEATURE가 여러개일 경우 
    # 2차원 데이터 시각화가 어려우므로 PCA 변환하여 2차원 시각화
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_features)
    dataframe = pd.DataFrame(pca_transformed, columns=['PCA1','PCA2'])
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서
    # KMeans 클러스터링 수행하고 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링으로 클러스터링 결과를 dataframe에 저장. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(pca_transformed)
        dataframe['cluster']=cluster_labels
        
        unique_labels = np.unique(clusterer.labels_)
        markers=['o', 's', '^', 'x', '*']
       
        # 클러스터링 결과값 별로 scatter plot 으로 시각화
        for label in unique_labels:
            label_df = dataframe[dataframe['cluster']==label]
            if label == -1:
                cluster_legend = 'Noise'
            else :
                cluster_legend = 'Cluster '+str(label)           
            axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70,\
                        edgecolor='k', marker=markers[label], label=cluster_legend)

        axs[ind].set_title('Number of Cluster : '+ str(n_cluster))    
        axs[ind].legend(loc='upper right')
    
    plt.show()


visualize_silhouette([2,3,4,5], X_scaled)
visualize_kmeans_plot_multi([2,3,4,5], X_scaled)

# ## R,M,F 데이터 로그 변환 후 군집 분석

# #### 로그 변환

log_X = np.log1p(X)
log_X.describe()

# +
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.title(log_X.columns[i-1])
    plt.hist(log_X.iloc[:,1])
    
plt.tight_layout()
plt.show()
# -

# #### 로그변환된 R,M,F를 표준화

log_scaled_X = StandardScaler().fit_transform(log_X)
log_scaled_X_df = pd.DataFrame(log_scaled_X, columns=ftr_names)
log_scaled_X_df.describe()

# #### 클러스터 3으로 K-Means 수행

kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(log_scaled_X)

silhouette_score(log_scaled_X, labels)

retail_UK_rfm['log_cluster'] = labels

# ### 로그변환된 데이터세트 기반의 군집결과 시각화
# : 군집수에 따른 실루엣 스코어, 군집 결과

visualize_silhouette([2,3,4,5], log_scaled_X)
visualize_kmeans_plot_multi([2,3,4,5], log_scaled_X)

# => 로그 변환한 데이터세가 군집이 더 잘 나타남
