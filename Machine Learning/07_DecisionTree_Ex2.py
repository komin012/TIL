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
import warnings
warnings.filterwarnings('ignore')

# # 타이타닉 데이터에 대한 결정트리를 생성하고 예측
# - 캐글에서 제공하는 타이타닉 데이터

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score

# ### 데이터 로딩 및 전처리

titanic = pd.read_csv('data/titanic/train.csv')
titanic.head()


# 타이타닉 데이터 전처리 함수
def transform_features(df, encoding=1):
   # 변수 삭제
    labels = ['PassengerId','Name','Ticket']
    df.drop(labels=labels, axis=1, inplace=True)
    
    # 결측치 처리
    df['Age'].fillna(df.Age.mean(), inplace=True)  
    df['Cabin'].fillna('N', inplace=True)   
    df['Embarked'].fillna('N', inplace=True)
    
    # 인코딩
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Sex','Cabin','Embarked'] 
    if encoding == 1:
        for feature in features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
    elif encoding == 2 :
        dummy = pd.get_dummies(df[features])
        df.drop(features, axis=1, inplace=True)
        df = pd.concat([df, dummy], axis=1)
    else:
        print('encode code error')
    
    return df


# #### 라벨인코딩

titanic_le = transform_features(titanic)
titanic_le.head()

# +
X_le = titanic_le.drop('Survived', axis=1)
y_le = titanic_le['Survived']

X_train_le, X_test_le, y_train_le, y_test_le = train_test_split(X_le, y_le, test_size=0.2, random_state=205)
# -

# #### 원핫인코딩

titanic = pd.read_csv('data/titanic/train.csv')
titanic_ohe = transform_features(titanic)
titanic_ohe.head()

# +
X_ohe = titanic_ohe.drop('Survived', axis=1)
y_ohe = titanic_ohe['Survived']

X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_ohe, y_ohe, test_size=0.2, random_state=205)
# -

# ### 결정트리 생성

# #### 라벨인코딩

dt_le = DecisionTreeClassifier(random_state=205)
dt_le.fit(X_train_le, y_train_le)
print(f'depth:{dt_le.get_depth()}, leaf:{dt_le.get_n_leaves()}')

# +
export_graphviz(dt_le, out_file='output/tree_le.dot', class_names=['N','Y'], feature_names=X_train_le.columns, impurity=True, filled=True)

from subprocess import run
from IPython.display import Image
run(['dot', '-Tpng', 'output/tree_le.dot','-o', 'output/tree_le.png', '-Gdpi=600'])
Image('output/tree_le.png')
# -

# #### 원핫인코딩

dt_ohe = DecisionTreeClassifier(random_state=205)
dt_ohe.fit(X_train_ohe, y_train_ohe)
print(f'depth:{dt_ohe.get_depth()}, leaf:{dt_ohe.get_n_leaves()}')

# +
export_graphviz(dt_ohe, out_file='output/tree_ohe.dot', class_names=['N','Y'], feature_names=X_train_ohe.columns, impurity=True, filled=True)

run(['dot', '-Tpng', 'output/tree_ohe.dot','-o', 'output/tree_ohe.png', '-Gdpi=600'])
Image('output/tree_ohe.png')


# -

# #### 피처 중요도 시각화

def plot_importance(model, feature):
    importances = pd.Series(model.feature_importances_, index=feature).sort_values(ascending=False)
    sns.barplot(x=importances, y=importances.index)
    plt.show()


# #### 라벨인코딩

plot_importance(dt_le, X_train_le.columns)

# #### 원핫인코딩

plot_importance(dt_ohe, X_train_ohe.columns)


# ### Decision Boundary 시각화

def visualize_boundary(model, X, y, i, j):
    fig,ax = plt.subplots()
    
    # 학습 데이타 scatter plot으로 나타내기
    ax.scatter(X[i], X[j], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start , xlim_end = ax.get_xlim()
    ylim_start , ylim_end = ax.get_ylim()
    
    # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
    model.fit(X[[i, j]], y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행. 
    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),
                         np.linspace(ylim_start,ylim_end, num=200))
    
    # ravel() : 다차원을 1차원으로 품
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # contourf() 를 이용하여 class boundary 를 visualization 수행. 
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', clim=(y.min(), y.max()), zorder=1)


# #### 라벨인코딩

visualize_boundary(dt_le, X_train_le, y_train_le, 'Sex', 'Age')

# #### 원핫인코딩

visualize_boundary(dt_ohe, X_train_le, y_train_le, 'Sex', 'Age')

# ### max_depth를 2에서 20까지 변화시켜가면서 최적의 max_depth 찾기
# - 성능지표는 roc_auc 사용

# #### 라벨인코딩

scores = cross_validate(dt_le, X_train_le, y_train_le, scoring='roc_auc')
scores

# +
scores_dic = {}
for i in range(2, 21):
    dt_le = DecisionTreeClassifier(max_depth=i, random_state=205)
    scores = cross_validate(dt_le, X_train_le, y_train_le, scoring='roc_auc')
    scores = np.mean(scores['test_score'])
    scores_dic[str(i)] = scores
    
print('LabelEncoder')
print(f'최적의 max_depth={max(scores_dic)}. 이때 roc_auc={scores_dic[max(scores_dic)]:.4f}')
# -

# #### 원핫인코딩

# +
scores_dic = {}
for i in range(2, 21):
    dt_ohe = DecisionTreeClassifier(max_depth=i, random_state=205)
    scores = cross_validate(dt_ohe, X_train_ohe, y_train_ohe, scoring='roc_auc')
    scores = np.mean(scores['test_score'])
    scores_dic[str(i)] = scores
    
print('OneHotEncoder')
print(f'최적의 max_depth={max(scores_dic)}. 이때 roc_auc={scores_dic[max(scores_dic)]:.4f}')
# -

# ### GridSearchCV로 최적의 하이퍼파라미터 찾기'
# - min_samples_splits = {2, 4, 8, 16, 24}
# - max_depth = {2, 4, 6, 8, 10}
# - min_samples_leaf = {1, 3, 5, 7, 9, 15}
# - 성능지표는 roc_auc_score로 지정

# #### 라벨인코딩

# +
params = {'max_depth':[2,4,6,8,10],
          'min_samples_split':[2,4,8,16,24],
          'min_samples_leaf':[1,3,5,7,9,15]}

grid_cv_le = GridSearchCV(dt_le, param_grid=params, scoring='roc_auc')
grid_cv_le.fit(X_train_le, y_train_le)
# -

print(f'최적의 파라미터: {grid_cv_le.best_params_}')
print(f'최고 정확도: {grid_cv_le.best_score_:.4f}')

# #### 원핫인코딩

grid_cv_ohe = GridSearchCV(dt_ohe, param_grid=params, scoring='roc_auc')
grid_cv_ohe.fit(X_train_ohe, y_train_ohe)
print(f'최적의 파라미터: {grid_cv_ohe.best_params_}')
print(f'최고 정확도: {grid_cv_ohe.best_score_:.4f}')

# ### GridSearchCV로 찾은 최적의 estimator로 재학습

# #### 라벨인코딩

# +
best_le = grid_cv_le.best_estimator_
pred_le = best_le.predict(X_test_le)
pred_proba_le = best_le.predict_proba(X_test_le)[:,1]

roc_auc_score(y_test_le, pred_proba_le)
# -

# #### 원핫인코딩

# +
best_ohe = grid_cv_ohe.best_estimator_
best_ohe.fit(X_train_ohe, y_train_ohe)
pred_ohe = best_ohe.predict(X_test)
pred_proba_ohe = best_ohe.predict_proba(X_test)[:,1]

roc_auc_score(y_test, pred_proba_ohe)
# -


