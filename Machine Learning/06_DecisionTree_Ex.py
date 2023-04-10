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

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import train_test_split

# # iris 데이터에 결정 트리 적용 및 시각화

# +
# 모델 객체 생성
dt_clf = DecisionTreeClassifier(random_state=156)

# 데이터 로딩
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# 학습
dt_clf.fit(X_train, y_train)
# -

plot_tree(dt_clf)
plt.show()

# #### graphviz로 결정트리 시각화

export_graphviz(dt_clf, out_file='output/tree.dot', class_names=iris.target_names, feature_names=iris.feature_names, impurity=True, filled=True)

# #### .dot 파일 출력 방법 2가지
# - graphviz 시각화툴 사용
# - 이미지 파일로 변환해 저장 후 출력

# +
# grapviz의 Source() 사용
import graphviz

with open('output/tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
# -

# png 파일로 저장 후 출력
from subprocess import run
run(['dot', '-Tpng', 'output/tree.dot', '-o', 'output/decision_tree.png', '-Gdpi=600'])

from IPython.display import Image
Image('output/decision_tree.png')

# ### 결정트리 모델의 높이와 노드 수 반환

dt_clf.get_depth()

dt_clf.get_n_leaves()

dt_clf.get_params()

# ## feature_importances_ : 피처 중요도
# - tree 생성 시 각 피처가 얼마나 중요한지를 평가한 값
# - 피처별 중요도가 0~1 사이의 수치로 ndarray 형태로 값을 반환
# - 특정 노드의 중요도 값이 클수록 그 노드에서 불순도가 크게 감소됨을 의미

# #### 피처별 중요도 출력

for name, value in zip(iris.feature_names, dt_clf.feature_importances_):
    print(f'{name}:{value}')

# #### 피처별 중요도를 막대그래프로 시각화

sns.barplot(x=dt_clf.feature_importances_, y=iris.feature_names)
plt.show()


# 피처 중요도가 높은 순으로 그리기 함수 정의
def plot_importance(model, feature):
    importances = dt_clf.feature_importances_
    idx = list(reversed(np.argsort(importances)))
    feature_names = [feature[i] for i in idx]
    sns.barplot(x=importances[idx], y=feature_names)


# ### 하이퍼 파라미터 적용

# #### max_depth = 4

# +
dt_clf = DecisionTreeClassifier(max_depth=4, random_state=156)
dt_clf.fit(X_train, y_train)

export_graphviz(dt_clf, out_file='output/tree_depth3.dot', class_names=iris.target_names, feature_names=iris.feature_names, impurity=True, filled=True)

run(['dot', '-Tpng', 'output/tree_depth3.dot', '-o', 'output/descision_tree1.png', '-Gdpi=600'])
Image(filename='output/descision_tree1.png')
# -

plot_importance(dt_clf, iris.feature_names)

# #### min_samples_split=5

# +
dt_clf = DecisionTreeClassifier(min_samples_split=5, random_state=156)
dt_clf.fit(X_train, y_train)

export_graphviz(dt_clf, out_file='output/tree_split4.dot', class_names=iris.target_names, feature_names=iris.feature_names, impurity=True, filled=True)

run(['dot', '-Tpng', 'output/tree_split4.dot', '-o', 'output/descision_tree2.png', '-Gdpi=600'])
Image(filename='output/descision_tree2.png')
# -

plot_importance(dt_clf, iris.feature_names)

# #### min_samples_leaf=5

# +
dt_clf = DecisionTreeClassifier(min_samples_leaf=5, random_state=156)
dt_clf.fit(X_train, y_train)

export_graphviz(dt_clf, out_file='output/tree_leaf4.dot', class_names=iris.target_names, feature_names=iris.feature_names, impurity=True, filled=True)

run(['dot', '-Tpng', 'output/tree_leaf4.dot', '-o', 'output/descision_tree3.png', '-Gdpi=600'])
Image(filename='output/descision_tree3.png')
# -

plot_importance(dt_clf, iris.feature_names)
