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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 데이터 준비
iris = load_iris()
iris.target_names

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# +
# 모델 객체 생성
dt_clf = DecisionTreeClassifier()

# 데이터 훈련
dt_clf.fit(X_train, y_train)

# 데이터 예측
pred = dt_clf.predict(X_test)

# 평가
acc = accuracy_score(y_test, pred)

print(acc)

# +
##### 교차검증
X = iris.data
y = iris.target

# KFold
from sklearn.model_selection import KFold

# +
kfold = KFold()

# 객체 생성
dt_clf = DecisionTreeClassifier()

# 훈련 / 예측 / 평가
accuracy_list = []
for n, (train_idx, test_idx) in enumerate(kfold.split(X)):
    Xtrain, Xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    dt_clf.fit(Xtrain, ytrain)
    pred = dt_clf.predict(Xtest)
    acc = accuracy_score(ytest, pred)
    accuracy_list.append(acc)

print(np.mean(accuracy_list))

# +
## k=3

kfold = KFold(n_splits=3)

accuracy_list = []
for n,(train_idx, test_idx) in enumerate(kfold.split(X)):
    Xtrain, Xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    dt_clf.fit(Xtrain, ytrain)
    pred = dt_clf.predict(Xtest)
    acc = accuracy_score(ytest, pred)
    accuracy_list.append(acc)
    
print(accuracy_list)
print(np.mean(accuracy_list))

# +
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

for n,(train_idx, test_idx) in enumerate(kfold.split(X)):
    ytrain, ytest = iris_df.target.iloc[train_idx], iris_df.target.iloc[test_idx]
    print(ytrain.value_counts())
    print(ytest.value_counts())
    
# shuffle을 주지 않아 군집이 모여서 나뉨

# +
kfold = KFold(n_splits=3, shuffle=True)

accuracy_list = []
for n,(train_idx, test_idx) in enumerate(kfold.split(X)):
    Xtrain, Xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    dt_clf.fit(Xtrain, ytrain)
    pred = dt_clf.predict(Xtest)
    acc = accuracy_score(ytest, pred)
    accuracy_list.append(acc)
    
print(accuracy_list)
print(np.mean(accuracy_list))

# +
#### stratified KFold

from sklearn.model_selection import StratifiedKFold

# +
dt_clf = DecisionTreeClassifier()

skf = StratifiedKFold(n_splits=3)

accuracy_list=[]
for n, (train_idx, test_idx) in enumerate(skf.split(X,y)):
    Xtrain, Xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    dt_clf.fit(Xtrain, ytrain)
    pred = dt_clf.predict(Xtest)
    acc = accuracy_score(ytest, pred)
    accuracy_list.append(acc)
    
print(accuracy_list)
print(np.mean(accuracy_list))

# +
dt_clf = DecisionTreeClassifier()

skf = StratifiedKFold(n_splits=3)

accuracy_list=[]
for n, (train_idx, test_idx) in enumerate(skf.split(X,y)):
    Xtrain, Xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    dt_clf.fit(Xtrain, ytrain)
    pred = dt_clf.predict(Xtest)
    acc = accuracy_score(ytest, pred)
    accuracy_list.append(acc)
    
print(accuracy_list)
print(np.mean(accuracy_list))

# +
#### cross_val_score
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_clf, X, y, cv=3, scoring='accuracy')
print(scores)
print(np.mean(scores))

# +
#### GridSearchCV
from sklearn.model_selection import GridSearchCV

params = {'max_depth':[1,2,3],
         'min_samples_leaf':[2,3]}
grid = GridSearchCV(dt_clf, param_grid=params, cv=3, refit=True, return_train_score=True)
grid.fit(X_train, y_train)
result = pd.DataFrame(grid.cv_results_)
result
# -

grid.best_score_

grid.best_params_

best_dt = grid.best_estimator_
pred = best_dt.predict(X_test)
accuracy_score(y_test,pred)


