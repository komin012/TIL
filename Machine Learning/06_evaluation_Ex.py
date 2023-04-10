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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# # 타이타닉 데이터 전처리
# - 캐글에서 제공하는 타이타닉 데이터

titanic = pd.read_csv('data/titanic/train.csv')


# 타이타닉 데이터 전처리 함수 정의
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


titanic = transform_features(titanic)

titanic.head()

# # 모델 생성

# +
# 피처와 타깃
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=203)

# 로지스틱 회귀 모델 생성
lr = LogisticRegression(solver='liblinear', random_state=203)
### default : solver ='lbfgs'
### 데이터가 작을 땐 solover='liblinear'

# 학습
lr.fit(X_train, y_train)

# 예측
pred = lr.predict(X_test)
# -

# # 예측 평가

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score,\
                            roc_auc_score, precision_recall_curve, roc_curve

accuracy_score(y_test, pred)

# +
confusion_matrix(y_test, pred)

# TN FP
# FN TP
# -

# 정밀도
print(precision_score(y_test, pred))
print(46/(19+46))

# 재현율(민감도)
print(recall_score(y_test, pred))
print(46/(19+46))

# f1 
f1_score(y_test,pred)

# g measure
g = np.sqrt(precision_score(y_test, pred)*recall_score(y_test, pred))
g


# ### 성능평가지수 출력 함수 정의

def get_eval_score(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test,pred)
    g = np.sqrt(precision*recall)
    print(f'오차행렬\n{confusion_matrix(y_test,pred)}')
    print(f'정확도:{accuracy:.4f}, 정밀도:{precision:.4f}, 재현율:{recall:.4f}, F1:{f1:.4f}, G-measure:{g:.4f}')


get_eval_score(y_test, pred)

# ## predict_proba() 메소드
# - negative / positive로 예측할 확률 (두 열의 합은 1) <br>
#     => 0.5보다 큰 걸로 예측

predict_proba = lr.predict_proba(X_test)
predict_proba

np.concatenate([predict_proba, pred.reshape(-1,1)], axis=1)

# ### 분류 결정 임계값(threshold) 조절을 통한 평가지표 결과
# :Binarizer 클래스 활용

from sklearn.preprocessing import Binarizer

# Binarizer : 지정한 임계값(threshold)을 기준으로 나눠줌
# Binarizer 클래스 예시
x = [[0.5, -1, 2],
    [2, 0, 0],
    [0, 1, 1.2]]
bina = Binarizer(threshold=1)
bina.fit_transform(x)

# positive확률 열만 가져와서 2차원으로 변환
pred_ = predict_proba[:,1].reshape(-1,1)
b_pred = Binarizer(threshold=0.5).fit_transform(pred_)
## pred_proba를 0.5를 기준으로 0/1 결정
np.concatenate([pred.reshape(-1,1), b_pred], axis=1)


# #### 분류 결정 임계값을 0.4로 변환

def get_eval_by_threshold(y_test, pred_proba, threshold=0.5):
    b_pred = Binarizer(threshold=threshold).fit_transform(pred_proba)
    get_eval_score(y_test, b_pred)


pred_ = lr.predict_proba(X_test)[:,1].reshape(-1,1) ## positive 확률만 가져와 2차원으로
get_eval_by_threshold(y_test, pred_, 0.4)

thresholds = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
for t in thresholds:
    print(f'------threshold = {t}-------')
    get_eval_by_threshold(y_test, pred_, t)
    print()

# => threshold 증가할수록 정밀도 증가, 재현율 감소<br>
# ==> if 정밀도를 높이는데 목적이 있다면 threshold = 0.55 보다 0.6이 적합함(f1, g-measure가 더 높기 때문)<br>
# (* 정밀도와 재현율이 너무 차이나도 안좋음)

# ### roc & auc

## fpr : 1 - 특이도
## tpr : 재현율(민감도)
fpr, tpr, threshold = roc_curve(y_test, pred_)

plt.plot(fpr, tpr, label='ROC')
plt.plot([0,1],[0,1], 'k--', label='Random')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# roc 커브 밑 면적 : 1에 가까울수록 좋음
roc_auc_score(y_test, pred_)

# ### precision_recall_curve()

# threshold 마다 precision, recall
precision, recall, threshold = precision_recall_curve(y_test, pred_)
len(precision), len(recall), len(threshold)

plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.legend()
plt.xlabel('Threshold Value')
plt.ylabel('Preision & Recall Value')
start, end = plt.xlim()
plt.xticks(np.round(np.arange(start, end, 0.1), 2))
plt.grid()
plt.show()


# ### 성능평가지수 출력 함수 수정

def get_eval_score(y_test, pred, pred_proba=None):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test,pred)
    g = np.sqrt(precision*recall)
    print(f'오차행렬\n{confusion_matrix(y_test,pred)}')
    print(f'정확도:{accuracy:.4f}, 정밀도:{precision:.4f}, 재현율:{recall:.4f}, F1:{f1:.4f}, G-measure:{g:.4f}')
    if pred_proba is not None:
        auc = roc_auc_score(y_test, pred_proba)
        print(f'AUC:{auc}')
    else:
        print('')


get_eval_score(y_test, pred)

get_eval_score(y_test, pred, pred_)


