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
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
# -

# ## 데이터 준비

from tensorflow import keras
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

# +
(train_input, train_target),(test_input, test_target) = fashion_mnist.load_data()

# 정규화
train_scaled = train_input/255
test_scaled = test_input/255

X_train, X_val, y_train, y_val = train_test_split(train_scaled, train_target, test_size=0.2, random_state=205)
# -

# **Fashion_Mnist 레이블 정보**
# - 0: T_shirt/top
# - 1: Trouser
# - 2: Pullover
# - 3: Dress
# - 4: Coat
# - 5: Sandal
# - 6: Shirt
# - 7: Sneakers
# - 8: Bag
# - 9: Ankle Boots

# ## 모델 생성

# +
from keras import Sequential
from keras.layers import Dense, Flatten

def build_model(a_layer=None):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28))) ## 2차원을 1차원으로 풀어줌
    model.add(Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(Dense(10, activation='softmax'))
    return model


# -

model = build_model()
model.summary()

# ## 모델 컴파일 및 훈련

# +
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
## 원핫인코딩이 안된 범주형 변수의 경우 sparse_categorical_crossentropy 사용

history = model.fit(X_train, y_train, epochs=5)


# -

# ## 손실곡선

def draw_loss_metric_plot(history):
    plt.figure(figsize=(10,5))
    for i, item in enumerate(history.history.keys()):
        plt.subplot(1,2,i+1)
        plt.plot(history.history[item])
        plt.xlabel('epoch')
        plt.ylabel(item)
        plt.title(item+' Plot')
    plt.show()


draw_loss_metric_plot(history)

# ## epochs=20인 경우

model2 = build_model()
model2.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history2 = model2.fit(X_train, y_train, epochs=20)

draw_loss_metric_plot(history2)

# ## 검증손실

model3 = build_model()
model3.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history3 = model3.fit(X_train, y_train, epochs=20, verbose=0, validation_data=(X_val,y_val))

history3.history.keys()


def draw_loss_val_plot(history):
    loss = ['loss', 'val_loss']
    acc = ['accuracy', 'val_accuracy']
    title = ['Loss', 'Accuracy']
    plt.figure(figsize=(10,5))
    for i, item in enumerate([loss, acc]):
        plt.subplot(1,2,i+1)
        plt.plot(history.history[item[0]], label=item[0])
        plt.plot(history.history[item[1]], label=item[1])        
        plt.xlabel('epoch')
        plt.ylabel(title[i])
        plt.legend()
        plt.title(title[i]+' Plot')
    plt.show()


draw_loss_val_plot(history3)

# 학습용 데이터 loss와 검증용 데이터 loss가 반대방향을 보이는 것은 과대적합이라는 뜻<br>
# => epochs=6 이상부터 loss 값 증가하므로 과적합

# ### optimizer='adam'

model4 = build_model()
model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history4 = model4.fit(X_train, y_train, epochs=20, verbose=0, validation_data=(X_val, y_val))
draw_loss_val_plot(history4)

# -> 검증손실(val_loss)가 더 안정적으로 떨어지고 있음

# #### adam의 학습률 조정

model5 = build_model()
adam = keras.optimizers.Adam(learning_rate=0.0005)
model5.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics='accuracy')
history5 = model5.fit(X_train, y_train, epochs=20, verbose=0, validation_data=(X_val, y_val))
draw_loss_val_plot(history5)

# -> 검증손실(loss_val)이 더 안정적으로 떨어짐

# ## 드롭아웃(dropout)
# - 은닉층에서 뉴런의 출력을 랜덤하게 꺼서 과대적합을 막음
#     - 이전 층의 일부 뉴런이 랜덤하게 꺼지면 특정 뉴런에 과대하게 의존하는 것을 줄일 수 있음

from keras.layers import Dropout

# 30% 드롭아웃
model6 = build_model(Dropout(0.3))
model6.summary()

model6.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history6 = model6.fit(X_train, y_train, epochs=20, verbose=0, validation_data=(X_val, y_val))
draw_loss_val_plot(history6)

# ## 모델 저장과 복원

# #### 훈련된 모델의 파라미터 저장: save_weights()

model6.save_weights('model/model-weights.h5')

# #### 훈련된 모델의 구조와 파라미터 모두 저장: save()

model6.save('model/model-whole.h5')

# #### 저장된 훈련모델 가중치 적용
# - load_weights() 사용 시 save_weights()로 저장된 모델과 정확히 같은 구조를 가져야 함

model7 = build_model(Dropout(0.3))
model7.load_weights('model/model-weights.h5')

val_proba = model7.predict(X_val)
val_proba[:5]

val_labels = np.argmax(val_proba, axis=-1)
val_labels[:5]

y_val[:5]

# - load_model(): 모델구조, 가중치 모두 복원

from keras.models import load_model

model8 = load_model('model/model-whole.h5')
model8.evaluate(X_val, y_val)

# ## 콜백(callback)
# - 훈련 과정 중간에 어떤 작업을 수행할 수 있도록 도와주는 객체

# ### 1. ModelCheckpoint
# - 최상의 검증 점수를 만드는 모델 저장

from keras.callbacks import ModelCheckpoint

model9 = build_model(Dropout(0.3))
model9.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = ModelCheckpoint('model/best-model.h5') ## 저장
model9.fit(X_train, y_train, epochs=20, verbose=0, validation_data=(X_val, y_val), callbacks=[checkpoint_cb])

# #### 최상의 검증 점수를 낸 모델 다시 읽어 예측하기

model10 = load_model('model/best-model.h5')
model10.evaluate(X_val, y_val)

# ### 2. EarlyStopping
# - 과대적합이 되기 전에 훈련을 미리 중지하는 것
#     - 검증점수가 더 이상 감소하지 않고 상승하여 과대적합이 일어나면 훈련을 계속 진행하지 않고 멈추는 기법

# EarlyStopping(monitor='val_loss', patience, restore_best_weights=False)
#
# - patience: 검증 점수가 향상되지 않더라도 지속할 수 있는 최대 에포크 횟수
# - restore_best_weights: 최상 모델 가중치를 복원할지 지정

from keras.callbacks import EarlyStopping

# +
model11 = build_model(Dropout(0.3))
model11.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

earlystop_cb = EarlyStopping(patience=2, restore_best_weights=True)
checkpt_cb = ModelCheckpoint('model/best-model2.h5')

history11 = model11.fit(X_train, y_train, epochs=20, verbose=2, validation_data=(X_val, y_val), callbacks=[earlystop_cb, checkpt_cb])
# -

# -> epochs = 20 전에 중단됨

draw_loss_val_plot(history11)

model11.evaluate(X_val, y_val)

model11.evaluate(test_input, test_target)
