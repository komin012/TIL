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
import tensorflow as tf
from tensorflow import keras

# # Fashion MNIST 데이터

from keras.datasets import fashion_mnist

(train_input, train_target),(test_input, test_target) = fashion_mnist.load_data()

train_input.shape, train_target.shape

test_input.shape, test_target.shape

# ## 전처리

train_input.dtype

# 정규화
train_scaled = train_input/255
test_scaled = test_input/255

# +
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(train_scaled, train_target, test_size=0.2)
# -

# ## 모델 구축 및 훈련

# ### 1. DNN

from keras import Sequential
from keras.layers import Dense, Flatten, Dropout


def build_model(a_layer=None):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(Dense(10, activation='softmax'))
    return model


model = build_model()
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

history = model.fit(X_tr, y_tr, epochs=20, verbose=0)


def draw_loss_metric_plot(history):  
    plt.figure(figsize=(10, 5))
    for i, item in enumerate(history.history.keys()):
        plt.subplot(1,2,i+1)
        plt.plot(history.history[item],'bo-', label=item)
        plt.xlabel('epoch')
        plt.ylabel(item)
        plt.title(item+' Plot')
        plt.legend()
    plt.show()


draw_loss_metric_plot(history)

# #### 손실 검증

model2 = build_model()
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history2 = model2.fit(X_tr, y_tr, epochs=20, verbose=0, validation_data=(X_val, y_val))


def draw_loss_val_plot(history):
    loss = ['loss', 'val_loss']
    acc = ['accuracy', 'val_accuracy']
    title = ['Loss', 'Accuracy']
    plt.figure(figsize=(10, 5))
    for i,item in enumerate([loss, acc]):
        plt.subplot(1,2,i+1)
        plt.plot(history.history[item[0]],'bo-', label=item[0])
        plt.plot(history.history[item[1]],'ro-', label=item[1])
        plt.xlabel('epoch')
        plt.ylabel(title[i])
        plt.legend()
        plt.title(title[i]+' Plot')
    plt.tight_layout()
    plt.show()


draw_loss_val_plot(history2)

# #### 모델 성능 개선

# - 드롭 아웃

model3 = build_model(Dropout(0.3))
model3.summary()

model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history3 = model3.fit(X_tr, y_tr, epochs=20, verbose=0, validation_data=(X_val, y_val))
draw_loss_val_plot(history3)

# - 콜백

from keras.callbacks import ModelCheckpoint, EarlyStopping

# +
model4 = build_model(Dropout(0.3))
model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

checkpt = ModelCheckpoint('model/fashion_bestmodel.h5')
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

history4 = model4.fit(X_tr, y_tr, epochs=20, verbose=2, validation_data=(X_val, y_val), callbacks=[checkpt, early_stop])
draw_loss_val_plot(history4)
# -

# #### 모델 평가

model4.evaluate(X_val, y_val)

# #### 예측

y_pred_proba = model4.predict(test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

y_pred[:10]

test_target[:10]

# ## 2. CNN

(train_input, train_target),(test_input, test_target) = fashion_mnist.load_data()


