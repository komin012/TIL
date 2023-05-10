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
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

SEED = 12
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Reshape, BatchNormalization
# -

# # CNN: 전이학습

# ## 예제. CIFAR10 이미지 데이터

# +
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train/255
X_test = X_test/255

X_train.shape, X_test.shape
# -

plt.figure(figsize=(10,8))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(X_train[i])
    plt.axis('off')
plt.show()


# ### CNN 모델 구축

# +
def build_cnn():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5, strides=(2,2),  ## stride: 이동 크기
                    activation='relu', input_shape=[32,32,3]))
    model.add(BatchNormalization()) ## 정규화
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=5, strides=(2,2),
                    activation='relu'))
    model.add(BatchNormalization()) ## 정규화
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                 metrics='accuracy')
    
    return model

cnn_model = build_cnn()
cnn_model.summary()
# -

cnn_history = cnn_model.fit(X_train, y_train, batch_size=256, epochs=10,
                           validation_split=0.2, verbose=0)

cnn_history.history['loss'][:20]


def plot_metrics(history, start=1, end=10):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Loss: 손실 함수
    axes[0].plot(range(start, end+1), history.history['loss'][start-1:end], 
                label='Train')
    axes[0].plot(range(start, end+1), history.history['val_loss'][start-1:end], 
                label='Validation')
    axes[0].set_title('Loss')
    axes[0].legend()
    # Accuraccy: 예측 정확도
    axes[1].plot(range(start, end+1), history.history['accuracy'][start-1:end], 
                label='Train')
    axes[1].plot(range(start, end+1), history.history['val_accuracy'][start-1:end], 
                label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    plt.show()


plot_metrics(cnn_history)

# ### 전이학습(transfer learning)
# -  이미 충분한 데이터와 여러 연구와 실험으로 만들어진 모델을 학습한 가중치를 가지고 우리 모델에 맞게 재보정해서 사용하는 것

# +
# pre-trained 모델 가져오기 (ResNet50)
from keras.applications import ResNet50

cnn_base = ResNet50(include_top=False, # 출력층 제외
                    weights='imagenet',
                    input_shape=[32,32,3],
                    classes=10)


# +
def build_transfer():
    transfer_model = Sequential()
    transfer_model.add(cnn_base)
    transfer_model.add(Flatten())
    transfer_model.add(Dense(units=64, activation='relu'))
    transfer_model.add(Dropout(0.5))
    transfer_model.add(Dense(units=32, activation='relu'))
    transfer_model.add(Dropout(0.5))
    transfer_model.add(Dense(units=10, activation='softmax'))
    
    transfer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                          metrics='accuracy')
    return transfer_model

transfer_model = build_transfer()
transfer_model.summary()
# -

tm_history = transfer_model.fit(X_train, y_train, batch_size=256, epochs=10,
                               validation_split=0.1)
plot_metrics(tm_history)


