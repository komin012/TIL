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
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# # 예제. Fashion-MNIST 데이터

# +
from keras.datasets import fashion_mnist

(train_input, train_target),(test_input, test_target) = fashion_mnist.load_data()
# -

train_input.shape, train_target.shape

test_input.shape, test_target.shape

# +
fig, axs = plt.subplots(figsize=(24,10), ncols=10)

for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
plt.show()
# -

train_input[0].dtype

np.unique(train_target, return_counts=True)

# ## 네트워크에 맞는 이미지 데이터로 준비
# - [0,255] 사이의 값인 uint8 타입의 (60000, 28, 28)크기를 가진 배열을 0과 1사이의 값을 가지는 float32 타입의 (60000, 28 * 28) 크기 배열로 변환
#
# ※ unit8 타입은 0~255사이만 반환되기 때문에 사칙연산이 잘 안됨

# +
train_input = train_input.reshape((60000, 28*28)) ## 2차원으로 변경
train_input = train_input.astype('float32')/255

test_input = test_input.reshape((10000, 28*28)) ## 2차원으로 변경
test_input = test_input.astype('float32')/255
# -

# - 타깃 레이블을 범주형으로 인코딩

from keras.utils import to_categorical

train_target = to_categorical(train_target)
train_target[0]

test_target = to_categorical(test_target)
test_target[0]

# ## 모델 설계 및 실행
# ### 1) 신경망 모델 정의
# #### Sequential 클래스로 신경망 모델 생성

# Dense(뉴런 개수, activation='뉴런 출력에 적용할 함수', input_shape=(입력크기))
#
# - 활성화 함수
#     1. 은닉층 활성화 함수: 자유로움
#         - 시그모이드 함수
#         - 렐루(ReLU) 함수
#     2. 출력층 활성화 함수: 종류 제한
#         - 시그모이드 함수: 이진 분류인 경우
#         - 소프트맥스 함수: 다중 분류인 경우
#         - 회귀의 경우 적용 X

# +
from keras.models import Sequential
from keras.layers import Dense

network = Sequential()
network.add(Dense(100, activation='relu', input_shape=(28*28,))) ## 은닉층
network.add(Dense(10, activation='softmax')) ## 출력층(타겟 분류 개수만큼)

network.summary()
# -

# #### 모델 시각화: plot_model()

# +
from keras.utils import plot_model

plot_model(network, show_shapes=True)
# -

# ### 2) 모델 컴파일
# 1. optimizer : 최적화 알고리즘
#     - 신경망의 가중치와 절편을 학습하기 위한 알고리즘
#     - SGD
#     - 네스트로프 모멘텀
#     - RMSprop
#     - Adagrad
#     - Adam
# 2. loss: 손실함수
#     - 이진분류: binary_crossentropy
#     - 다중분류: categorical_crossentropy
#     - 정수레이블을 갖는 다중 분류: sparse_categorical_crossentropy
#     - 회귀모델: mean_square_error
# 3. metrics: 정확도 지표
#     - accuracy

network.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

# ### 3) 모델 실행(훈련)
# fit(x, y, epochs, batch_size, verbose)
#
#     - epochs: 모든 학습 데이터 셋을 학습하는 횟수
#     - verbose: 'auto', 0, 1, 2
#         - 0: silent
#         - 1, 'auto': progress bar
#         - 2: one line per epoch

history = network.fit(train_input, train_target, epochs=5, batch_size=127)

history.history

history.history['loss']

history.history['accuracy']

# #### 손실(loss) 곡선

plt.plot(history.history['loss'], label='loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['accuracy'], label='loss')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# ### 4) 모델 예측(평가)

test_loss, test_acc = network.evaluate(test_input, test_target)

# ----
