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

train_input.shape, train_target.shape

test_input.shape, test_target.shape

# - 데이터 차원 변경 및 정규화
#     - 곱신경망 => 4차원 데이터로 변경

train_scaled = train_input.reshape(-1, 28, 28, 1)/255.0
test_scaled = test_input.reshape(-1,28,28,1)/255.0

train_scaled.shape, test_scaled.shape

# #### 데이터 분할

X_tr, X_val, y_tr, y_val = train_test_split(train_scaled, train_target, test_size=0.2, random_state=205)

X_tr.shape, X_val.shape

# ### 모델정의
#
# ```python
# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), adding='valid', data_format=None, dilation_rate=(1, 1), groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
# ```

from keras.layers import Conv2D, MaxPooling2D


def build_cnn_model():
    model = Sequential()
    
    ### 합성곱, 맥스풀링층
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2))
    
    ### 덴스, 드롭아웃
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    
    ### 출력층
    model.add(Dense(10, activation='softmax'))
    
    return model


model_cnn = build_cnn_model()
model_cnn.summary()

from keras.utils import plot_model

plot_model(model_cnn, show_shapes=True, to_file='model/cnn-model.png', dpi=300)

# #### 모델 컴파일

model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

# #### 모델 학습

from keras.callbacks import ModelCheckpoint, EarlyStopping

# +
checkpt = ModelCheckpoint('model/best-cnn-model.h5')
earlystop = EarlyStopping(patience=2, restore_best_weights=True)

history_cnn = model.fit(X_tr, y_tr, epochs=20, validation_data=(X_val, y_val), callbacks=[checkpt, earlystop])
# -

# #### 손실검증

draw_loss_val_plot(history_cnn)


# #### 모델 재정의

def build_cnn_model():
    model = Sequential()
    
    ### 합성곱, 맥스풀링층
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2))
    
    ### 덴스, 드롭아웃
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    
    ### 출력층
    model.add(Dense(10, activation='softmax'))
    
    return model


model_cnn2 = build_cnn_model()

# +
model_cnn2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

chekpt2 = ModelCheckpoint('model/best-cnn-model2.h5')
earlystop2 = EarlyStopping(patience=2, restore_best_weights=True)

history_cnn2 = model_cnn2.fit(X_tr, y_tr, epochs=20, validation_data=(X_val, y_val), callbacks=[chekpt2, earlystop2])
draw_loss_val_plot(history_cnn2)
# -

# #### 모델평가
# - Early Stopping의 restore_best_weights 매개변수 True로 지정되어 현재 모델의 객체가 최적의 모델파라미터로 복원되어 있음
#     - False였으면 epoch stopping 했을 때의 매개변수가 저장됨
#
#
# => ModelCheckpoint 콜백이 저장한 파일은 다시 읽을 필요가 없음

model_cnn2.evaluate(X_val, y_val)

# => DNN보다 성능 향상

# #### 예측

target_labels = {0: 'T_shirt/top',
                 1: 'Trouser',
                 2: 'Pullover',
                 3: 'Dress',
                 4: 'Coat',
                 5: 'Sandal',
                 6: 'Shirt',
                 7: 'Sneakers',
                 8: 'Bag',
                 9: 'Ankle Boots'}

# - 원래 target

# +
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(X_val[i].reshape(28,28), cmap='gray_r')
    plt.title(target_labels[y_val[i]])
    
plt.tight_layout()
plt.show()
# -

# - 예측 target

pred_proba = model_cnn2.predict(X_val)
preds = np.argmax(pred_proba, axis=1)

# +
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(X_val[i].reshape(28,28), cmap='gray_r')
    if preds[i] == y_val[i]:
        plt.title(target_labels[preds[i]], fontdict={'color':'green'})
    else:
        plt.title(target_labels[preds[i]], fontdict={'color':'red'})
    
plt.tight_layout()
plt.show()
# -

test_pred_proba = model_cnn2.predict(test_scaled)
test_preds = np.argmax(test_pred_proba, axis=1)

n = 53
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(test_scaled[i+n].reshape(28,28), cmap='gray_r')
    if test_preds[i+n] == test_target[i+n]:
        plt.title(target_labels[test_preds[i+n]], fontdict={'color':'green'})
    else:
        plt.title(target_labels[test_preds[i+n]], fontdict={'color':'red'})
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

# ----

# ## CNN 시각화
# - 가중치 시각화
# - 특성맵 시각화

model_cnn2.layers

# ### 첫 번째 합성곱 층의 가중치 시각화

conv1 = model_cnn2.layers[0]
print('가중치의 크기', conv1.weights[0].shape)
print('절편의 크기', conv1.weights[1].shape)

## 텐서형식의 가중치를 numpy로 변환
conv1_weights = conv1.weights[0].numpy()
conv1_weights.shape

conv1_weights.mean(), conv1_weights.std()

plt.hist(conv1_weights.reshape(-1, 1))  ## 1차원 형식으로 변환
plt.xlabel('weight')
plt.ylabel('count')
plt.title("Conv1 layer's weights dist.")
plt.show()

# - 첫 번째 Conv층 필터들 시각화

plt.figure(figsize=(15,2))
for i in range(32):
    plt.subplot(2,16,i+1)
    plt.imshow(conv1_weights[:,:,0,i], vmin=-0.5, vmax=0.5)
    plt.axis('off')
plt.tight_layout()
plt.show()

# => 필터의 가중치들이 다 다름

# #### 훈련하지 않은 빈 합성곱 신경망의 필터 시각화

no_model_cnn = build_cnn_model()

no_conv1 = no_model_cnn.layers[0]
no_conv1.weights[0].shape

# +
no_train_weights = no_conv1.weights[0].numpy()

plt.hist(no_train_weights.reshape(-1, 1))  ## 1차원 형식으로 변환
plt.xlabel('weight')
plt.ylabel('count')
plt.title("no train Conv1 layer's weights dist.")
plt.show()
# -

# => 훈련 전에는 균일분포 형태

plt.figure(figsize=(15,2))
for i in range(32):
    plt.subplot(2,16,i+1)
    plt.imshow(no_train_weights[:,:,0,i], vmin=-0.5, vmax=0.5)
    plt.axis('off')
plt.tight_layout()
plt.show()

# #### 참고) 신경망모델 구성을 위한 함수형 API

# +
inputs = keras.Input(shape=(784,))  ## 28x28

dense1 = Dense(100, activation='sigmoid')
dense2 = Dense(10, activation='softmax')
hidden = dense1(inputs)
outputs = dense2(hidden)

api_model = keras.Model(inputs=inputs, outputs=outputs)
api_model.summary()
# -

plot_model(api_model, show_shapes=True)

# ### 특성맵 시각화

model_cnn2.input

model_cnn2.layers[0].output

conv_acti_1 = keras.Model(model_cnn2.input, model_cnn2.layers[0].output)

inputs = train_scaled[0].reshape(-1,28,28,1)
feature_maps = conv_acti_1.predict(inputs)
feature_maps.shape

# - 원래 데이터

plt.imshow(train_scaled[0])
plt.show()

# - 필터 적용 특성맵

# +
plt.figure(figsize=(15,8))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.imshow(feature_maps[0,:,:,i])
    plt.axis('off')
    
plt.tight_layout()
plt.show()
# -

conv_acti_2 = keras.Model(model_cnn2.input, model_cnn2.layers[2].output)
feature_maps2 = conv_acti_2.predict(inputs)
feature_maps2.shape

plt.figure(figsize=(15,15))
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.imshow(feature_maps2[0,:,:,i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# => 합성곱 신경망의 앞부분 합성곱 층은 이미지의 시각적 정보를 감지하며, 뒤쪽의 합성곱층은 앞쪽에서 감지한 정보를 바탕으로 추상적인 정보를 학습함

# ---
