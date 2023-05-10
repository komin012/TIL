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

# # 오토인코더: 차원축소와 이미지 복원
# - 입력을 출력으로 복사하는 신경망

# +
# 라이브러리 설정
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Reshape

# 랜덤 시드 고정
SEED=12
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)  

# +
from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 스케일링
X_train = X_train/255
X_test = X_test/255

# 차원추가
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

X_train.shape, X_test.shape


# -

# #### 오토인코더: Autoencoder() 정의

# +
def Autoencoder():
    model = Sequential()
    
    # Encoder 부분
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=[28,28,1]))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    
    # Decoder 부분
    model.add(Dense(units=28*28, activation='sigmoid'))
    model.add(Reshape((28,28)))
    
    # Compile
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

ae_model = Autoencoder()
ae_model.summary()
# -

# ## 이미지 복원

history = ae_model.fit(X_train, X_train, batch_size=64, epochs=20,
                      validation_data=(X_test, X_test))
images = ae_model.predict(X_test)
images.shape

plt.figure(figsize=(20,8))
for i in range(5):
    # 원본 이미지
    ax = plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape((28,28)), cmap='gray_r')
    plt.axis('off')
    
    # 복원 이미지
    ax = plt.subplot(2, 5, i+6)
    plt.imshow(images[i], cmap='gray_r')
    plt.axis('off')
plt.tight_layout()
plt.show()


