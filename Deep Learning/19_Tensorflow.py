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

import tensorflow as tf
tf.__version__

# # 텐서(tensor)
# - 다차원 넘파이 배열
# - 임의의 차원 개수를 가지는 행렬의 일반화된 모습
# - 텐서에 차원을 축(axis) 또는 랭크(rank)라고 부름
#
# **실제 사례**
# - 일반적인 데이터셋 형식: 2D텐서 (samples, features)
#     - 나이, 성별, 소득으로 구성된 데이터 인구통계 데이터셋
#         - (100000,3)
#     - 각 문서별 2만개 단어로 구성된 500개 문서 데이터셋
#         - (500, 20000)
# - 시계열 데이터: 3D텐서 (samples, times, features)
#     - 주식가격 데이터셋: 1분마다 현재주식가격, 최고가격, 최소가격 저장, 하루 거래시간 390분, 250일치 데이터
#         - (250, 390, 3)
#     - 트윗 데이터셋: 각 트윗은 128개 알파벳으로 구성된 280개 문자 시퀀스로 100만개 트윗 데이터
#         - (1000000, 280, 128)
# - 이미지 데이터: 4D텐서 (samples, height, width, color_depth)
#     - 흑백 이미지
#         - (128, 256, 256, 1)
#     - 컬러 이미지
#         - (128, 256, 256, 3)
# - 비디오 데이터: 5D텐서(samples, frames, height, width, color_depth)
#     - 60초짜리 144*256 유튜브 비디오 클립을 4초 프레임으로 샘플링하면 240프레임, 이런 비디오 클립 4개인 배치 데이터
#         - (4, 240, 144, 256, 3)

# ## 1. 스칼라(0D 텐서)
# - 하나의 숫자만 담고 있는 텐서
# - 0차원 텐서 또는 rank 0 텐서

# +
import numpy as np

x = np.array(12)
x
# -

x.ndim  ##  차원

x.shape

# ## 2. 벡터(1D 텐서)
# - 숫자 배열
# - 하나의 축을 가짐
# - rank -1 텐서

x = np.array([12, 3, 4, 5, 10, 7])
x

x.ndim

x.shape

# ## 3. 행렬(2D 텐서)
# - 벡터의 배열
# - 2개의 축: 행(row)과 열(column)

x = np.array([[1,2,3,4,5], [3,5,2,1,0]])
x

x.ndim

x.shape

# ## 4. 3D 텐서와 고차원 텐서

x = np.array([[[5,7,1],[1,2,3]],
             [[1,4,5],[3,1,0]]])
x

x.ndim

x.shape

# ## 텐서 연산
# ### 1. 원소별 연산 (element-wise operation)
# - 곱셉, 덧셈, 뺄셈

x = np.random.random((20, 100))
y = np.random.random((20, 100))
z = x+y
x.shape, y.shape, z.shape

# ### 2. 브로드캐스팅(broadcasting)
# 1단계. 큰 텐서의 ndim에 맞도록 작은 텐서에 (브로드캐스팅 축이라고 부르는) 축이 추가됨<br>
# 2단계. 작은 텐서가 새 축을 따라서 큰 텐서의 크기에 맞도록 반복됨

# +
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x,y)

x.shape, y.shape, z.shape
# -

print(z[1,1,1])

# ### 3. 텐서 점곱(dot operation)
# - np.dot(x,y)
# - 텐서 원소간 연산

# +
x = np.random.random((2,3))
y = np.random.random((3,8))
z = np.dot(x,y)

x.shape, y.shape, z.shape
# -

# ## 텐서 크기 변환
# - reshape()
# - transpose()
#
# ## 텐서 연산의 기하학적 해석
# - 텐서의 내용은 기하학적 공간에 있는 좌표 포인트
# - 기하학적 연산은 텐서 연산으로 표현
#     - 이동(traslatioin)
#     - 회전(rotation)
#     - 크기변경(scaling)
#     - 기울이기(skewing)
# - 선형변환(linear transform)
#     - 임의의 행렬과 점곱(dot)하면 선형변환 수행
# - 아핀변환(Affin transformation)
#     - (어떤 행렬과 점곱하여 얻는) 선형변환과 (벡터를 더해 얻는) 이동의 조합
#     - $y = Wx+b$
#     
# ## 신경망의 기하학적 해석
# - 신경망은 텐서 연산의 연결로 구성된 것으로 모든 텐서 연산은 입력 데이터의 기하학적 변환
# - 신경망은 고차원 공간에서 매우 복잡한 기하학적 변환을 하는 것
# - ex. 3차원 공간에서 빨간색 색종이와 파란색 색종이 2장을 겹친 후 뭉쳐서 작은 공으로 만든 경우
#     - 종이 공이 입력데이터
#     - 색종이는 분류 문제의 데이터 클래스
#     - 신경망은 종이 공을 펼쳐서 두 클래스가 다시 깔끔하게 분리되는 변환을 찾는 것
# - 고차원 공간에서 복잡하고 심하게 꼬여있는 데이터의 매니폴드에 대한 깔끔한 표현을 찾는 일
#     - 매니폴드: 국부적으로 저차원 유클리드 거리로 볼 수 있는 고차원 공간

# # 예제. MNIST 데이터셋

from tensorflow.keras.datasets import mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images.shape

test_images.shape

train_labels.shape

test_labels.shape

digit = train_images[205]
digit

digit.shape

# +
import matplotlib.pyplot as plt

plt.imshow(digit, cmap='gray_r')
plt.show()
# -

# ## 배치(batch) 데이터
# - 딥러닝에서 데이터 사용 시 전체 데이터셋을 처리하지 않음
# - 작은 배치(batch)로 나누어 학습

batch_size = 127
# batch = train_images[:batch_size]
# batch = train_images[batch_size:batch_size*2]
# batch = train_images[batch_size*n:batch_size*(n+1)]
