# 합성곱 신경망(CNN)

### 심층 신경망(DNN)의 밀집층
- 밀집층에는 뉴런마다 입력개수 만큼의 가중치가 있으며, 모든 입력에 가중치를 곱함

## 합성곱(convolution)
- 밀집층과 비슷하게 입력과 가중치를 곱하고 절편을 더하는 선형계산을 수행
- 밀집층과 다르게 각 합성곱은 입력 전체가 아니라 일부만 사용하여 선형 계산을 수행

### 합성곱의 밀집층
- 심층신경망의 밀집층 계산과 다름
- 입력 데이터 전체의 가중치를 적용하지 않고 일부에만 가중치를 곱함

### 필터
- 합성곱 신경망의 뉴런
- 커널(kernel)

### 합성곱의 장점
- 1차원이 아닌 2차원 입력에 적용 가능
- 입력이 2차원 배열인 경우 필터도 2차원


### 연산과정
![](https://taewanmerepo.github.io/2018/01/cnn/filter.jpg)
(http://taewan.kim/post/cnn/)
- 왼쪽 위 모서리부터 입력의 9개 원소와 커널의 9개 가중치를 곱한 후 절편을 더하여 1개의 출력 생성
- 필터가 이동하면서 합성곱 수행
- 합성곱 계산을 통해 얻은 출력을 **특성맵(feature map)** 이라고 함

#### 여러 개의 필터를 사용한 합성곱 층
![](https://taewanmerepo.github.io/2018/01/cnn/conv2.jpg)
(http://taewan.kim/post/cnn/)


### 케라스 합성곱 층
```python
from tensorflow.keras.layers import Conv2D
Conv2D(10, kernel_size=(3,3), activation='relu')
```
-첫 번째 매개변수: 합성곱 필터 개수
- kernel_size: 필터의 커널 크기 지정
    - 가로세로 크기가 같은 경우 정수 하나로 지정
    - 가로세로 크기가 다른 경우 튜플로 지정
    - 커널의 깊이는 입력의 깊이와 동일하므로 따로 정하지 않음
- strides: 필터의 이동 간격
    - 가로세로 크기가 같은 경우 정수 하나로 지정
    - 가로세로 크기가 다른 경우 튜플로 지정
    - default: 1
- padding: 입력의 패딩타입 지정
    - default: 'valid'(패딩을 하지 않음)
    - 'same': 합성곱 층의 출력의 크기를 입력과 동일하게 맞추도록 입력에 패딩을 추가함
- activation: 합성곱 층에 적용할 활성화 함수 지정

- 커널의 크기는 하이퍼파라미터로 여러가지 값을 시도해봐야 하며, 일반적으로 (3,3)이나 (5,5) 크기가 권장됨

## 패딩(padding)과 스트라이드(strides)
### 패딩(padding)
- 배열 주위를 가상의 원소로 채우는 것
- 실제 입력값이 아니므로 패딩은 0으로 채움
![](https://taewanmerepo.github.io/2018/01/cnn/padding.png)
(http://taewan.kim/post/cnn/)
- 패딩이 없는 경우 모서리에 있는 픽셀(원소)들보다 중앙부에 있는 원소들이 참여하는 비율이 더 많음
- 중요한 정보가 모서리에 있는 특성 맵으로 잘 전달되지 않은 경우가 발생할 수 있음

#### 세임 패딩(same padding)
- 입력과 특성 맵의 크기를 동일하게 만들기 위해 입력 주위에 0을 패딩하는 것
- 합성곱 신경망에서 많이 사용됨
```python
Conv2D(10, kernel_size=(3,3), activation='relu', padding='same')
```

#### 밸리드 패딩(valid padding)
- 패딩없이 순수한 입력 배열에 합성곱을 하여 특성 맵을 만들어 특성 맵의 크기가 줄어든 경우
```python
Conv2D(10, kernel_size=(3,3), activation='relu', padding='valid')
```

### 스트라이드(stride)
- 커널의 이동 크기

<br>

## 풀링(pooling)
- 합성곱 층에서 만든 특성 맵의 가로세로 크기를 줄이는 기법(특성 맵의 개수는 줄어들지 않음)
- 특성 맵에 커널 없는 필터를 적용하는 것과 비슷
- 필터와 다르게 겹치지 않고 이동
- 풀링 계산 시 가중치 없음
- 가로세로 방향으로만 진행
- 평균 풀링보다 최대 풀리을 더 많이 사용

![](https://taewanmerepo.github.io/2018/02/cnn/maxpulling.png)
(http://taewan.kim/post/cnn/)

### 최대풀링(Max Pooling)
- 특성 맵 요소 중 가장 큰 값을 선택
```python
from keras.layers import MaxPooling2D
MaxPooling2D(2, strides=2, padding='valid')
```
- 첫 번째 매개변수: 풀링의 크기
- strides: 기본값은 자동으로 풀링의 크기이므로 따로 지정할 필요 없음
- padding: 기본값은 'valid'로 거의 변경하지 않음

### 평균풀링(Average Pooling)
- 특성 맵 요소의 평균값을 선택
```python
from keras.layers import AveragePooling2D
AveragePooling2D(2, strides=2, padding='valid')
```