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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# # 군집화 실습: 고객 세그멘테이션
#
# **군집화 기준**
# - RFM 기법
#     - Recency(R): 가장 최근 상품 구입 일에서 오늘까지의 기간
#     - Frequency(F): 상품 구매 횟수
#     - Monetary Value(M): 총 구매 금액
#     
# ## 예제 데이터 : Online Retail Data Set
# https://archive.ics.uci.edu/ml/datasets/online+retail

#
