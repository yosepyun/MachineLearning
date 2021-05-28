#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.decomposition import PCA


# In[2]:


#데이터 불러오고 합치기

datapath = './homework8/avila-tr.txt'

train = pd.read_csv(datapath, sep=",",                            names = ["IntercolumnarDistance","UpperMargin","LowerMargin",                                    "Exploitation", "RowNumber",                                    "ModularRatio","InterlinearSapacing","Weight",                                    "PeakNumber", "ModularRatio/InterlinearSapcing",                                    "Class"])

datapath = "./homework8/avila-ts.txt"

test = pd.read_csv(datapath, sep =",",                   names = ["IntercolumnarDistance","UpperMargin","LowerMargin",                                    "Exploitation", "RowNumber",                                    "ModularRatio","InterlinearSapacing","Weight",                                    "PeakNumber", "ModularRatio/InterlinearSapcing",                                    "Class"])

data = pd.concat([train, test])

data.shape
# Input : "IntercolumnarDistance","UpperMargin","LowerMargin","Exploitation", "RowNumber","ModularRatio","InterlinearSapacing","Weight","PeakNumber", "ModularRatio/InterlinearSapcing", output : "Class"
# classification입니다. 행 20867개, 열 11개로 이루어져있습니다.


# In[3]:


# lable은 A, F, E, I, X, H, G, D, Y, C, W, B 로 12개입니다.

data["Class"].value_counts()


# In[4]:


# 데이터 앞 부분 살펴보기

data.head()


# In[5]:


# 데이터 결측치 확인

data.isnull().sum()


# In[6]:


#lable 값의 불균형 확인

data["Class"].value_counts()


# In[7]:


# label 값의 불균형을 해결하기 위한 downsampling

data_copy = data.copy()
downsampledata = data_copy[data_copy.Class == "A"]
delte_A = data_copy[data_copy['Class']=='A'].index
data_copy=data_copy.drop(delte_A)
downsample = resample(downsampledata, replace = False, n_samples =4000, random_state = 123)
downsample.head()
len(downsample)
df_downsample = pd.concat([data_copy, downsample])


# In[8]:


# label 값의 불균형이 해결되었는지 확인

df_downsample["Class"].value_counts()


# In[9]:


# scale 하기 위해 feature만 불러오기

scale_data = df_downsample.loc[:,["IntercolumnarDistance","UpperMargin","LowerMargin",                                    "Exploitation", "RowNumber",                                    "ModularRatio","InterlinearSapacing","Weight",                                    "PeakNumber", "ModularRatio/InterlinearSapcing"]]


# In[10]:


# feature만 불러온 데이터 앞 부분 확인

scale_data.head()


# In[11]:


# MinMaxScale 을 통해 scale 진행

scaler = MinMaxScaler()
scaler.fit(scale_data)
scale_data = scaler.transform(scale_data)
print(scale_data)


# In[12]:


# scale한 데이터를 데이터프레임형태로 바꾸고 feature 이름 넣어주기

scale_data = pd.DataFrame(scale_data)
scale_data.columns = ["IntercolumnarDistance","UpperMargin","LowerMargin",                                    "Exploitation", "RowNumber",                                    "ModularRatio","InterlinearSapacing","Weight",                                    "PeakNumber", "ModularRatio/InterlinearSapcing"
                                   ]


# In[13]:


# scale한 데이터를 downsampling한 데이터에 넣어주기

df_downsample[["IntercolumnarDistance","UpperMargin","LowerMargin",                                    "Exploitation", "RowNumber",                                    "ModularRatio","InterlinearSapacing","Weight",                                    "PeakNumber", "ModularRatio/InterlinearSapcing"]] = scale_data


# In[14]:


# 데이터 앞 부분 확인

df_downsample.head()


# In[15]:


# RandomForest 1
# 모든 Feature를 선택해서 downsample 데이터로 Modeling

features = ["IntercolumnarDistance","UpperMargin","LowerMargin",                                    "Exploitation", "RowNumber",                                    "ModularRatio","InterlinearSapacing","Weight",                                    "PeakNumber", "ModularRatio/InterlinearSapcing"]

f1_mean = []
# K-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)
fold_idx =1
for  train_index, test_index in kf.split(df_downsample):
    train_data, test_data = df_downsample.iloc[train_index], df_downsample.iloc[test_index]

    train_y = train_data["Class"]
    train_x = train_data[features]
    
    test_y = test_data["Class"]
    test_x = test_data[features]
    
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    
    y_pred=(model.predict(test_x))
    
    # F1 score 값 구하기
    f1 = f1_score(y_pred, test_y, average = "weighted")
    
    print("model1_f1_score = {}".format(f1))
    
    fold_idx+=1
    
    # Model 10개의 평균 f1 score 값
    f1_mean.append(f1)
    
print("######## Total average ########")
    
print("f1_average = {}".format(np.mean((f1_mean))))


# In[16]:


# RandomForest 2
# 모든 Feature를 선택해서 downsample 데이터로 Modeling
# Model parameter 설정

features = ["IntercolumnarDistance","UpperMargin","LowerMargin",                                    "Exploitation", "RowNumber",                                    "ModularRatio","InterlinearSapacing","Weight",                                    "PeakNumber", "ModularRatio/InterlinearSapcing"]
f1_mean = []
# K-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)
fold_idx =1
for  train_index, test_index in kf.split(df_downsample):
    train_data, test_data = df_downsample.iloc[train_index], df_downsample.iloc[test_index]
    
     # PCA를 통해 차원 축소
    pca = PCA(n_components=10)
    
    train_y = train_data["Class"]
    train_x = train_data[features]
    
    test_y = test_data["Class"]
    test_x = test_data[features]
    
    model = RandomForestClassifier(max_depth = 5, n_estimators = 200)
    model.fit(train_x, train_y)
    y_pred=(model.predict(test_x))
    
    # F1 score 값 구하기
    f1 = f1_score(y_pred, test_y, average = "weighted")
    
    print("model2_f1_score = {}".format(f1))
        
    fold_idx+=1
    
    # Model 10개의 평균 f1 score 값
    f1_mean.append(f1)
    
print("######## Total average ########")
    
print("f1_average = {}".format(np.mean((f1_mean))))


# In[17]:


# RandomForest 3
# 모든 Feature를 선택해서 downsample 데이터로 Modeling
# Model parameter 설정

features = ["IntercolumnarDistance","UpperMargin","LowerMargin",                                    "Exploitation", "RowNumber",                                    "ModularRatio","InterlinearSapacing","Weight",                                    "PeakNumber", "ModularRatio/InterlinearSapcing"]
f1_mean = []
# K-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)
fold_idx =1
for  train_index, test_index in kf.split(df_downsample):
    train_data, test_data = df_downsample.iloc[train_index], df_downsample.iloc[test_index]
    
     # PCA를 통해 차원 축소
    pca = PCA(n_components=10)
    
    train_y = train_data["Class"]
    train_x = train_data[features]
    
    test_y = test_data["Class"]
    test_x = test_data[features]
    
    model = RandomForestClassifier(max_depth = 10, n_estimators = 300)
    model.fit(train_x, train_y)
    y_pred=(model.predict(test_x))
    
    # F1 score 값 구하기
    f1 = f1_score(y_pred, test_y, average = "weighted")
    
    print("model3_f1_score = {}".format(f1))
        
    fold_idx+=1
    
    # Model 10개의 평균 f1 score 값
    f1_mean.append(f1)
    
print("######## Total average ########")
    
print("f1_average = {}".format(np.mean((f1_mean))))

# Model parameter와 PCA 를 설정해준 RandomForest Model 2 와 Model 3이 Model 1보다 좋은 성능을 보임


# In[19]:


# MLP 1
# 모든 Feature를 선택해서 downsample 데이터로 Modeling
# Model parameter 설정

features = ["IntercolumnarDistance","UpperMargin","LowerMargin",                                    "Exploitation", "RowNumber",                                    "ModularRatio","InterlinearSapacing","Weight",                                    "PeakNumber", "ModularRatio/InterlinearSapcing"]
f1_mean = []
# K-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)
fold_idx =1
for  train_index, test_index in kf.split(df_downsample):
    train_data, test_data = df_downsample.iloc[train_index], df_downsample.iloc[test_index]
    
    # PCA를 통해 차원 축소
    pca = PCA(n_components = 10)
    
    train_y = train_data["Class"]
    train_x = pca.fit_transform(train_data[features])
    
    test_y = test_data["Class"]
    test_x = pca.transform(test_data[features])
    
    model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='relu',                     solver='adam', alpha=0.01, batch_size=32,                     learning_rate_init=0.1, max_iter=500)
    model.fit(train_x, train_y)
    y_pred=(model.predict(test_x))
    
    # F1 score 값 구하기
    f1 = f1_score(y_pred, test_y, average = "weighted")
    
    print("model_f1_score = {}".format(f1))
        
    fold_idx+=1
    
    # Model 10개의 평균 f1 score 값
    f1_mean.append(f1)
    
print("######## Total average ########")
print("최종 모델 : MLP Classifier")
print("f1_average = {}".format(np.mean((f1_mean))))

# 최종 모델로 MLP를 선택했습니다.
## Parameter Engineering ##
# 1개의 입력 계층 3개의 은닉 게층, 1개의 출력 계층으로 이루어진 MLP입니다.
# 각 은닉 계층은 10개의 퍼셉트론으로 이루어져있고 Relu를 사용했습니다.


# In[ ]:




