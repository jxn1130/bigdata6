#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.layers import LSTM
from deepctr.models import DeepFM
import numpy as np
from deepctr.feature_column import SparseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.layers import Embedding, Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names


train = pd.read_csv("./dataset/book/train.csv")
test = pd.read_csv("./dataset/book/test.csv")


# In[2]:


# 'User-ID'를 기준으로 각 행에 해당하는 'Book-Rating'의 평균 값을 가져와서 새로운 열로 추가
train['Rating-Mean'] = train['User-ID'].map(train.groupby('User-ID')['Book-Rating'].mean())

train['individual_point'] = train['Rating-Mean']-train['Book-Rating']




train['new_Rating'] = ((train['individual_point'] - train['individual_point'].min()) / (train['individual_point'].max() - train['individual_point'].min())) * 10




combined_df = pd.concat([train, test])



# In[5]:


#모든 열 라벨인코딩
cat_cols = ["ID", "User-ID", "Book-ID", "Age", "Location", "Book-Title", "Year-Of-Publication", "Book-Author", "Publisher"]


for col in cat_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))



# MinMaxScaler
scaler = MinMaxScaler()
combined_df[cat_cols] = scaler.fit_transform(combined_df[cat_cols])



# train 데이터프레임 분리
train_le = combined_df.iloc[:len(train), :]

# test 데이터프레임 분리
test_le = combined_df.iloc[len(train):, :]



# 특성 열 정의
sparse_features = cat_cols
target = ['new_Rating']

# 데이터 전처리 및 특성 열 생성
train_model_input = {name: train_le[name] for name in sparse_features}
test_model_input = {name: test_le[name] for name in sparse_features}

# 모델 정의
feature_columns = [SparseFeat(feat, vocabulary_size=combined_df[feat].nunique(), embedding_dim=4) for feat in cat_cols]
feature_names = get_feature_names(feature_columns)

inputs = {name: Input(shape=(1,), name=name) for name in sparse_features}
sparse_inputs = [inputs[name] for name in sparse_features]

# 임베딩 레이어 초기화
embedding_layers = {name: Embedding(input_dim=combined_df[name].nunique(), output_dim=4)(input_layer) for name, input_layer in inputs.items()}

# 모든 임베딩 레이어를 연결(concatenate)
concat_embeddings = Concatenate(axis=1)(list(embedding_layers.values()))

# 모델 생성
output_layer = Dense(1, activation='relu')(concat_embeddings)
model = Model(inputs=list(inputs.values()), outputs=output_layer)

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# 모델 학습
history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2)

# 모델 평가
#pred_ans = model.predict(test_model_input, batch_size=256)
#mse = mean_squared_error(test[target].values, pred_ans)
#print("RMSE:", np.sqrt(mse))


# 

# 
