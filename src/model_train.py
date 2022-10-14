# -*- coding: utf-8 -*-
# time: 2022/10/8 10:51
# file: model_train.py

'''
该脚本是用于训练模型的脚本
主要内容：标签转换，模型训练和评估
'''

# 导入transformers
import transformers
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer,AutoConfig, AdamW, get_linear_schedule_with_warmup

# 导入torch
import torch
from torch import nn, optim
import torch.nn.functional as F

# 常用包
import re
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import yaml
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import joblib

#警告忽略
import warnings
warnings.filterwarnings("ignore")

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#自定义内容
from src.bert_model import EnterpriseDataset,create_data_loader,train_epoch,eval_model,EnterpriseDangerClassifier

#优先使用gpu，没有则使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#读取配置文件
with open("config/config.yaml", 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

PRE_TRAINED_MODEL_NAME= config['cnews']['PRE_TRAINED_MODEL_NAME']
train_data_path = config['cnews']['train_data_path']
label2onehot= config['cnews']['label2onehot']
model_save_path= config['cnews']['model_save_path']
max_len = config['cnews']['max_len']

#自定义参数
test_size=0.3
#max_len = 230
batch_size = 30
epochs=10

#读取数据（确保只有一列’text'，一列'y',数据为xlsx格式）
df_raw =pd.read_table(train_data_path,header=None)
df_raw.columns=['y','text']
le=LabelEncoder()
df_raw['y']=le.fit_transform(df_raw['y'])

if not os.path.exists('labelencode/'):
    os.makedirs('labelencode/')
joblib.dump(le,label2onehot)

#确定类别
class_names=list(set(df_raw['y']))

#划分训练集和测试集
if 'text_len' in df_raw.columns:
    del df_raw['text_len']
df_raw.columns=['label','text']
df_train, df_val =  train_test_split(df_raw, test_size=test_size, stratify=df_raw['label'],random_state=42,shuffle=True)

#建立分词器
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#对句子进行分词，添加特殊符号
sample_txt="ALT"
encoding=tokenizer.encode_plus(
        sample_txt,
        # sample_txt_another,
        max_length=max_len,
        add_special_tokens=True,# [CLS]和[SEP]
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',# Pytorch tensor张量
    )

#创建数据集
train_data_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
val_data_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)

#模型配置
bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
last_hidden_state, pooled_output = bert_model(
    input_ids=encoding['input_ids'],
    attention_mask=encoding['attention_mask'],
    return_dict = False
)


#模型初始化
model = EnterpriseDangerClassifier(PRE_TRAINED_MODEL_NAME,len(class_names))
model = model.to(device)

#优化器设置
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

#训练模型
history = defaultdict(list) # 记录10轮loss和acc
best_accuracy = 0

#训练过程
if not os.path.exists('best_models/'):
    os.makedirs('best_models/')
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')
    torch.save(model, model_save_path)

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        #torch.save(model.state_dict(), 'best_models/best_bioclinicalbert_model_state.bin')
        torch.save(model, model_save_path)
        best_accuracy = val_acc
