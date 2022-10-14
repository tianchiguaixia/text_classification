# -*- coding: utf-8 -*-
# time: 2022/10/8 15:39
# file: model_deploy.py


'''
该脚本主要部署分类模型，将模型部署到服务器上，然后通过服务器提供的接口进行调用
'''

import traceback
import logging
from transformers import BertModel, BertTokenizer, BertConfig
import json
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import joblib
import yaml
from flask import Flask, request, render_template, jsonify, session
from src.bert_model import EnterpriseDataset,create_data_loader,train_epoch,eval_model,EnterpriseDangerClassifier
import pandas as pd
import numpy as np
app = Flask(__name__)


#优先使用gpu，没有则使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#读取配置文件
with open("config/config.yaml", 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
PRE_TRAINED_MODEL_NAME= config['cnews']['PRE_TRAINED_MODEL_NAME']
label2onehot= config['cnews']['label2onehot']
class_names_num = config['cnews']['class_names_num']
MAX_LEN = config['cnews']['max_len']
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model_path = config['cnews']['model_save_path']

#模型加载
model = EnterpriseDangerClassifier(PRE_TRAINED_MODEL_NAME,class_names_num)
model = torch.load(model_path)
model = model.eval()
model = model.to(device)

#标签映射加载
le = joblib.load(label2onehot)

#日志配置
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s : %(name)s : %(levelname)s :%(filename)s : [%(lineno)d]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='app.log',
    filemode='a')

#单标签预测
def predict_label(sample_text):
    encoded_text = tokenizer.encode_plus(
        sample_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    probs = float(torch.max(F.softmax(output, dim=1)).cpu().detach().numpy())
    label=le.inverse_transform(prediction.cpu().numpy())[0]
    result = [label, probs]
    return result


#多标签预测
def multi_predict_label(sample_text):
    encoded_text = tokenizer.encode_plus(
        sample_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    prediction_value = pd.Series(output[0].cpu().detach().numpy().tolist()).sort_values(
        ascending=False).values.tolist()[:3]
    index = pd.Series(output[0].cpu().detach().numpy().tolist()).sort_values(ascending=False).index[:3]
    probs = F.softmax(output, dim=1).cpu().detach().numpy()[0]
    result = le.inverse_transform(prediction.cpu().numpy())[0]

    label_list = []
    prob_list = []
    for i in index.tolist():
        m = np.array([i])
        label_list.append(le.inverse_transform(m)[0])
        prob_list.append(float(probs[i]))
    result = [label_list, prob_list]
    return result

@app.route("/news/predict_label", methods=['POST'])
def hepatopathy_predict_label():
    try:
        comment = request.get_data()
        json_data = json.loads(comment.decode())
        content = json_data["content"]
        result = predict_label(content)
        result = {"text": content, "label": result[0], "probs": result[1],"code":200}
        return jsonify(result)
    except Exception as e:
        logging.error("错误信息：" + str(e))
        logging.error("\n" + traceback.format_exc())
        return jsonify({"error":traceback.format_exc(),"code":500})

@app.route("/news/multi_predict_label", methods=['POST'])
def hepatopathy_multi_predict_label():
    try:
        comment = request.get_data()
        json_data = json.loads(comment.decode())
        content = json_data["content"]
        result = multi_predict_label(content)
        print(result)
        result = {"text": content, "label": result[0], "probs": result[1],"code":200}
        return jsonify(result)
    except Exception as e:
        logging.error("错误信息：" + str(e))
        logging.error("\n" + traceback.format_exc())
        return jsonify({"error":traceback.format_exc(),"code":500})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10713, debug=True)
