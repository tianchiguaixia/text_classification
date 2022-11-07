# -*- coding: utf-8 -*-
# time: 2022/10/8 14:01
# file: bert_model.py


'''
该脚本定义了一个BERT模型，用于分类.
主要内容：编码器，自定义数据集，训练和验证函数
'''

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer,BertConfig, AdamW, get_linear_schedule_with_warmup



class EnterpriseDataset(Dataset):
    '''自定义数据集'''
    def __init__(self, texts, labels, tokenizer, max_len):
        '''
        :param texts: 文本内容
        :param labels: 文本标签
        :param tokenizer: 分词器
        :param max_len: 文本最大长度
        '''
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    '''
    :param df: 数据集（text，label两列
    :param tokenizer: 分词器
    :param max_len: 句子最大长度
    :param batch_size: 批次大小
    :return: 返回dataloader数据格式
    '''
    ds = EnterpriseDataset(
        texts=df['text'].values,
        labels=df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        #         num_workers=4 # windows多线程
    )

class EnterpriseDangerClassifier(nn.Module):
    '''
    分类器
    '''
    def __init__(self, PRE_TRAINED_MODEL_NAME,n_classes):
        '''
        :param PRE_TRAINED_MODEL_NAME: 模型名称
        :param n_classes: 标签类别
        '''
        super(EnterpriseDangerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)  # 两个类别

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)  # dropout
        return self.out(output)


def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
    '''
    :param model: 模型
    :param data_loader: 数据加载器
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 默认gpu
    :param scheduler: 学习率调整器
    :param n_examples: 样本数
    :return: 返回损失和准确率
    '''
    model = model.train() # train模式
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    '''
    :param model: 模型
    :param data_loader: 数据加载器
    :param loss_fn: 损失函数
    :param device: 默认gpu
    :param n_examples: 样本数
    :return: 返回损失和准确率
    '''
    model = model.eval() # 验证预测模式
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)