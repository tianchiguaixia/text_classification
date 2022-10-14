# -*- coding: utf-8 -*-
# time: 2022/10/14 15:43
# file: show_demo.py

import pandas as pd
import streamlit as st
import requests

st.title("AI-文本分类")

uploaded_file = st.file_uploader("选择一个文件")
if uploaded_file is not None:
    print(uploaded_file)
    data = pd.read_excel(uploaded_file)
    st.success("表格内容如下：")
    st.write(data)

with st.container():
    with st.form(key="my_form"):
        content=st.text_input(label="请输入终点名称")
        json_data={
        "content":content
        }
        parse_json=requests.post("http://你的ip:10713/news/predict_label",json=json_data)
        text=eval(parse_json.text)
        submit_button = st.form_submit_button(label="✨ 启动!")



if not submit_button:
    st.stop()

with st.container():
    st.write("---")
    with st.expander(label="json结果展示",expanded=False):
        st.write(text)
    st.write("**{}**：{}".format(text['label'],text['probs']))
