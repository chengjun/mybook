#!/usr/bin/env python
# coding: utf-8

# # 抓取网络小说
# 
# 《1/13密室杀人》
# 
# 
# - 作者: 鸡丁
# - 副标题: 鸡丁密室推理短篇集
# - 出版年: 2013-9
#     
# 鸡丁，生于上海，因喜爱吃宫保鸡丁而取此笔名，80后推理作者，天蝎座，较宅。高中时代起迷恋推理小说，钟爱江户川乱步和约翰·狄克森·卡尔。2008年开始撰写短篇推理小说和谜题，崇尚本格，热衷于密室和不可能犯罪题材，在《岁月·推理》和《推理世界》上发表《斩首缆车》《神的密室》《憎恶之锤》《雪祭》等多部密室佳作。系《推理世界》签约作者。 

# In[1]:


import requests
from bs4 import BeautifulSoup


# In[2]:


url = 'https://www.bixiadu.com/bxd-3501/'
page = requests.get(url)
page.encoding = 'utf-8'
soup = BeautifulSoup(page.text, 'html.parser') 


# In[3]:


urls = {i.a['href'] for i in soup.find_all('dd')}
urls = sorted(list(urls))
urls = [url+i for i in list(urls)]


# In[4]:


urls


# In[5]:


for k, i in enumerate(urls):
    print(k, i)
    page = requests.get(i)
    page.encoding = 'utf-8'
    soup = BeautifulSoup(page.text, 'html.parser') 
    title = soup.select(".bookname")[0]('h1')[0].text
    body = soup.select('#content')[0].text
    body = body.replace('\u3000\u3000\xa0\xa0\xa0\xa0', '\n')
    story = title + '\n' + body
    with open('../data/13chapters.txt', 'a') as f:
        f.write(story)

