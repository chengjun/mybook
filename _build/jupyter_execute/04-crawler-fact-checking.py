#!/usr/bin/env python
# coding: utf-8

# # 抓取实时辟谣数据
# 
# 
# https://vp.fact.qq.com/home

# 
# 
# ![image.png](./images/fact.png)
# 
# https://vp.fact.qq.com/article?id=be3aea585b07c193778985e180cf164b
# 
# 

# https://vp.fact.qq.com/loadmore?artnum=0&page=0
# 
# ![image.png](./images/fact2.png)
# 

# https://vp.fact.qq.com/loadmore?artnum=0&page=0
# 
# ![image.png](./images/fact3.png)

# In[1]:


import requests
from bs4 import BeautifulSoup

path = 'https://vp.fact.qq.com/loadmore?artnum=0&page='
url = path + '0'
content = requests.get(url)
d = content.json()


# In[9]:


d['content'][0]


# In[11]:


print(*range(61))


# In[13]:


import random

random.random()


# In[14]:


from time import sleep
import random

jsons = []
for i in range(61):
    print(i)
    sleep(random.random())
    path = 'https://vp.fact.qq.com/loadmore?artnum=0&page='
    url = path + str(i)
    content = requests.get(url)
    d = content.json()
    for j in d['content']:
        jsons.append(j)


# In[15]:


len(jsons)


# In[16]:


import pandas as pd
df = pd.DataFrame(jsons)
df.head()


# In[17]:


df.to_excel('../data/vpqq2020-06-06.xlsx')

