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

# In[3]:


import requests
from bs4 import BeautifulSoup

path = 'https://vp.fact.qq.com/loadmore?artnum=0&token=U2FsdGVkX18vf24Ll8I4bLXafIv7BY3GfCU0q9PJ%252FhM2kTmyPKf78WYKL5KP%252BGx4&page='
url = path + '0'
content = requests.get(url)
d = content.json()


# In[4]:


d['content'][0]


# In[6]:


print(*range(74))


# In[7]:


import random

random.random()


# In[8]:


from time import sleep
import random

jsons = []
for i in range(74):
    print(i)
    sleep(random.random())
    path = 'https://vp.fact.qq.com/loadmore?artnum=0&token=U2FsdGVkX18vf24Ll8I4bLXafIv7BY3GfCU0q9PJ%252FhM2kTmyPKf78WYKL5KP%252BGx4&page='
    url = path + str(i)
    content = requests.get(url)
    d = content.json()
    for j in d['content']:
        jsons.append(j)


# In[9]:


len(jsons)


# In[10]:


import pandas as pd
df = pd.DataFrame(jsons)
df.head()


# In[11]:


df.to_excel('./data/vpqq2021-05-15.xlsx')


# In[ ]:




