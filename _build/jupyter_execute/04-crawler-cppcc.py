#!/usr/bin/env python
# coding: utf-8

# # 抓取江苏省政协十年提案

# 打开http://www.jszx.gov.cn/zxta/2019ta/
# 
# - 点击下一页，url不变!
# 
# > 所以数据的更新是使用js推送的
# - 分析network中的内容，发现proposalList.jsp
#     - 查看它的header，并发现了form_data
#     
# <img src = './img/form_data.png'>
# 
# http://www.jszx.gov.cn/zxta/2019ta/

# In[1]:


import requests
from bs4 import BeautifulSoup


# In[2]:


form_data = {'year':2019,
        'pagenum':1,
        'pagesize':20
}
url = 'http://www.jszx.gov.cn/wcm/zxweb/proposalList.jsp'
content = requests.get(url, form_data)
content.encoding = 'utf-8'
js = content.json()


# In[3]:


js['data']['totalcount']


# In[4]:


dat = js['data']['list']
pagenum = js['data']['pagecount']


# In[5]:


for i in range(2, pagenum+1):
    print(i)
    form_data['pagenum'] = i
    content = requests.get(url, form_data)
    content.encoding = 'utf-8'
    js = content.json()
    for j in js['data']['list']:
        dat.append(j)


# In[6]:


len(dat)


# In[7]:


dat[0]


# In[8]:


import pandas as pd

df = pd.DataFrame(dat)
df.head()


# In[9]:


df.groupby('type').size()


# ## 抓取提案内容
# http://www.jszx.gov.cn/zxta/2019ta/index_61.html?pkid=18b1b347f9e34badb8934c2acec80e9e

# In[10]:


url_base = 'http://www.jszx.gov.cn/wcm/zxweb/proposalInfo.jsp?pkid='
urls = [url_base + i  for i in df['pkid']]


# In[11]:


import sys
def flushPrint(www):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % www)
    sys.stdout.flush()
    
text = []
for k, i in enumerate(urls):
    flushPrint(k)
    content = requests.get(i)
    content.encoding = 'utf-8'
    js = content.json()
    js = js['data']['binfo']['_content']
    soup = BeautifulSoup(js, 'html.parser') 
    text.append(soup.text)


# In[12]:


len(text)


# In[13]:


df['content'] = text


# In[14]:


df.head()


# In[15]:


#df.to_csv('../data/jszx2019.csv', index = False)

