#!/usr/bin/env python
# coding: utf-8

# 
# # 使用Selenium提取人民网搜索数据
# 
# 

# ![image.png](./images/people.png)
# 
# http://search.people.com.cn/cnpeople/news/getNewsResult.jsp

# ![image.png](./images/people2.png)
# 
# 
# 点击下一页页面的URL不变

# ![image.png](./images/people3.png)
# 
# 
# 鼠标右键查看页码

# In[12]:


url = 'http://search.people.com.cn/cnpeople/search.do?pageNum='
path = '&keyword=%D0%C2%B9%DA+%D6%D0%D2%BD&siteName=news&facetFlag=null&nodeType=belongsId&nodeId='
page_num = range(1, 30)
urls = [url+str(i)+path for i in page_num]
for i in urls[-3:]:
    print(i)
    


# ## 无法通过requests直接获取
# 
# 提醒：您的访问可能对网站造成危险，已被云防护安全拦截
# 
# ```
# import requests
# from bs4 import BeautifulSoup
# 
# content = requests.get(urls[0])
# content.encoding = 'utf-8'
# 
# soup = BeautifulSoup(content.text, 'html.parser') 
# soup
# ```

# In[ ]:


import requests
from bs4 import BeautifulSoup

content = requests.get(urls[0])
content.encoding = 'utf-8'
soup = BeautifulSoup(content.text, 'html.parser') 
soup


# In[13]:


from selenium import webdriver
from bs4 import BeautifulSoup
import time

browser = webdriver.Chrome()
dat = []
for k, j in enumerate(urls):
    print(k+1)
    time.sleep(1)
    browser.get(j) 
    source = browser.page_source
    soup = BeautifulSoup(source, 'html.parser') 
    d = soup.find_all('ul')
    while len(d) < 2:
        print(k+1, 'null error and retry')
        time.sleep(1)
        browser.get(j) 
        source = browser.page_source
        soup = BeautifulSoup(source, 'html.parser') 
        d = soup.find_all('ul')
        
    for i in d[1:]:
        urli = i.find('a')['href']
        title = i.find('a').text
        time_stamp = i.find_all('li')[-1].text.split('\xa0')[-1]
        dat.append([k+1, urli, title, time_stamp])

browser.close()
len(dat)


# In[14]:


import pandas as pd
df = pd.DataFrame(dat, columns = ['pagenum', 'url', 'title', 'time'])
df.head()


# In[16]:


len(df)


# In[17]:


df.to_csv('../data/people_com_search20200606.csv', index = False)


# ## Reading data with Pandas

# In[20]:


with open('../data/people_com_search20200606.csv', 'r') as f:
    lines = f.readlines()
len(lines)


# In[21]:


import pandas as pd
df2 = pd.read_csv('../data/people_com_search20200606.csv')
df2.head()
len(df2)


# In[24]:


for i in df2['url'].tolist()[:10]:
    print(i)


# In[ ]:




