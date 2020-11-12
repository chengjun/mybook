#!/usr/bin/env python
# coding: utf-8

# # 使用requests + Xpath抓取豆瓣电影数据
# 

# Xpath 即为 XML 路径语言（XML Path Language），它是一种用来确定 XML 文档中某部分位置的语言。
# 
# Xpath 基于 XML 的树状结构，提供在数据结构树中找寻节点的能力。起初 Xpath 的提出的初衷是将其作为一个通用的、介于 Xpointer 与 XSL 间的语法模型。但是Xpath 很快的被开发者采用来当作小型查询语言。
# 
# 

# 获取元素的Xpath信息并获得文本：
# 这里的“元素的Xpath信息”是需要我们手动获取的，获取方式为：
# - 定位目标元素
# - 在网站上依次点击：右键 > 检查
# - copy xpath
# - xpath + '/text()'
# 
# 参考：https://mp.weixin.qq.com/s/zx3_eflBCrrfOqFEWjAUJw
# 

# In[2]:


import requests
from lxml import etree

url = 'https://movie.douban.com/subject/26611804/'
requests.get(url)


# 如果不加headers，响应状态：418， 正常返回状态应该是 200
# 
# - 418啥意思？就是你爬取的网站有反爬虫机制，我们要向服务器发出爬虫请求，需要添加请求头：headers
# - 如何加请求头headers?
#     - 网页右键“检查元素”-Network-Doc 如上图

# ![image.png](images/headers.png)

# In[3]:


import requests
from lxml import etree

url = 'https://movie.douban.com/subject/26611804/'

headers ={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36'}

data = requests.get(url, headers = headers).text
s = etree.HTML(data)  


# 豆瓣电影的名称对应的的xpath为xpath_title，那么title表达为：
# 
# `title = s.xpath('xpath_info/text()')`
# 
# 其中，xpath_info为：
# 
# `//*[@id="content"]/h1/span[1]`
# 
# 

# In[4]:


title = s.xpath('//*[@id="content"]/h1/span[1]/text()')[0]
director = s.xpath('//*[@id="info"]/span[1]/span[2]/a/text()')
actors = s.xpath('//*[@id="info"]/span[3]/span[2]/a/text()')
type1 = s.xpath('//*[@id="info"]/span[5]/text()')
type2 = s.xpath('//*[@id="info"]/span[6]/text()')
type3 = s.xpath('//*[@id="info"]/span[7]/text()')
time = s.xpath('//*[@id="info"]/span[11]/text()')
length = s.xpath('//*[@id="info"]/span[13]/text()')
score = s.xpath('//*[@id="interest_sectl"]/div[1]/div[2]/strong/text()')[0]


# In[5]:


print(title, director, actors, type1, type2, type3, time, length, score)


# ## Douban API
# 
# https://developers.douban.com/wiki/?title=guide
# 
# https://github.com/computational-class/douban-api-docs

# In[6]:


import requests
# https://movie.douban.com/subject/26611804/
url = 'https://api.douban.com/v2/movie/subject/26611804?apikey=0b2bdeda43b5688921839c8ecb20399b&start=0&count=20&client=&udid='
jsonm = requests.get(url).json()


# In[7]:


jsonm.keys()


# In[8]:


jsonm['msg']


# In[3]:


#jsonm.values()
jsonm['rating']


# In[4]:


jsonm['alt']


# In[21]:


jsonm['casts'][0]


# In[10]:


jsonm['directors']


# In[13]:


jsonm['genres']


# ## 作业：抓取豆瓣电影 Top 250

# In[9]:


import requests
from bs4 import BeautifulSoup
from lxml import etree

url0 = 'https://movie.douban.com/top250?start=0&filter='
data = requests.get(url0, headers=headers).text
s = etree.HTML(data)


# In[ ]:


(/*[@id="content"]/div/div[1]/ol/li[1]/div/div[2]/div[1]/a/span[1])

html(/body/div[3]/div[1]/div/div[1]/ol/li[1]/div/div[2]/div[1]/a/span[1])


# In[13]:


str1 = '//*[@id="content"]/div/div[1]/ol/li['
str2 = ']/div/div[2]/div[1]/a/span[1]/text()'

xstr_list = [str1 + str(i+1) +str2 for i in range(25)]
[s.xpath(i)[0]  for i in xstr_list]


# In[11]:


s.xpath('//*[@id="content"]/div/div[1]/ol/li[1]/div/div[2]/div[1]/a/span[1]/text()')[0]


# In[57]:


s.xpath('//*[@id="content"]/div/div[1]/ol/li[2]/div/div[2]/div[1]/a/span[1]/text()')[0]


# In[227]:


s.xpath('//*[@id="content"]/div/div[1]/ol/li[3]/div/div[2]/div[1]/a/span[1]/text()')[0]


# In[14]:


import requests
from bs4 import BeautifulSoup

url0 = 'https://movie.douban.com/top250?start=0&filter='
data = requests.get(url0, headers = headers).text
soup = BeautifulSoup(data, 'lxml')


# In[15]:


movies = soup.find_all('div', {'class', 'info'})


# In[16]:


len(movies)


# In[19]:


movies[0].a['href']


# In[23]:


movies[0].find('span', {'class', 'title'}).text


# In[25]:


movies[0].find('div', {'class', 'star'})


# In[26]:


movies[0].find('span', {'class', 'rating_num'}).text


# In[27]:


people_num = movies[0].find('div', {'class', 'star'}).find_all('span')[-1]
people_num.text.split('人评价')[0]


# In[28]:


for i in movies:
    url = i.a['href']
    title = i.find('span', {'class', 'title'}).text
    des = i.find('div', {'class', 'star'})
    rating = des.find('span', {'class', 'rating_num'}).text
    rating_num = des.find_all('span')[-1].text.split('人评价')[0]
    print(url, title, rating, rating_num)


# In[29]:


for i in range(0, 250, 25):
    print('https://movie.douban.com/top250?start=%d&filter='% i)


# In[30]:


import requests
from bs4 import BeautifulSoup
dat = []
for j in range(0, 250, 25):
    urli = 'https://movie.douban.com/top250?start=%d&filter='% j
    print(urli)
    data = requests.get(urli, headers = headers).text
    soup = BeautifulSoup(data, 'lxml')
    movies = soup.find_all('div', {'class', 'info'})
    for i in movies:
        url = i.a['href']
        title = i.find('span', {'class', 'title'}).text
        des = i.find('div', {'class', 'star'})
        rating = des.find('span', {'class', 'rating_num'}).text
        rating_num = des.find_all('span')[-1].text.split('人评价')[0]
        listi = [url, title, rating, rating_num]
        dat.append(listi)


# In[31]:


import pandas as pd
df = pd.DataFrame(dat, columns = ['url', 'title', 'rating', 'rating_num'])
df['rating'] = df.rating.astype(float)
df['rating_num'] = df.rating_num.astype(int)
df.head()


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.hist(df.rating_num)
plt.show()


# In[19]:


plt.hist(df.rating)
plt.show()


# In[11]:


# viz
fig = plt.figure(figsize=(16, 16),facecolor='white')
plt.plot(df.rating_num, df.rating, 'bo')
for i in df.index:
    plt.text(df.rating_num[i], df.rating[i], df.title[i], 
             fontsize = df.rating[i], 
             color = 'red', rotation = 45)
plt.show() 


# In[123]:


df[df.rating > 9.4]


# In[69]:


alist = []
for i in df.index:
    alist.append( [df.rating_num[i], df.rating[i], df.title[i] ])

blist =[[df.rating_num[i], df.rating[i], df.title[i] ] for i in df.index] 

alist


# In[8]:


# from IPython.display import display_html, HTML
# HTML('<iframe src=http://nbviewer.jupyter.org/github/computational-class/bigdata/blob/gh-pages/vis/douban250bubble.html \
#      width=1000 height=500></iframe>')


# ## 作业：
# 
# - 抓取复旦新媒体微信公众号最新一期的内容
# 

# ## requests.post模拟登录豆瓣（包括获取验证码）
# https://blog.csdn.net/zhuzuwei/article/details/80875538
