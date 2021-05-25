#!/usr/bin/env python
# coding: utf-8

# 
# 
# # 第三章 数据抓取
# 
# Requests和Beautifulsoup简介
# 
# 
# ![image.png](./images/author.png)
# 

# ## 基本原理
# 
# 爬虫就是请求网站并提取数据的自动化程序。其中请求，提取，自动化是爬虫的关键！爬虫的基本流程：
# 
# - 发起请求
#     - 通过HTTP库向目标站点发起请求，也就是发送一个Request，请求可以包含额外的header等信息，等待服务器响应
# 
# - 获取响应内容
#     - 如果服务器能正常响应，会得到一个Response。Response的内容便是所要获取的页面内容，类型可能是HTML、Json字符串、二进制数据（图片或者视频）等类型
# 
# 
# 

# - 解析内容
#     - 得到的内容可能是HTML,可以用页面解析库、正则表达式进行解析；可能是Json,可以直接转换为Json对象解析；可能是二进制数据，可以做保存或者进一步的处理
# 
# - 保存数据
#     - 保存形式多样，可以存为文本，也可以保存到数据库，或者保存特定格式的文件
# 
# 浏览器发送消息给网址所在的服务器，这个过程就叫做**Http Request**;服务器收到浏览器发送的消息后，能够根据浏览器发送消息的内容，做相应的处理，然后把消息回传给浏览器，这个过程就是**Http Response**.

# ## 需要解决的问题 
# 
# - 页面解析
# - 获取Javascript隐藏源数据
# - 自动翻页
# - 自动登录
# - 连接API接口
# 

# 一般的数据抓取，使用requests和beautifulsoup配合就可以了。
# - 尤其是对于翻页时url出现规则变化的网页，只需要处理规则化的url就可以了。
# - 以简单的例子是抓取天涯论坛上关于某一个关键词的帖子。
#     - 在天涯论坛，关于雾霾的帖子的第一页是：
# http://bbs.tianya.cn/list.jsp?item=free&nextid=0&order=8&k=雾霾
#     - 第二页是：
# http://bbs.tianya.cn/list.jsp?item=free&nextid=1&order=8&k=雾霾
# 

# ## 第一个爬虫
# 
# ![](images/alice.png)
# 
# Beautifulsoup Quick Start 
# 
# http://www.crummy.com/software/BeautifulSoup/bs4/doc/
# 
# 
# http://computational-class.github.io/bigdata/data/test.html
# 

# 
# 'Once upon a time there were three little sisters,' the Dormouse began in a great hurry; 'and their names were Elsie, Lacie, and Tillie; and they lived at the bottom of a well--'
# 
# <img src="./images/alice2.png" align='right'>
# 
# 'What did they live on?' said Alice, who always took a great interest in questions of eating and drinking.
# 
# 'They lived on treacle,' said the Dormouse, after thinking a minute or two.
# 
# 'They couldn't have done that, you know,' Alice gently remarked; 'they'd have been ill.'
# 
# 'So they were,' said the Dormouse; 'very ill.'
# 
# **Alice's Adventures in Wonderland** CHAPTER VII A Mad Tea-Party http://www.gutenberg.org/files/928/928-h/928-h.htm

# In[1]:


import requests
from bs4 import BeautifulSoup 


# ```
# import requests
# from bs4 import BeautifulSoup 
# 
# url = 'https://vp.fact.qq.com/home'
# content = requests.get(url)
# soup = BeautifulSoup(content.text, 'html.parser') 
# 
# ```
# 

# In[77]:


help(requests.get)


# In[2]:


url = 'https://socratesacademy.github.io/bigdata/data/test.html'
content = requests.get(url)
#help(content)


# In[3]:


print(content.text)


# In[4]:


content.encoding


# ## Beautiful Soup
# > Beautiful Soup is a Python library designed for quick turnaround projects like screen-scraping. Three features make it powerful:
# 
# - Beautiful Soup provides a few simple methods. It doesn't take much code to write an application
# - Beautiful Soup automatically converts incoming documents to Unicode and outgoing documents to UTF-8. Then you just have to specify the original encoding.
# - Beautiful Soup sits on top of popular Python parsers like `lxml` and `html5lib`.
# 

# ### Install beautifulsoup4
# 
# open your terminal/cmd
# 
# <del> $ pip install beautifulsoup4

# ### html.parser
# Beautiful Soup supports the html.parser included in Python’s standard library
# 
# ### lxml
# but it also supports a number of third-party Python parsers. One is the lxml parser `lxml`. Depending on your setup, you might install lxml with one of these commands:
# 
# > $ apt-get install python-lxml
# 
# > $ easy_install lxml
# 
# > $ pip install lxml

# ### html5lib
# Another alternative is the pure-Python html5lib parser `html5lib`, which parses HTML the way a web browser does. Depending on your setup, you might install html5lib with one of these commands:
# 
# > $ apt-get install python-html5lib
# 
# > $ easy_install html5lib
# 
# > $ pip install html5lib

# In[6]:


url = 'http://socratesacademy.github.io/bigdata/data/test.html'
content = requests.get(url)
content = content.text
soup = BeautifulSoup(content, 'html.parser') 
soup


# In[7]:


print(soup.prettify())


# - html
#     - head
#         - title
#     - body
#         - p (class = 'title', 'story' )
#             - a (class = 'sister')
#                 - href/id

# ## Select 方法
# 
# 
# - 标签名不加任何修饰
# - 类名前加点
# - id名前加 #
# 
# 我们也可以利用这种特性，使用soup.select()方法筛选元素，返回类型是 list

# Select方法三步骤
# 
# - Inspect (检查)
# - Copy
#     - Copy Selector
#     

# - 鼠标选中标题`The Dormouse's story`, 右键检查Inspect
# - 鼠标移动到选中的源代码
# - 右键Copy-->Copy Selector 
# 
# `body > p.title > b`
# 

# In[10]:


soup.select('body > p.title > b')[0].text


# ### Select 方法: 通过标签名查找

# In[11]:


soup.select('title')[0].text


# In[12]:


soup.select('a')


# In[13]:


soup.select('b')


# ### Select 方法: 通过类名查找

# In[14]:


soup.select('.title')


# In[15]:


soup.select('.sister')


# In[16]:


soup.select('.story')


# ### Select 方法: 通过id名查找

# In[17]:


soup.select('#link1')


# In[19]:


soup.select('#link1')[0]['href']


# ### Select 方法: 组合查找
# 
# 将标签名、类名、id名进行组合
# 
# - 例如查找 p 标签中，id 等于 link1的内容
#  

# In[20]:


soup.select('p #link1')


# ### Select 方法:属性查找
# 
# 加入属性元素
# - 属性需要用大于号`>`连接
# - 属性和标签属于同一节点，中间不能加空格。
#  
# 
# 

# In[21]:


soup.select("head > title")


# In[22]:


soup.select("body > p")


# ## find_all方法

# In[23]:


#soup('p')
soup.find_all('p')


# In[29]:


soup.find_all('p')  


# In[24]:


[i.text for i in soup('p')]


# In[25]:


for i in soup('p'):
    print(i.text)


# In[26]:


for tag in soup.find_all(True):
    print(tag.name)


# In[27]:


soup('head') # or soup.head


# In[28]:


soup('body') # or soup.body


# In[29]:


soup('title')  # or  soup.title


# In[30]:


soup('p')


# In[40]:


soup.p


# In[33]:


soup.title.name


# In[34]:


soup.title.string


# In[36]:


soup.title.text
# 推荐使用text方法


# In[38]:


soup.title.parent.name


# In[37]:


soup.p


# In[38]:


soup.p['class']


# In[42]:


soup.find_all('p', {'class', 'story'}) 


# In[43]:


#soup.find_all('p', class_= 'title')


# In[44]:


soup.find_all('a', {'class', 'sister'})


# In[34]:


soup.find_all('p', {'class', 'story'})[0].find_all('a')


# In[45]:


soup.a


# In[46]:


soup('a')


# In[48]:


soup.find(id="link1")


# In[49]:


soup.find_all('a')


# In[50]:


soup.find_all('a', {'class', 'sister'}) # compare with soup.find_all('a')


# In[51]:


soup.find_all('a', {'class', 'sister'})[0]


# In[57]:


soup.find_all('a', {'class', 'sister'})[0].text 


# In[58]:


soup.find_all('a', {'class', 'sister'})[0]['href']


# In[59]:


soup.find_all('a', {'class', 'sister'})[0]['id']


# In[74]:


soup.find_all(["a", "b"])


# In[75]:


print(soup.get_text())


# ![image.png](./images/end.png)

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# 04-crawler-fact-checking
# 04-crawler-13chambers
# 04-crawler-wechat
# 04-crawler-douban
# 04-crawler-gov-report
# 04-crawler-cppcc
# 04-crawler-netease-music
# 04-crawler-selenium
# 04-crawler-selenium-music-history
# 04-crawler-selenium-people-com-search
# 04-crawler-tripadvisor
# ```
# 
