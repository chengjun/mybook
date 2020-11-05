#!/usr/bin/env python
# coding: utf-8

# 
# # 抓取微信公众号文章内容
# 
# 

# In[11]:


from IPython.display import display_html, HTML
HTML(url = 'http://mp.weixin.qq.com/s?__biz=MzA3MjQ5MTE3OA==&mid=206241627&idx=1&sn=471e59c6cf7c8dae452245dbea22c8f3&3rd=MzA3MDU4NTYzMw==&scene=6#rd')
# the webpage we would like to crawl


# ## 查看源代码 Inspect

# In[12]:


url = "http://mp.weixin.qq.com/s?__biz=MzA3MjQ5MTE3OA==&mid=206241627&idx=1&sn=471e59c6cf7c8dae452245dbea22c8f3&3rd=MzA3MDU4NTYzMw==&scene=6#rd"
content = requests.get(url).text #获取网页的html文本
soup = BeautifulSoup(content, 'html.parser') 


# In[17]:


title = soup.select("#activity-name") # #activity-name
title[0].text.strip()


# In[18]:


soup.find('h2', {'class', 'rich_media_title'}).text.strip()


# In[25]:


print(soup.find('div', {'class', 'rich_media_meta_list'}) )


# In[26]:


soup.select('#publish_time')


# In[27]:


article = soup.find('div', {'class' , 'rich_media_content'}).text
print(article)


# In[30]:


rmml = soup.find('div', {'class', 'rich_media_meta_list'})
#date = rmml.find(id = 'post-date').text
rmc = soup.find('div', {'class', 'rich_media_content'})
content = rmc.get_text()
print(title[0].text.strip())
#print(date)
print(content) 


# ## wechatsogou 
# 
# > pip install wechatsogou --upgrade
# 
# 
# https://github.com/Chyroc/WechatSogou

# In[15]:


get_ipython().system('pip install wechatsogou --upgrade')


# In[16]:


import wechatsogou

# 可配置参数

# 直连
ws_api = wechatsogou.WechatSogouAPI()

# 验证码输入错误的重试次数，默认为1
ws_api = wechatsogou.WechatSogouAPI(captcha_break_time=3)

# 所有requests库的参数都能在这用
# 如 配置代理，代理列表中至少需包含1个 HTTPS 协议的代理, 并确保代理可用
ws_api = wechatsogou.WechatSogouAPI(proxies={
    "http": "127.0.0.1:8889",
    "https": "127.0.0.1:8889",
})

# 如 设置超时
ws_api = wechatsogou.WechatSogouAPI(timeout=0.1)


# In[17]:


ws_api =wechatsogou.WechatSogouAPI()
ws_api.get_gzh_info('南航青年志愿者')


# In[19]:


articles = ws_api.search_article('南京航空航天大学')


# In[20]:


for i in articles:
    print(i)

