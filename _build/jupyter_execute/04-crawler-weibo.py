#!/usr/bin/env python
# coding: utf-8

# # 轻型微博爬虫
# 
# 
# - https://github.com/hidadeng/weibo_crawler
# - https://pypi.org/project/weibo-crawler/
# 
# weibo_crawler参考【nghuyong/WeiboSpider】 对代码用法进行了简化，可以做轻度的微博数据采集。
# 
# - 用户信息抓取
# - 用户微博抓取(全量/指定时间段)
# - 用户社交关系抓取(粉丝/关注)
# - 微博评论抓取
# - 基于关键词和时间段(粒度到小时)的微博抓取
# - 微博转发抓取
# 
# 
# 使用简介：https://www.douban.com/group/topic/247718378/

# ## 安装

# In[2]:


pip install weibo-crawler


# ## 获取cookie
# 
# - 使用chrome浏览器打开手机微博 https://weibo.cn 登录
# - 右键inspect（即打开开发者模式）
# - 查看network内容
# - 获取html文件header中的cookie信息
#     - 其中可能需要SSOLoginState字段
#     
# ![image.png](img/weibo-crawler.png)

# In[3]:


from weibo_crawler import Profile


# In[6]:


# 如果程序失败，需要传入你的微博cookies
cookies='_T_WM=9b80727fa0cc3b6b6c374b9262ff084d; SUB=_2A25MjZmZDeRhGeNI4lYX-S7FwjWIHXVscSfRrDV6PUJbktAKLXSmkW1NSAJ40ykLq1lxtFqpHJ4BRMiY1XKHNT6g; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhkPe1HSir85xF8hwHpTZa75NHD95QfSo.XSo.71K.4Ws4DqcjT9s8Xqgpyqoz7eK-t; SSOLoginState=1636428233'

# csv文件路径
prof=Profile(csvfile='./data/weibo-chenkun-intro.csv', delay=1, cookies=cookies)

prof.get_profile(userid='1087770692') # 陈坤微博的id


# 更多操作见：
# 
# https://github.com/hidadeng/weibo_crawler
