#!/usr/bin/env python
# coding: utf-8

# 
# # 对占中新闻进行数据清洗
# 

# In[1]:


# 使用with open读取每一行数据
with open("../data/occupycentral/zz-hk-2014-10.rtf", encoding = 'gb18030') as f:
    news = f.readlines()


# In[2]:


# 查看总共有多少行
len(news)


# In[3]:


# 注意：标题和版面之间存在一个空行！所以title是block的第4个元素。
for i in range(1, 80):
    print(news[i]) 


# In[5]:


# 需要对中文编码的对象使用中文的方式进行解码
print(news[17][:500])


# In[4]:


# 定义一个函数：实现解码、编码、清洗效果
def stringclean(s):
    #s = s.decode('gb18030').encode('utf8')
    s = s.replace(r'\loch\af0\hich\af0\dbch\f15 \b\cf6 ', '')
    s = s.replace(r'\loch\af0\hich\af0\dbch\f15 \b0\cf0 ', '')
    s = s.replace('\par', '').replace('\n', '')
    return s


# In[5]:


'aabbccaadd ee aa'.strip('a')  


# In[6]:


'aabbccdd ee'.strip('ab')


# In[7]:


'aabbccdd ee'.replace('ab', '')


# In[8]:


# 调用stringclean函数
stringclean(news[17]) 


# In[9]:


# 列表内的for循环
news_clean = [stringclean(n) for n in news]
len(news_clean)


# In[10]:


news_clean[17][:120]


# In[11]:


# 定义两个函数
def deletetab(s):
    return s.replace('\t', '')

import sys
def flushPrint(s):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % s)
    sys.stdout.flush() # 清洗掉 


# In[12]:


help(sys.stdout)


# In[13]:


# 调用deletetab
deletetab('\ta')


# In[14]:


# 演示：flushPrint
import time, random
for i in range(10):
    time.sleep(random.random())
    flushPrint(i) 


# In[12]:


from collections import defaultdict

def readblocks(data):
    copy = False
    n = 0
    block = []
    chunk = defaultdict(lambda:[])
    for i in data:
        try:
            if "~~~~~~~~~~~~~~~~~~~~~~~~~~  #" in i:
                copy = True
            elif "文章编号:" in i:
                ids = i.replace('文章编号: ', '')
                source = block[0].split('|')[0]
                info = block[1]
                title = deletetab(block[3]) # 
                body = [j for j in block[6:] if j != '\n']
                body = ' '.join(body)
                body = deletetab(body)
                body = '"' + body  + '"'
                line = '\t'.join([ids, source, info, title, body])
                chunk[ids] = line
                block = []
                n += 1
                if n%10 == 0:
                    flushPrint(n)
                copy = False
            # copy must be here.
            elif copy:
                block.append(i)
        except Exception as e:
            print(i, e)
            pass
    return chunk


# In[13]:


# 注意：标题和版面之间存在一个空行！所以title是block的第4个元素。
for i in range(1, 8):
    print(news[i][:500])


# In[14]:


# 按block清洗新闻报道
news_result = readblocks(news_clean) 


# In[15]:


# 新闻的数量
len(news_result)


# In[16]:


# 查看字典的keys
list(news_result.keys())[:5]


# In[21]:


# 查看字典的values
list(news_result.values())[90]


# In[22]:


import pandas as pd
news_list = [i.split('\t') for i in news_result.values()]
df_news = pd.DataFrame(news_list,
                       columns= ['ids', 'source',\
                                 'info', 'title', 'body'])
df_news.head(3)


# In[24]:


# 保存数据：将数据写入硬盘
with open('../data/zz-hk-2014-9-clean.txt','a') as p:
     for record in news_result.values():
            p.write(record+"\n")


# In[25]:


# 使用pandas读取数据，并查看。
import pandas as pd
df = pd.read_csv( '../data/zz-hk-2014-9-clean.txt', 
                 sep = "\t", names=['ids', 'source','info', 'title', 'body'])
df[:3]


# In[23]:


# 使用os改变默认的工作路径
#import os
#os.chdir('../data/occupycentral/')

# 使用glob读取某一类文件的所有名称
import glob
filenames = glob.glob('../data/occupycentral/*.rtf')
filenames


# In[28]:


for i in filenames:
    print(i)
    with open(i, encoding = 'gb18030', errors = 'ignore') as f:  # for windows users: errors = 'ignore'
        news = f.readlines()
        news = [stringclean(n) for n in news]
        news_result = readblocks(news)
        with open('../data/zz-hk-all-clean2018.txt','a') as p:
            for record in news_result.values():
                p.write(record+"\n")


# 
# ## 自学Pandas使用
# 
# 《Python Data Science Handbook》第三章 by Jake VanderPlas
# 
# 
# 
# https://github.com/computational-class/datascience/blob/gh-pages/4.datasci/notebooks/03.00-Introduction-to-Pandas.ipynb

# ![](./images/end.png)
