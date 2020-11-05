#!/usr/bin/env python
# coding: utf-8

# 
# # 天涯论坛的回帖网络分析
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


dtt = []
file_path = '../data/tianya_bbs_threads_network.txt'
with open(file_path, 'r') as f:
    for line in f:
        pnum, link, time, author_id, author,        content = line.replace('\n', '').split('\t')
        dtt.append([pnum, link, time, author_id, author, content])
len(dtt)


# In[3]:


import pandas as pd
dt = pd.DataFrame(dtt)
dt=dt.rename(columns = {0:'page_num', 1:'link', 2:'time', 3:'author',4:'author_name', 5:'reply'})
dt[:5]


# In[8]:


# extract date from datetime
date = [i[:10] for i in dt.time]
dt['date'] = pd.to_datetime(date)


# In[9]:


dt[:5]


# In[10]:


import pandas as pd

df = pd.read_csv('../data/tianya_bbs_threads_list.txt', sep = "\t", header=None)
df=df.rename(columns = {0:'title', 1:'link', 2:'author',3:'author_page', 4:'click', 5:'reply', 6:'time'})
df[:2]


# In[11]:


from collections import defaultdict

link_user_dict = defaultdict(list)
for i in range(len(dt)):
    link_user_dict[dt.link[i]].append(dt.author[i])


# In[12]:


df['user'] = [len(link_user_dict[l]) for l in df.link]
df[:2] 


# In[13]:


import statsmodels.api as sm
import numpy as np

x = np.log(df.user+1)
y = np.log(df.reply+1)
xx = sm.add_constant(x, prepend=True)
res = sm.OLS(y,xx).fit()
constant,beta = res.params
r2 = res.rsquared
fig = plt.figure(figsize=(8, 4),facecolor='white')
plt.plot(df.user, df.reply, 'rs', label= 'Data')
plt.plot(np.exp(x), np.exp(constant + x*beta),"-", label = 'Fit')
plt.yscale('log');plt.xscale('log')
plt.xlabel(r'$Users$', fontsize = 20)
plt.ylabel(r'$Replies$', fontsize = 20)
plt.text(max(df.user)/300,max(df.reply)/20,
         r'$\beta$ = ' + str(round(beta,2)) +'\n' + r'$R^2$ = ' + str(round(r2, 2)))
plt.legend(loc=2,fontsize=10, numpoints=1)
plt.axis('tight')
plt.show()


# In[14]:


x = np.log(df.user+1)
y = np.log(df.click+1)
xx = sm.add_constant(x, prepend=True)
res = sm.OLS(y,xx).fit()
constant,beta = res.params
r2 = res.rsquared
fig = plt.figure(figsize=(8, 4),facecolor='white')
plt.plot(df.user, df.click, 'rs', label= 'Data')
plt.plot(np.exp(x), np.exp(constant + x*beta),"-", label = 'Fit')
plt.yscale('log');plt.xscale('log')
plt.xlabel(r'$Users$', fontsize = 20)
plt.ylabel(r'$Clicks$', fontsize = 20)
plt.text(max(df.user)/300,max(df.click)/20,
         r'$\beta$ = ' + str(round(beta,2)) +'\n' + r'$R^2$ = ' + str(round(r2, 2)))
plt.legend(loc=2,fontsize=10, numpoints=1)
plt.axis('tight')
plt.show()


# In[15]:


# convert str to datetime format
dt.time = pd.to_datetime(dt.time)
dt['month'] = dt.time.dt.month
dt['year'] = dt.time.dt.year
dt['day'] = dt.time.dt.day
type(dt.time[0])


# In[16]:


d = dt.year.value_counts()
dd = pd.DataFrame(d)
dd = dd.sort_index(axis=0, ascending=True)
ds = dd.cumsum()


# In[18]:


def getDate(dat):
    dat_date_str = map(lambda x: str(x) +'-01-01', dat.index)
    dat_date = pd.to_datetime(list(dat_date_str))
    return dat_date

ds.date = getDate(ds)
dd.date = getDate(dd)


# In[19]:


fig = plt.figure(figsize=(12,5))
plt.plot(ds.date, ds.year, 'g-s', label = '$Cumulative\: Number\:of\: Threads$')
plt.plot(dd.date, dd.year, 'r-o', label = '$Yearly\:Number\:of\:Threads$')
#plt.yscale('log')
plt.legend(loc=2,numpoints=1,fontsize=13)
plt.show()


# ## 提取评论信息

# In[20]:


dt.reply[:55]


# @贾也2012-10-297:59:00　　导语：人人宁波，面朝大海，春暖花开　　........
# 
#         @兰质薰心2012-10-2908:55:52　　楼主好文！　　相信政府一定有能力解决好这些...
#         
#                 回复第20楼，@rual_f　　“我相信官场中，许多官员应该葆有社会正能量”　　通篇好文，顶...

# In[21]:


import re
tweet = u"//@lilei: dd //@Bob: cc//@Girl: dd//@魏武:     利益所致 自然念念不忘// @诺什: 吸引优质  客户，摆脱屌丝男！！！//@MarkGreene: 转发微博"
RTpattern = r'''//?@(\w+)'''
for word in re.findall(RTpattern, tweet, re.UNICODE):
    print(word)


# In[25]:


print(dt.reply[11])


# In[26]:


RTpattern = r'''@(\w+)\s'''
re.findall(RTpattern, dt.reply[11], re.UNICODE)


# In[27]:


if re.findall(RTpattern, dt.reply[0], re.UNICODE):
    print(True)
else:
    print(False)


# In[28]:


for k, tweet in enumerate(dt.reply[:100]):
#     tweet = tweet.decode('utf8')
    RTpattern = r'''@(\w+)\s'''
    for person in re.findall(RTpattern, tweet, re.UNICODE):
        print(k,'\t',dt.author_name[k],'\t', person,'\t\t', tweet[:30])


# In[29]:


print(dt.reply[80])


# In[30]:


link_author_dict = {}
for i in range(len(df)):
    link_author_dict[df.link[i]] =df.author[i] 
    


# In[31]:


graph = []
for k, tweet in enumerate(dt.reply):
    url = dt.link[k]
    RTpattern = r'''@(\w+)\s'''
    persons = re.findall(RTpattern, tweet, re.UNICODE)
    if persons:
        for person in persons:
            graph.append([dt.author_name[k], person])
    else:
        graph.append( [dt.author_name[k], link_author_dict[url]]  )
        


# In[32]:


len(graph)


# In[33]:


for x, y in graph[:3]:
    print(x, y)


# In[34]:


import networkx as nx


# In[35]:


G = nx.DiGraph()
for x,y in graph:
    if x != y:
        G.add_edge(x,y)


# In[36]:


nx.info(G)


# In[37]:


GU=G.to_undirected(reciprocal=True)
graphs = list(nx.connected_component_subgraphs(GU))


# In[38]:


import numpy as np
size = []
for i in graphs:
    size.append(len(i.nodes()))
len(size), np.max(size)


# In[39]:


gs = []
for i in graphs:
    if len(i.nodes()) >5:
        gs.append(i)
len(gs)


# In[40]:


for g in gs:
    print(len(g.nodes()))


# In[42]:


g_max = gs[1]
len(g_max.nodes())


# In[43]:


pos = nx.spring_layout(g_max)          
#定义一个布局，此处采用了spectral布局方式，后变还会介绍其它布局方式，注意图形上的区别
nx.draw(g_max,pos,with_labels=False,node_size = 30)  
#绘制规则图的图形，with_labels决定节点是非带标签（编号）,node_size是节点的直径
plt.show()  #显示图形


# In[46]:


# with open('../data/tianya_network_120.csv', 'a') as f:
#     for x, y in g_max.edges():
#         f.write(x + ',' + y + '\n')


# ![](./img/tianyaGephi.png)
# 
# ## 使用 Gephi进行网络可视化

# 1	/post-free-2849477-1.shtml	2012-10-29 11:11:32	51150428	生生0326	一环　　天涯现在太滞后了，看消息还是得靠微博。太失望了。
# 

# 作业
# - 1. 计算节点的度，并绘制其散点图。
# - 2. 计算节点的聚类系数
# - 3. 计算节点的pagerank

# In[ ]:




