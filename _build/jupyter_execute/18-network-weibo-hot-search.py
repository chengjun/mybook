#!/usr/bin/env python
# coding: utf-8

# # 微博热搜分析

# In[73]:


import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号, 注意['SimHei']对应这句不行.


# In[2]:


df = pd.read_excel('./data/微博热搜数据2020.xlsx')


# In[3]:


df.head()


# In[4]:


len(df)


# ## 中文分词

# In[5]:


import jieba


# In[32]:


list(jieba.cut(df['title'][0], cut_all=False))


# In[19]:


get_ipython().system('pip install thulac')


# In[25]:


import thulac

thu1 = thulac.thulac()  #默认模式
text = thu1.cut("快本为何炅改播出时间", text=True)  #进行一句话分词
print(text)


# In[28]:


text = thu1.cut("快本为何炅改播出时间", text=False)  #进行一句话分词
print(text)


# In[29]:


text = thu1.cut("橘子洲烟花", text=False)  #进行一句话分词
print(text)


# In[30]:


text = thu1.cut("吴亦凡脖子", text=False)  #进行一句话分词
print(text)


# In[31]:


text = thu1.cut("吴昕状态", text=False)  #进行一句话分词
print(text)


# n/名词 np/人名 ns/地名 ni/机构名 nz/其它专名
# m/数词 q/量词 mq/数量词 t/时间词 f/方位词 s/处所词
# v/动词 a/形容词 d/副词 h/前接成分 k/后接成分 
# i/习语 j/简称 r/代词 c/连词 p/介词 u/助词 y/语气助词
# e/叹词 o/拟声词 g/语素 w/标点 x/其它 

# In[36]:


def cut_words(txt):
    text = thu1.cut(txt, text=False) 
    text = [i for i,j in text]
    return text

# flush print
import sys
def flushPrint(d):
    sys.stdout.write('\r')
    sys.stdout.write(str(d))
    sys.stdout.flush()


# In[35]:


cut_words('快本为何炅改播出时间')


# In[37]:


wlist = []
for k, i in enumerate(df['title']):
    if k % 100 ==0:
        flushPrint(k)
    text = cut_words(i)
    wlist.append(text)


# In[38]:


df['wlist'] = wlist


# In[40]:


df = df.drop('Unnamed: 0', axis=1,)  


# In[41]:


df


# In[43]:


df.to_excel('./data/微博热搜数据2020.xlsx', index = False)


# In[48]:


print(*df['title'][:100])


# In[50]:


df[100:]


# ## 语义网络

# In[59]:


def successive_list(alist):
    return [[alist[:-1][i], alist[1:][i]] for i in range(len(alist)-1)]


# In[67]:


successive_list(a)


# In[98]:


import networkx as nx
import itertools

G=nx.Graph()

for slist in wlist:
    if len(slist)>1:
        slist = [i  for i in slist if len(i) > 1]
        edgelist = successive_list(slist)
        for e1, e2 in edgelist:
            G.add_edge(e1, e2)


# In[99]:


nx.info(G)


# In[100]:


pr = nx.pagerank(G, alpha=0.9)
deg = dict(nx.degree_centrality(G))
dei = dict(nx.eigenvector_centrality(G))
dd = [(i, deg[i], dei[i], pr[i]) for i in deg]


dd = pd.DataFrame(dd, columns = ('behavior', 'Centrality','Eigenvector Centrality', 'PageRank'))
dd = dd.sort_values(by=['Centrality'], ascending = False)
dd = dd.reset_index()
dd[:20]


# In[101]:


dd[20:40]


# In[108]:


print(*dd['behavior'][:500])


# In[109]:


plt.figure(figsize = (6, 6))    

plt.style.use('ggplot')
sns.scatterplot(data=dd, x="Centrality", y="PageRank")

plt.xlabel('$Centrality$', fontsize = 20)
plt.ylabel('$PageRank$', fontsize = 20)

plt.show()


# In[75]:


from collections import defaultdict
import numpy as np

def plotDegreeDistribution(G):
    degs = defaultdict(int)
    for i in dict(G.degree()).values(): degs[i]+=1
    items = sorted ( degs.items () )
    x, y = np.array(items).T
    y_sum = np.sum(y)
    y = [float(i)/y_sum for i in y]
    plt.plot(x, y, 'bo')
    plt.xscale('log')
    plt.yscale('log')
    #plt.legend(['Degree'])
    plt.xlabel('$K$', fontsize = 20)
    plt.ylabel('$P(K)$', fontsize = 20)
    plt.title('$Degree\,Distribution$', fontsize = 20)
    plt.show()   


# In[110]:


plt.style.use('ggplot')


plt.figure(figsize = (6, 6))    

plotDegreeDistribution(G)


# In[115]:


G.remove_edges_from(nx.selfloop_edges(G))


# In[120]:


g = nx.k_core(G,k=16)
nx.info(g)


# In[121]:


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号, 注意['SimHei']对应这句不行.

plt.figure(figsize = (16, 16), dpi = 300)    

nx.draw(g, with_labels = True, edge_color="grey", node_color='blue', rotate = True, node_size = 26)
plt.margins(x=0.2)


# In[ ]:




