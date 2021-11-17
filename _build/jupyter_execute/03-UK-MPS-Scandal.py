#!/usr/bin/env python
# coding: utf-8

# # 案例：2009年英国国会议员开支丑闻
# 
# 
# MPs expenses scandal 
# 
# https://github.com/data-journalism/data-journalism.github.io/discussions/54
# 

# In[74]:


import pandas as pd

import pylab as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号, 注意['SimHei']对应这句不行.
plt.style.use('ggplot')


# In[1]:


ls './data/'


# In[4]:


df = pd.read_excel("./data/MPs' expenses claims, Jul-Dec, 2009.xlsx")
df.head()


# In[7]:


df.columns


# In[5]:


len(df)


# In[6]:


df.describe()


# ## 清洗数据

# In[61]:


df[df['Amount, £']=='Amount']


# In[62]:


df = df[df['Amount, £']!='Amount']


# In[63]:


df[df['Name of member']=='John Randall']


# In[64]:


df['Amount, £'] = [abs(float(i)) for i in df['Amount, £']]


# ## 描述数据

# In[11]:


df[['Name of member', 'Amount, £']].groupby("Name of member").agg('sum')


# In[71]:


dat1 = df[['Name of member', 'Amount, £']].groupby("Name of member").agg('sum')
dat1 = dat1.sort_values(by=['Amount, £'], ascending = False)
dat1[:10]


# In[72]:


dat1['Rank'] = range(1, len(dat1)+1)


# In[73]:


dat1[:10]


# In[83]:


plt.figure(figsize =(16, 6), dpi = 100)

plt.plot(dat1['Rank'], dat1['Amount, £'], 'ro')
# plt.yscale('log')
# plt.xscale('log')

plt.xlabel('Rank')
plt.ylabel('Amount, £')
plt.show()


# In[84]:


plt.figure(figsize =(16, 6), dpi = 100)

dat10 = dat1[:10]
plt.plot(dat10['Rank'], dat10['Amount, £'], 'ro')
# plt.yscale('log')
# plt.xscale('log')

plt.xlabel('Rank')
plt.ylabel('Amount, £')
plt.show()


# In[87]:


dat2 = df[['Allowance Type', 'Amount, £']].groupby("Allowance Type").agg('sum')
dat2 = dat2.sort_values(by=['Amount, £'], ascending = False)
dat2


# In[93]:


dat3 = df[['Expenditure Type', 'Amount, £']].groupby("Expenditure Type").agg('sum')
dat3 = dat3.sort_values(by=['Amount, £'], ascending = False)
dat3


# In[96]:


plt.figure(figsize =(16, 4), dpi = 100)
dat3['Amount, £'].plot(kind = 'bar')
plt.yscale('log')
plt.show()


# ## 桑基图

# In[104]:


from pyecharts import options as opts
from pyecharts.charts import Sankey

nodes = [
    {"name": "category1"},
    {"name": "category2"},
    {"name": "category3"},
    {"name": "category4"},
    {"name": "category5"},
    {"name": "category6"},
]

links = [
    {"source": "category1", "target": "category2", "value": 10},
    {"source": "category2", "target": "category3", "value": 15},
    {"source": "category3", "target": "category4", "value": 20},
    {"source": "category5", "target": "category6", "value": 25},
]
c = (
    Sankey()
    .add(
        "sankey",
        nodes,
        links,
        linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
        label_opts=opts.LabelOpts(position="right"),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="Sankey-基本示例"))
    #.render("sankey_base.html")
    .render_notebook()
)

c


# In[118]:


dat4 = df[['Allowance Type','Expenditure Type', 'Amount, £']].groupby(['Expenditure Type', 'Allowance Type'],  as_index = False).agg('sum')
dat4 = dat4.sort_values(by=['Amount, £'], ascending = False)
dat4


# In[140]:


nodes = dat4['Allowance Type'].unique().tolist() + dat4['Expenditure Type'].unique().tolist()
nodes = [{'name': i} for i in nodes]
links = [{'source': dat4['Allowance Type'][i], 'target': dat4['Expenditure Type'][i], 'value': dat4['Amount, £'][i]}  for i in dat4.index]


# In[144]:


from pyecharts import options as opts
from pyecharts.charts import Sankey

s = (
    Sankey()
    .add(
        "sankey",
        nodes,
        links,
        linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
        label_opts=opts.LabelOpts(position="right"),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="Sankey Graph of UK MPS"))
)

s.render_notebook()


# In[145]:


s.render("sankey_mps.html")


# https://data-journalism.github.io/notebook/sankey_mps.html

# In[ ]:




