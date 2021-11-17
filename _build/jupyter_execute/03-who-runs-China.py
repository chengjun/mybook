#!/usr/bin/env python
# coding: utf-8

# # 案例：Who runs China？背后的数据
# 
# 作品链接：https://news.cgtn.com/event/2019/whorunschina/index.html
# 
# 案例解读 https://github.com/data-journalism/data-journalism.github.io/discussions/48

# In[42]:


import pandas as pd
import pylab as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号, 注意['SimHei']对应这句不行.


# In[43]:


df = pd.read_csv('./data/data.js')
df.head()


# In[44]:


len(df)


# In[4]:


df.columns


# ## 问题的提出
# 
# 使用这些变量，可以提出哪些问题？

# ## 描述性分析

# In[7]:


df.groupby('党派').mean()


# In[8]:


df['党派'].value_counts()


# In[9]:


df["党派"].value_counts(normalize = True)


# In[11]:


df["性别"].value_counts(normalize = True)


# **Pandas Plot**
# 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html

# In[3]:


plt.figure(figsize =(16, 4), dpi = 100)

df["民族"].value_counts(normalize = True).plot(kind = 'bar')
plt.yscale('log');


# In[4]:


plt.figure(figsize =(16, 4), dpi = 100)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号, 注意['SimHei']对应这句不行.

df["民族"].value_counts(normalize = True).plot(kind = 'bar');
#plt.yscale('log');


# In[25]:


plt.style.use('seaborn')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.figure(figsize =(4, 4))
df["性别"].value_counts(normalize = True).plot(kind = 'barh');


# In[19]:


plt.figure(figsize =(16, 4))
df["Birth year"].value_counts(normalize = True).plot(kind = 'bar');


# In[20]:


plt.figure(figsize =(16, 4))
df["Generation"].value_counts(normalize = True).plot(kind = 'bar');


# In[6]:


plt.figure(figsize =(16, 4), dpi = 100)
df["籍贯"].value_counts(normalize = True).plot(kind = 'bar');


# In[7]:


plt.figure(figsize =(16, 4), dpi = 100)
df["区域"].value_counts(normalize = True).plot(kind = 'bar');


# In[8]:


plt.figure(figsize =(16, 4), dpi = 100)
df["专业分类"].value_counts(normalize = True).plot(kind = 'bar');


# In[9]:


plt.figure(figsize =(16, 4), dpi = 100)
df["人文社科拆后专业"].value_counts(normalize = True).plot(kind = 'bar');


# In[11]:


df["人文社科拆后专业"].value_counts().sum()


# In[12]:


len(df)


# In[13]:


plt.figure(figsize =(16, 4), dpi = 100)
df["学历"].value_counts(normalize = True).plot(kind = 'bar');


# In[22]:


df["海外留学经验"] = [ str(i).replace(']', '')  for i in df["海外留学经验]"].tolist() ]


# In[ ]:


df["海外留学经验"]


# In[23]:


plt.figure(figsize =(16, 4), dpi = 100)
df["海外留学经验"].value_counts(normalize = True).plot(kind = 'bar');


# ## 列联表分析

# In[26]:


pd.crosstab(df['性别'],df['学历'],margins=True)


# In[27]:


pd.crosstab(df['性别'],df['学历'],margins=True, normalize='index')


# In[28]:


94/742


# In[38]:


import numpy as np
from scipy import stats

alist = np.array(pd.crosstab(df['性别'],df['学历'],margins=True)).tolist()
alist


# In[39]:


# 卡方检验

chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[41]:


pd.crosstab(df['性别'],df['专业分类'],margins=True, normalize='index')


# In[ ]:





# In[48]:


print(*df['姓名'].tolist())


# In[50]:


for i in df['姓名']:
    print('https://baike.baidu.com/item/'+str(i))

