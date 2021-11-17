#!/usr/bin/env python
# coding: utf-8

# 
# 
# # 社交媒体可以预测新冠疫情吗？
# 
# 基于知微事见数据
# 
# 
# http://xgml.zhiweidata.net/?from=floating#/
# 

# ![image.png](./images/pew6.png)

# In[1]:


get_ipython().run_cell_magic('html', '', '<iframe frameborder="no" border="0" marginwidth="0" marginheight="0" \n    width=900 height=600 \n    src="//xgml.zhiweidata.net/?from=floating#/">\n</iframe>')


# In[7]:


import pylab as plt
import pandas as pd
import seaborn as sns
import numpy as np
import json

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号, 注意['SimHei']对应这句不行.


# In[3]:


#j = json.load(open('../data/zhiwei_line.json'))
j = json.load(open('./data/zhiwei_line0417.json'))
df = pd.DataFrame(j)
df.tail()


# In[4]:


df.info()


# In[5]:


df['time'][:3]


# In[11]:


df['heat'] = [np.float64(i) for i in df['heat']]
df['case'] = [np.int32(i) for i in df['case']]


# In[12]:


df['heat']


# In[15]:


plt.style.use('ggplot')
plt.figure(figsize = [10, 6], dpi = 100)
plt.plot(df['heat'], df['case'], 'bo')
plt.yscale('log')
#plt.xscale('log')
plt.ylabel('新增确诊', fontsize = 16)
plt.xlabel('舆论热度', fontsize = 16)
plt.xlim([100000, 800000])
plt.show()


# In[16]:


MIN = df['heat'].min()
MAX = df['heat'].max()

bins = 10 ** np.linspace( np.log10(MIN), np.log10(MAX),20 )
plt.hist(df['heat'], bins = bins)
plt.xlabel('舆论热度', fontsize = 20)
plt.ylabel('频数', fontsize = 20)
plt.xscale('log')
plt.yscale('log')
plt.show()


# In[17]:


MIN = df['case'].min()+1
MAX = df['case'].max()

bins = 10 ** np.linspace( np.log10(MIN), np.log10(MAX),20 )
plt.hist(df['case'], bins = bins)
plt.xlabel('新增确诊', fontsize = 20)
plt.ylabel('频数', fontsize = 20)
plt.xscale('log')
plt.yscale('log')
plt.show()


# In[18]:


plt.hist( df['heat'], bins = 50)
plt.yscale('log')
plt.xscale('log')
plt.show()


# In[19]:


#plt.hist( df['heat'], bins = 50)
plt.hist( df['case'], bins = 50)
plt.yscale('log')
plt.xscale('log')
plt.show()


# In[31]:


# plot
fig = plt.figure(figsize=(30,10),dpi = 200)
plt.style.use('fivethirtyeight')

#plt.tick_params(labelsize = 20) 

ax1=fig.add_subplot(111)
ax1.plot(df['time'],  df['heat'], 'r-s')
ax1.set_ylabel('舆论热度', fontsize = 26)
ax1.tick_params(axis='x', rotation=60)
ax1.legend(('舆论热度',),loc='upper left', fontsize = 26)
#ax1.set_yscale('log')

ax2=ax1.twinx()
ax2.plot(df['time'], df['case'], 'g-o')
ax2.set_ylabel('新增确诊', fontsize = 26)
ax2.legend(('新增确诊',),loc='upper right', fontsize = 26)
#ax2.set_yscale('log')

plt.show()


# In[61]:


# plot
plt.figure(figsize=(12, 6), dpi = 200)
plt.style.use('fivethirtyeight')
plt.plot(df['time'], [float(i)/100 for i in df['heat']], 'r-s', label = '舆论热度/100')
plt.plot(df['time'], [int(i) for i in df['case']], 'g-o', label = '新增确诊')
plt.legend()
plt.xticks(rotation=60)
plt.ylabel('数量', fontsize = 20)
plt.show()


# ## 格兰杰因果检验
# 
# http://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html

# In[62]:


import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np


# help(df.pct_change)
# 
# Percentage change between the current and a prior element.
#     
#     Computes the percentage change from the immediately previous row by
#     default. This is useful in comparing the percentage of change in a time
#     series of elements.
#     

# ### The Null hypothesis for grangercausalitytests 
# 
# > H0: the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1. 
# 
# Grange causality means that past values of x2 have a statistically significant effect on the current value of x1, taking past values of x1 into account as regressors. We reject the null hypothesis that x2 does not Granger cause x1 if the pvalues are below a desired size of the test.

# In[63]:


get_ipython().run_line_magic('pinfo', 'grangercausalitytests')
# The data for test whether the time series in the second column Granger
#     causes the time series in the first column. Missing values are not
#     supported.


# In[64]:


data = df[21:][['case','heat' ]].pct_change().dropna()


# In[65]:


data.head()


# In[66]:


data.plot();


# In[67]:


gc_res = grangercausalitytests(data,4)


# In[68]:


data1 = df[21:][['heat','case']].pct_change().dropna()
gc_res1 = grangercausalitytests(data1,4)


# ### without peak

# In[32]:


df['case'][df['time']=='2020-02-12'] = np.nan
df['case'][df['time']=='2020-02-13'] = np.nan
df = df.fillna(method='ffill')


# In[33]:


# df = pd.read_excel('zhiwei_line_no_peak.xlsx')
df['heat'] = [float(i) for i in df['heat']]
df['case'] = [int(i) for i in df['case']]
df.tail()


# In[36]:


# plot
fig = plt.figure(figsize=(30,10),dpi = 200)
plt.style.use('fivethirtyeight')
ax1=fig.add_subplot(111)
ax1.plot(df['time'],  df['heat'], 'r-s')
ax1.set_ylabel('舆论热度', fontsize = 26)
ax1.tick_params(axis='x', rotation=60)
ax1.legend(('舆论热度',),loc='upper left', fontsize = 26)
ax2=ax1.twinx()
ax2.plot(df['time'], df['case'], 'g-o')
ax2.set_ylabel('新增确诊', fontsize = 26)
ax2.legend(('新增确诊',),loc='upper right', fontsize = 26)
plt.show()


# In[79]:


data = df[21:][['case','heat' ]].pct_change().dropna()


# In[80]:


data.plot();


# In[81]:


gc_res = grangercausalitytests(data,4)


# In[82]:


data = df[21:][['heat','case' ]].pct_change().dropna()
gc_res = grangercausalitytests(data,4)


# ### test the tails

# In[83]:


#df = pd.read_excel('zhiwei_line_no_peak.xlsx')
df['heat'] = [float(i) for i in df['heat']]
df['case'] = [int(i) for i in df['case']]
df[40:]


# In[84]:


data = df[40:][['heat','case' ]].pct_change().dropna()
gc_res = grangercausalitytests(data,3)


# In[85]:


data = df[40:][['case','heat' ]].pct_change().dropna()
gc_res = grangercausalitytests(data,3)


# ## Spurous Correlation
# 
# http://www.tylervigen.com/spurious-correlations

# In[132]:


import numpy as np
suicide = [5427,5688,6198,6462,6635,7336,7248,7491,8161,8578,9000]
spending = [18.079,18.594,19.753,20.734,20.831,23.029,23.597,23.584,25.525,27.731,29.449]
d = np.array([suicide, spending])


# In[133]:


df = pd.DataFrame(d.T, columns = ['suicide', 'spending']) # .pct_change().dropna()


# In[135]:


data = df[['suicide','spending' ]].pct_change().dropna()
gc_res = grangercausalitytests(data,2)


# In[136]:


data = df[['spending', 'suicide' ]].pct_change().dropna()
gc_res = grangercausalitytests(data,2)

