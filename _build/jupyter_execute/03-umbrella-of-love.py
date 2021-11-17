#!/usr/bin/env python
# coding: utf-8

# # 案例：《转角遇到爱》背后的数据
# 
# 数据新闻的另一种视角与实现——《转角遇到爱》作品分析 #43
# 
# - 作品链接：https://h5.thepaper.cn/html/zt/2018/08/seekinglove/index.html
# - 简介：获得2018年SND（美国新闻媒体视觉设计协会）最佳数字设计铜奖。选一个晴天的周日，从上海人民广场地铁站9号口出门，左手边就是闻名全国的人民广场相亲角。五六十岁模样的大叔大妈们带着伞和小板凳，在这里为他们的晚辈寻觅一份姻缘。澎湃新闻 www.thepaper.cn 和姐妹英文媒体“第六声”的数据记者花费了六个周末的时间，收集了874份相亲广告。从中可以读出关于618位女士和256位男士的觅爱故事。
# - 解读：https://github.com/data-journalism/data-journalism.github.io/discussions/43

# In[26]:


import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号, 注意['SimHei']对应这句不行.


# In[2]:


ls './data/'


# In[4]:


df = pd.read_csv('./data/db_new.csv')
df.head()


# In[5]:


len(df)


# In[6]:


df.describe()


# In[7]:


df.columns


# In[11]:


df['Age'] = 2018 - df['Year.self']


# In[16]:


plt.figure(figsize =(8, 4), dpi = 100)

sns.histplot(
    df,
    x="Age", hue="Gender.self",
    edgecolor=".3",
    linewidth=.5,
    log_scale=True,
);


# In[18]:


# https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot
plt.figure(figsize =(8, 4), dpi = 100)
sns.violinplot(x="Gender.self", y="Age", data=df);


# In[28]:


# deal with missing data
df['Height.self'] = [float(i) if i != 'N' else np.nan for i in df['Height.self']]


# In[31]:


plt.figure(figsize =(8, 4), dpi = 100)
sns.violinplot(x="Gender.self", y="Height.self", data=df);


# In[20]:


df['Hukou.self'].unique()


# In[21]:


df['Hukou.self'].value_counts()


# In[34]:


df['Looking.self'].value_counts()


# In[35]:


df['Personality.self']


# In[39]:


df['Edu.self'].value_counts()


# In[56]:


df['Eduno.self'].value_counts()


# In[57]:


df['top.self'].value_counts()


# In[61]:


df['Abroad.self'].value_counts()


# In[66]:


df['Job.self'].value_counts()


# In[65]:


df['Major.self'].value_counts()


# In[67]:


df['Salary.self'].value_counts()


# In[68]:


df['Apt.self'].value_counts()


# In[72]:


df['Family.self'].value_counts()


# In[74]:


df['Hobby.self'].value_counts()


# In[76]:


df['Other.self'].value_counts()


# In[78]:


df['interesting.self'].value_counts()


# In[81]:


df['Live.self'].value_counts()


# In[82]:


df['Similar.wanted'].value_counts()


# In[83]:


df['Looking.wanted'].value_counts()


# In[89]:


df['Hukou.wanted'].value_counts()


# In[90]:


df['Looking.self.dummy'] = [1 if i != 'N' else 0 for i in df['Looking.self']]
df['Looking.wanted.dummy'] = [1 if i != 'N' else 0 for i in df['Looking.wanted']]
df['Personality.self.dummy'] = [1 if i != 'N' else 0 for i in df['Personality.self']]
df['Family.self.dummy'] = [1 if i != 'N' else 0 for i in df['Family.self']]
df['Hobby.self.dummy'] = [1 if i != 'N' else 0 for i in df['Hobby.self']]
df['Other.self.dummy'] = [1 if i != 'N' else 0 for i in df['Other.self']]
df['interesting.self.dummy'] = [1 if i != 'N' else 0 for i in df['interesting.self']]
df['Hukou.wanted.dummy'] = [1 if i != 'N' else 0 for i in df['Hukou.wanted']]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[32]:


['id', 'Gender.self', 'Year.self', 'Born.self', 'Hukou.self',
       'Live.self', 'Marriage.self', 'Height.self', 'Weight.self',
       'Looking.self', 'Personality.self', 'Edu.self', 'Eduno.self',
       'top.self', 'Abroad.self', 'Major.self', 'Job.self', 'Salary.self',
       'Apt.self', 'Family.self', 'Hobby.self', 'Other.self',
       'interesting.self', 'Gender.wanted', 'Year.max.wanted',
       'Year.min.wanted', 'Year.text.wanted', 'Hukou.wanted', 'Live.wanted',
       'Marriage.wanted', 'Height.min.wanted', 'Looking.wanted',
       'Personality.wanted', 'Hobby.wanted', 'Edu.min.wanted',
       'Edu.min.n.wanted', 'Job.wanted', 'Salary.min.wanted', 'Apt.wanted',
       'Family.wanted', 'Other.wanted', 'interesting.wanted',
       'Similar.wanted']


# In[ ]:





# ## 列联表分析

# In[42]:


pd.crosstab(df['Gender.self'],df['Looking.self.dummy'],margins=True)


# In[43]:


pd.crosstab(df['Gender.self'],df['Looking.self.dummy'],margins=True, normalize='index')


# In[48]:


import numpy as np
from scipy import stats

alist = np.array(pd.crosstab(df['Gender.self'],df['Looking.self.dummy'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[44]:


pd.crosstab(df['Gender.self'],df['Personality.self.dummy'],margins=True)


# In[45]:


pd.crosstab(df['Gender.self'],df['Personality.self.dummy'],margins=True, normalize='index')


# In[49]:


alist = np.array(pd.crosstab(df['Gender.self'],df['Personality.self.dummy'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[ ]:





# In[50]:


pd.crosstab(df['Gender.self'],df['Edu.self'],margins=True)


# In[51]:


pd.crosstab(df['Gender.self'],df['Edu.self'],margins=True, normalize='index')


# In[52]:


alist = np.array(pd.crosstab(df['Gender.self'],df['Edu.self'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[ ]:





# In[53]:


pd.crosstab(df['Gender.self'],df['Hukou.self'],margins=True)


# In[54]:


pd.crosstab(df['Gender.self'],df['Hukou.self'],margins=True, normalize = 'index')


# In[55]:


alist = np.array(pd.crosstab(df['Gender.self'],df['Hukou.self'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[ ]:





# In[58]:


pd.crosstab(df['Gender.self'],df['top.self'],margins=True)


# In[59]:


pd.crosstab(df['Gender.self'],df['top.self'],margins=True, normalize = 'index')


# In[60]:


alist = np.array(pd.crosstab(df['Gender.self'],df['top.self'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[ ]:





# In[62]:


pd.crosstab(df['Gender.self'],df['Abroad.self'],margins=True)


# In[63]:


pd.crosstab(df['Gender.self'],df['Abroad.self'],margins=True, normalize = 'index')


# In[64]:


alist = np.array(pd.crosstab(df['Gender.self'],df['Abroad.self'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[ ]:





# In[69]:


pd.crosstab(df['Gender.self'],df['Apt.self'],margins=True)


# In[70]:


pd.crosstab(df['Gender.self'],df['Apt.self'],margins=True, normalize = 'index')


# In[71]:


alist = np.array(pd.crosstab(df['Gender.self'],df['Apt.self'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[85]:


alist = np.array(pd.crosstab(df['Looking.self.dummy'],df['Looking.wanted.dummy'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[88]:


alist = np.array(pd.crosstab(df['Looking.self.dummy'],df['Looking.wanted.dummy'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# ### 介绍自己容貌者比起不介绍自己容貌的人对户口有要求更明确！

# In[92]:


pd.crosstab(df['Looking.self.dummy'],df['Hukou.wanted.dummy'],margins=True, normalize = 'index')


# In[91]:


alist = np.array(pd.crosstab(df['Looking.self.dummy'],df['Hukou.wanted.dummy'],margins=False)).tolist()
print(alist)

# 卡方检验
chi2, p, ddof, expected = stats.chi2_contingency( alist )
msg = "Test Statistic: {}\n p-value: {}\n Degrees of Freedom: {}\n"
print( msg.format( chi2, p, ddof ) )
print( expected )


# In[ ]:




