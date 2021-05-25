#!/usr/bin/env python
# coding: utf-8

# # 万神殿项目（Pantheon Project）
# 
# 由塞萨尔·伊达尔戈（César Hidalgo）创建的一个在线工具。
# 
# - 伊达尔戈现在是麻省理工学院媒体实验室的教授，
#     - 他曾说:"真正著名的人在他们各自的领域外也相当知名"。
# - 一个人的维基百科页面使用了多少种语言，他就有多大的名气。
# 
# 若想被列入万神殿，一个人的名气必须跨越国家和语言障碍，必须在维基百科页面上出现至少25种语言。
# 
# 单单这一个要求就将名人的范围从所有的小名人或不太出名的人缩小到11341人——他们各有特色，魅力十足。

# Yu, A. Z., et al. (2016). Pantheon 1.0, a manually verified dataset of globally famous biographies. Scientific Data 2:150075. doi: 10.1038/sdata.2015.75
# 
# https://pantheon.world/data/datasets
# 
# ### pantheon.tsv
# A tab delimited file containing a row of data per person found in the Panthon 1.0 dataset.
# 
# ### wikilangs.tsv
# A tab delimited file of all the different Wikipedia language editions that each biography has a presence in.
# 
# ### pageviews_2008-2013.tsv
# A file containing the monthly pageview data for each individual, for all the Wikipedia language editions in which they have a presence.
# 
# Please refer to the methods section for more information on how this data was created. For detailed descriptions of these datasets, please refer to our data descriptor paper.

# - Jara-Figueroa, C., Yu, A.Z. and Hidalgo, C.A., 2015. The medium is the memory: how communication technologies shape what we remember. arXiv preprint arXiv:1512.05020.
# 
# - Yu, A.Z., Ronen, S., Hu, K., Lu, T. and Hidalgo, C.A., 2016. Pantheon 1.0, a manually verified dataset of globally famous biographies. Scientific data, 3.
# 
# - Ronen, S., Gonçalves, B., Hu, K.Z., Vespignani, A., Pinker, S. and Hidalgo, C.A., 2014. Links that speak: The global language network and its association with global fame. Proceedings of the National Academy of Sciences, 111(52), pp.E5616-E5622.
# 
# - Cesar A. Hidalgo and Ali Almossawi. "The Data-Visualization Revolution." Scientific American. March 2014.
# 
# - Hidalgo, C. A. "The Last 20 Inches: Data’s Treacherous Journey from the Screen to the Mind." MIT Technology Review. March 2014.
# 
# 

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[136]:


df = pd.read_csv('./data/person_2020_update.csv',low_memory=False)
df.head()


# In[5]:


len(df)


# In[45]:


df.iloc[0]


# In[6]:


df.columns


# The HPI combines the number of languages L, the effective number of languages L*, the age of the historical character A, the number of PageViews in Non-English Wikipedias v_NE (calculated in 2016), and the coefficient of variation in PageViews in all languages between CV (also calculated in 2016).
# 
# https://pantheon.world/about/methods

# In[18]:


plt.style.use('ggplot')
plt.plot(df['l_'], df['hpi'], 'o')
plt.xscale('log')
plt.show()


# In[15]:


plt.style.use('ggplot')
plt.plot(df['non_en_page_views'], df['hpi'], 'o')
plt.xscale('log')
plt.show()


# In[42]:


plt.style.use('ggplot')
plt.plot(df['l'], df['hpi'], 'o')
#plt.xscale('log')
plt.show()


# In[14]:


plt.hist(df['hpi']);


# In[17]:


plt.hist(df['non_en_page_views']);


# In[149]:


sns.pointplot(x='gender', y = 'hpi', data = df, color = 'blue', linestyles='');


# In[148]:


sns.pointplot(x='gender', y = 'hpi', ci = 'sd', data = df, color = 'blue', linestyles='');


# In[137]:


sns.boxplot(x="gender", y="hpi",
            #hue="smoker", 
            palette=["m", "g"],
            data=df);


# In[139]:


# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df, kind="bar",
    x="gender", y="hpi", hue="gender",
    ci="sd", palette="dark", alpha=.6, height=6
)


# In[151]:


df['age_group']=pd.cut(df['age'], bins = 6)


# In[29]:


plt.figure(figsize = (16, 5))
sns.pointplot(x="age_group", y="hpi", data=df)
plt.xlabel('Age', fontsize = 16)
plt.ylabel('HPI', fontsize = 16)
plt.show()


# In[30]:


plt.figure(figsize = (16, 5))
sns.pointplot(x="age_group", y="hpi", data=df, hue = 'gender')
plt.xlabel('Age', fontsize = 16)
plt.ylabel('HPI', fontsize = 16)
plt.show()


# In[31]:


plt.figure(figsize = (16, 5))
sns.pointplot(x="age_group", y="non_en_page_views", data=df, hue = 'gender')
plt.xlabel('Age', fontsize = 16)
plt.ylabel('HPI', fontsize = 16)
plt.show()


# In[153]:


plt.figure(figsize = (16, 5))
sns.pointplot(x="age_group", y="l", data=df, hue = 'gender')
plt.xlabel('Age', fontsize = 16)
plt.ylabel('Language Impact', fontsize = 16)
plt.show()


# In[152]:


plt.figure(figsize = (16, 5))
sns.pointplot(x="age_group", y="l", ci = 'sd', data=df, hue = 'gender')
plt.xlabel('Age', fontsize = 16)
plt.ylabel('Language Impact', fontsize = 16)
plt.show()


# In[35]:


df['occupation'].unique()


# In[39]:


plt.figure(figsize = (8, 20))

sns.boxplot(x="hpi", y="occupation", data=df,
            whis=[0, 100], width=.6, palette="vlag");


# In[44]:


plt.figure(figsize = (8, 20))

sns.boxplot(x="hpi", y="occupation", data=df[df['alive']==True],
            whis=[0, 100], width=.6, palette="vlag");


# In[72]:


import numpy as np

dat = df[(pd.isna(df['bplace_lat'])==False) &(pd.isna(df['dplace_lat'])==False)]
len(dat)


# In[109]:


dat0 = dat[dat['birthyear']<=0]
len(dat0)


# In[113]:


dat0['name']


# In[132]:


import plotly.graph_objects as go

fig = go.Figure()

#'bplace_lat', 'bplace_lon'，'dplace_lat', 'dplace_lon'

fig.add_trace(go.Scattergeo(
    #locationmode = 'USA-states',
    lon = dat0['bplace_lon'],
    lat = dat0['bplace_lat'],
    hovertext = dat0['name'],
    mode = 'markers',
    marker = dict(
        size = 2,
        color = 'rgb(255, 0, 0)',
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))


fig.update_layout(
    title_text = 'Pantheon Project',
    showlegend = False,
    geo = dict(
        scope = 'world',
        projection_type = 'natural earth',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',

    ),
)

fig.show()


# In[133]:


import plotly.graph_objects as go

fig = go.Figure()

#'bplace_lat', 'bplace_lon'，'dplace_lat', 'dplace_lon'


for i in dat0.index:
    fig.add_trace(
        go.Scattergeo(
            #locationmode = 'USA-states',
            lon = [dat0['bplace_lon'][i], dat0['dplace_lon'][i]],
            lat = [dat0['bplace_lat'][i], dat0['dplace_lat'][i]],
            mode = 'lines',
            line = dict(width = 1,color = 'red'),
            opacity = 0.5,
            hovertext = dat0['name'],
            hoverinfo="text",
        )
    )

fig.update_layout(
    title_text = 'Pantheon Project',
    showlegend = False,
    geo = dict(
        scope = 'world',
        projection_type = 'natural earth',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',

    ),
)

fig.show()


# In[ ]:




