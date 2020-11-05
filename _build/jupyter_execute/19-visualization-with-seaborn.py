#!/usr/bin/env python
# coding: utf-8

# # 第十一章 可视化
# 
# 
# ## Visualization with Seaborn

# - Seaborn is a Python data visualization library based on matplotlib. 
# - It provides a high-level interface for drawing attractive and informative statistical graphics.
# - it integrates with the functionality provided by Pandas DataFrames.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np; np.random.seed(22)
import seaborn as sns; 
import pylab as plt


# To be fair, the Matplotlib team is addressing this: 
# - it has recently added the plt.style tools, 
# - is starting to handle Pandas data more seamlessly.

# ## Matplotlib Styles

# In[8]:


plt.style.available


# The basic way to switch to a stylesheet is to call
# 
# ``` python
# plt.style.use('stylename')
# ```
# 
# But keep in mind that this will change the style for the rest of the session!
# Alternatively, you can use the style context manager, which sets a style temporarily:
# 
# ``` python
# with plt.style.context('stylename'):
#     make_a_plot()
# ```
# 

# In[5]:


x = np.linspace(0, 10, 1000)
plt.plot(x, np.sin(x));


# In[7]:


plt.style.use('fivethirtyeight')
x = np.linspace(0, 10, 1000)
plt.plot(x, np.sin(x));


# ## Seaborn Datasets

# In[49]:


sns.get_dataset_names()


# ## lineplot

# In[61]:


fmri = sns.load_dataset("fmri")


# In[20]:


fmri.head()


# In[33]:


ax = sns.lineplot(x="timepoint", y="signal", err_style="band",data=fmri)


# In[28]:


ax = sns.lineplot(x="timepoint", y="signal", err_style="bars",data=fmri)


# In[63]:


ax = sns.lineplot(x="timepoint", y="signal", ci=95, color="m",data=fmri)
ax = sns.lineplot(x="timepoint", y="signal", ci=68, color="b",data=fmri)


# In[35]:


ax = sns.lineplot(x="timepoint", y="signal", ci='sd', color="m",data=fmri)


# In[37]:


ax = sns.lineplot(x="timepoint", y="signal", estimator=np.median, data=fmri)


# In[41]:


#ax = sns.tsplot(data=data, err_style="boot_traces", n_boot=500)
ax = sns.lineplot(x="timepoint", y="signal", err_style="band", n_boot=500, data=fmri)


# http://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot

# ## Bar Plot

# In[42]:


import seaborn as sns; 
sns.set(color_codes=True)
tips = sns.load_dataset("tips")
ax = sns.barplot(x="day", y="total_bill", data=tips)


# In[43]:


ax = sns.barplot(x="day", y="total_bill", hue="sex", data=tips)


# ## Clustermap

# **Discovering structure in heatmap data**
# 
# http://seaborn.pydata.org/examples/structured_heatmap.html

# In[57]:


df = sns.load_dataset("titanic")
df.corr()


# In[60]:


# Draw the full plot
ax = sns.clustermap(df.corr(), center=0, cmap="vlag",
               #row_colors=network_colors, col_colors=network_colors,
               linewidths=.75, figsize=(5, 5))


# In[ ]:





# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# 19-visualization-with-pyecharts
# 19-visualization-maps-using-folium
# 19-visualization-datashader
# 19-visualization-datapane
# ```
# 
