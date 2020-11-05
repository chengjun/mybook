#!/usr/bin/env python
# coding: utf-8

# # 使用Datapane制作数据报告
# 
# Using Datapane to build shareable reports 
# 
# 
# ![image.png](images/datapane.png)
# 
# https://github.com/datapane/datapane/

# In[1]:


pip install datapane


# In[2]:


import pandas as pd
import altair as alt
import datapane as dp


# In[3]:


df = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/GOOG?period2=1585222905&interval=1mo&events=history')

chart = alt.Chart(df).encode(
    x='Date:T',
    y='Open'
).mark_line().interactive()

r = dp.Report(dp.Table(df), dp.Plot(chart))
r.save(path='report.html', open=True)


# In[ ]:




