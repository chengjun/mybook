#!/usr/bin/env python
# coding: utf-8

# # Survival Analysis with Python
# 
# 
# **lifelines** is a complete survival analysis library, written in pure Python. What benefits does lifelines have?
# 
# - easy installation
# - internal plotting methods
# - simple and intuitive API
# - handles right, left and interval censored data
# - contains the most popular parametric, semi-parametric and non-parametric models
# 
# 
# https://lifelines.readthedocs.io/
# 
# ```
# pip install lifelines
# ```

# Cheibub, José Antonio, Jennifer Gandhi, and James Raymond Vreeland. 2010. “Democracy and Dictatorship Revisited.” Public Choice, vol. 143, no. 2-1, pp. 67-101.

# In[86]:


from lifelines.datasets import load_dd

data = load_dd()
data.head()


# In[5]:


data['regime'].unique()


# In[15]:


data['democracy'].unique()


# In[58]:


from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()


# In[59]:


T = data["duration"]
E = data["observed"]

kmf.fit(T, event_observed=E)


# In[60]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize = (8, 8))

ax = plt.subplot(111)

dem = (data["democracy"] == "Democracy")

t = np.linspace(0, 50, 51)
kmf.fit(T[dem], event_observed=E[dem], timeline=t, label="Democratic Regimes")
ax = kmf.plot_survival_function(ax=ax)

kmf.fit(T[~dem], event_observed=E[~dem], timeline=t, label="Non-democratic Regimes")
ax = kmf.plot_survival_function(ax=ax)

plt.title("Lifespans of different global regimes");


# In[61]:


regime_types = data['regime'].unique()
plt.figure(figsize = (12, 8))


for i, regime_type in enumerate(regime_types):
    ax = plt.subplot(2, 3, i + 1)

    ix = data['regime'] == regime_type
    kmf.fit(T[ix], E[ix], label=regime_type)
    kmf.plot_survival_function(ax=ax, legend=False)

    plt.title(regime_type)
    plt.xlim(0, 50)

    if i==0:
        plt.ylabel('Frac. in power after $n$ years')

plt.tight_layout()


# In[62]:


data['un_continent_name'].unique()


# In[87]:


import pandas as pd
df = pd.get_dummies(data['regime'])
df.head()


# In[88]:


data = pd.concat([data, df], axis=1)
data


# In[89]:


data.columns


# In[90]:


data['Democracy'] = [1  if i == 'Democracy' else 0 for i in data['democracy']]


# In[91]:


from lifelines import CoxPHFitter

cph0 = CoxPHFitter()

dat = data[['duration', 'observed', 'start_year','Democracy']]

cph0.fit(dat, duration_col='duration', event_col='observed')

cph0.print_summary() 


# In[82]:


dat = data[['duration', 'observed', 'start_year',
     'Civilian Dict', 'Military Dict', #'Monarchy',
       'Mixed Dem',  'Parliamentary Dem', 'Presidential Dem'
]]

from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(dat, duration_col='duration', event_col='observed')

cph.print_summary() 


# In[84]:


plt.figure(figsize = (12, 8))
cph.plot()
plt.show()


# In[85]:


cph.plot_partial_effects_on_outcome(covariates='Presidential Dem', values=[0, 1], cmap='coolwarm');


# In[ ]:




