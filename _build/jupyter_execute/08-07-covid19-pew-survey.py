#!/usr/bin/env python
# coding: utf-8

# 
# # Analysing the Pew Survey Data of COVID19
# 
# 
# 
# ![image.png](./images/author.png)

# 阅读 https://www.journalism.org/dataset/election-news-pathways-april-2020-survey/
# 
# 1. 下载Pathways-April-2020-ATP-W66-1.zip 数据，该数据来自pewresearch针对2020年总统选举所做的panel study中的一次，其中加入了关于covid19的部分问题。
# 

# ![image.png](./images/pew.png)

# ![image.png](./images/pew2.png)
# 
# https://www.journalism.org/2020/05/08/americans-views-of-the-news-media-during-the-covid-19-outbreak/

# ![image.png](./images/pew3.png)

# In[1]:


# the statsmodels.api uses numpy array notation
# statsmodels.formula.api use formula notation (similar to R's formula notation)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


cd "/Users/datalab/bigdata/Pathways April 2020 (ATP W66)/"


# In[3]:


ls


# First of all, we install the pyreadstat module, which allows us to import SPSS files as DataFrames 

# In[10]:


get_ipython().system('pip install pyreadstat')


# In[185]:


# df = pd.read_spss('Pathways April 2020 (ATP W66).sav')
# df.head()


# In[5]:


import pyreadstat
df, meta = pyreadstat.read_sav('Pathways April 2020 (ATP W66).sav')
df.head(5)


# In[26]:


meta_dict = dict(zip(meta.column_names, meta.column_labels))
meta_dict.keys()


# In[61]:


meta_dict['F_INCOME'], meta_dict['COVIDCOVER1_W66'], meta_dict['COVIDFOL_W66']


# In[162]:


for i, j in meta_dict.items():
    print(i, j)


# In[6]:


df['WEIGHT_W66'].describe()


# In[183]:


df['F_AGECAT'].value_counts(normalize=True).sort_index()


# In[204]:


df['COVIDNEWSPAIR_a_W66'].value_counts(normalize=True).sort_index()


# In[205]:


df['COVIDNEWSPAIR_a_W66'].map(meta.variable_value_labels['COVIDNEWSPAIR_a_W66']).value_counts(normalize=True)


# ## Data Cleaning

# In[7]:


def clean_missing_data(var):
    # raw data using 99 as missing values
    df[var][df[var]==99] =np.nan 
    df[var] = df[var].fillna(df[var].median())
    return df[var]

for i in df.columns:
    df[i] = clean_missing_data(i)
    


# In[8]:


# dummy coding
df['republic'] = [1 if i==1 else 0 for i in df['F_PARTYLN_FINAL'] ]
df['edu'] = [7-i for i in df['F_EDUCCAT2']]


# ## Weighting
# 
# ![image.png](./images/pew4.png)
# 
# 按总体人口数据，总体平均收入应为：(5000 * 1000+4000 * 2000+3000 * 3000)/6000 = 3666.67 元。
# 
# - 按样本数据（不加权）：全部人口的样本平均收入 = (5000 * 200+4000 * 100+3000 * 100)/400 = 4250 元。
# - 采用总体/样本倍数加权，总体平均收入= (5000 * 200 * 5+4000 * 100 * 20+3000 * 100 * 30)/(200 * 5+100 * 20+100 * 30) = 3666.67元。

# **A weighting adjustment technique** can only be carried of proper auxiliary variables are available. 
# - Such variables must have been measured in the survey, and there population distribution must be available. 
# - Typical auxiliary variables are gender, age, marital status and region of the country. 
# - The population distribution of such variables can usually be obtained from national statistical institutes.
# 
# |            | Young | Middle | Old  |
# | ---------- | ----- | ------ | ---- |
# | Population | 30%   | 40%    | 30%  |
# | Sample     | 60%   | 30%    | 10%  |
# | Weight     | 0.5   | 1.33   |  3   |

# In[160]:


plt.hist(df["F_INCOME"], alpha =0.5, label = 'without weight')
plt.hist(df["F_INCOME"], weights=df["WEIGHT_W66"], alpha = 0.5, label = 'with weight')
plt.legend();


# In[161]:


plt.hist(df["COVIDCOVER1_W66"], alpha =0.5, label = 'without weight')
plt.hist(df["COVIDCOVER1_W66"], weights=df["WEIGHT_W66"], alpha = 0.5, label = 'with weight')
plt.legend();


# ## Sampling with weight

# In[108]:


df2 = df.sample(frac=0.66, weights=df['WEIGHT_W66'], random_state = 2020) 
df2.head(5)


# In[109]:


import numpy as np

print(df[['F_INCOME']].apply(np.average, weights=df['WEIGHT_W66']),'\n',
df[['F_INCOME']].apply(np.average),'\n',
df2[['F_INCOME']].apply(np.average)    )


# In[55]:


len(df), len(df2), df2[['F_INCOME']].apply(np.average)


# ## Weighted Regression: WLS vs. GLM
# 
# 
# The weights are presumed to be (proportional to) the inverse of the variance of the observations. That is, if the variables are to be transformed by 1/sqrt(W) you must supply weights = 1/W.
# 
# > sm.WLS?
# 
# https://www.coursera.org/learn/fitting-statistical-models-data-python
# 
# https://www.coursera.org/lecture/fitting-statistical-models-data-python/should-we-use-survey-weights-when-fitting-models-Qzt5p

# freq_weights : array_like
# 
#     1d array of frequency weights. The default is None. If None is selected
#     or a blank value, then the algorithm will replace with an array of 1's
#     with length equal to the endog.
#     WARNING: Using weights is not verified yet for all possible options
#     and results, see Notes.
#     
# var_weights : array_like
# 
#     1d array of variance (analytic) weights. The default is None. If None
#     is selected or a blank value, then the algorithm will replace with an
#     array of 1's with length equal to the endog.
#     WARNING: Using weights is not verified yet for all possible options
#     and results, see Notes.

# In[182]:


df1=pd.DataFrame({ 'x':range(1,101), 'wt':range(1,101) })

from statsmodels.stats.weightstats import DescrStatsW
wdf = DescrStatsW(df1.x, weights=df1.wt, ddof=1) 
print('without weight, the mean value is: ', np.mean(df1.x))
print( 'with weight, the mean value is: ', wdf.mean )
print( wdf.std )
print( wdf.quantile([0.25,0.50,0.75]) )


# In[201]:


# 'COVIDFOL_W66. How closely have you been following news about the outbreak of the coronavirus
import statsmodels.api as sm
X = sm.add_constant(df[['F_INCOME','F_AGECAT', 'edu', 'republic','COVIDCOVER1_W66', 'MH_TRACK_a_W66', 
                        'MH_TRACK_b_W66', 'MH_TRACK_d_W66', 'MH_TRACK_d_W66', 'MH_TRACK_e_W66']])
y = df['COVIDFOL_W66']
reg = sm.OLS(y,X, freq_weights=df['WEIGHT_W66'])
results = reg.fit()
reg1 = sm.GLM(y,X)
results1 = reg1.fit()
reg2 = sm.GLM(y,X, freq_weights=df['WEIGHT_W66'])
results2 = reg2.fit()
print(results.summary()) 


# In[202]:


print(results1.summary()) 


# In[203]:


print(results2.summary()) 


# ## GLM Binomial Regression
# 
# https://www.statsmodels.org/stable/glm.html
# 
# ![image.png](./images/pew5.png)

# In[9]:


# Largely accurate 
from statsmodels.genmod import families
dfs = df[df['COVIDNEWSPAIR_a_W66']!=2]
X = sm.add_constant(dfs[['F_INCOME','F_AGECAT', 'edu', 'republic','COVIDCOVER1_W66', 'MH_TRACK_a_W66', 
                        'MH_TRACK_b_W66', 'MH_TRACK_d_W66', 'MH_TRACK_d_W66', 'MH_TRACK_e_W66']])
y =[1 if i==1 else 0 for i in  dfs['COVIDNEWSPAIR_a_W66']]
reg1 = sm.GLM(y,X, family=families.Binomial())
results1 = reg1.fit()
reg2 = sm.GLM(y,X, family=families.Binomial(), freq_weights=dfs['WEIGHT_W66'])
results2 = reg2.fit()
print(results2.summary()) 


# In[10]:


# Largely inaccurate 
y =[1 if i==3 else 0 for i in  dfs['COVIDNEWSPAIR_a_W66']]
reg1 = sm.GLM(y,X, family=families.Binomial())
results1 = reg1.fit()
reg2 = sm.GLM(y,X, family=families.Binomial(), freq_weights=dfs['WEIGHT_W66'])
results2 = reg2.fit()
print(results2.summary()) 


# ![image.png](./images/end.png)
