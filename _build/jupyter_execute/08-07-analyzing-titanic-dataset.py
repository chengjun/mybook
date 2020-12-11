#!/usr/bin/env python
# coding: utf-8

# 
# # Logistic Regression of Titanic Data
# 
# 
# 

# ## Statsmodels
# 
# http://statsmodels.sourceforge.net/
# 
# ![](./img/statsmodels_hybi_banner.png)
# 
# Statsmodels is a Python module that allows users to explore data, estimate statistical models, and perform statistical tests. 
# 
# An extensive list of descriptive statistics, statistical tests, plotting functions, and result statistics are available for different types of data and each estimator. 
# 
# Researchers across fields may find that statsmodels fully meets their needs for statistical computing and data analysis in Python. 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


# Features include:
# 
# - Linear regression models
# - Generalized linear models
# - Discrete choice models
# - Robust linear models
# - Many models and functions for time series analysis
# - Nonparametric estimators
# - A collection of datasets for examples
# - A wide range of statistical tests
# - Input-output tools for producing tables in a number of formats and for reading Stata files into NumPy and Pandas.
# - Plotting functions
# - Extensive unit tests to ensure correctness of results
# - Many more models and extensions in development

# In[3]:


train = pd.read_csv('./data/tatanic_train.csv',sep = ",", header=0)
test = pd.read_csv('./data/tatanic_test.csv',sep = ",", header=0)


# ### Describing Data 
#  - .describe() summarizes the columns/features of the DataFrame, including the count of observations, mean, max and so on. 
#  - Another useful trick is to look at the dimensions of the DataFrame. This is done by requesting the .shape attribute of your DataFrame object. (ex. your_data.shape)

# In[3]:


train.head()


# In[3]:


train.describe()


# In[4]:


train.shape#, len(train)
#train.columns


# In[5]:


# Passengers that survived vs passengers that passed away
train["Survived"][:3] 


# ## Value Counts
# 
# 以Series形式返回指定列的不同取值的频率

# In[6]:


# Passengers that survived vs passengers that passed away
train["Survived"].value_counts()


# In[7]:


# As proportions
train["Survived"].value_counts(normalize = True)


# In[8]:


train['Sex'].value_counts()


# In[9]:


train[train['Sex']=='female'][:3]#[train['Pclass'] == 3]


# In[10]:


# Males that survived vs males that passed away
train[["Survived", 'Fare']][train["Sex"] == 'male'][:3]


# In[11]:


# Males that survived vs males that passed away
train["Survived"][train["Sex"] == 'male'].value_counts() 


# In[12]:


# Females that survived vs Females that passed away
train["Survived"][train["Sex"] == 'female'].value_counts() 


# In[13]:


# Normalized male survival
train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True) 


# In[14]:


# Normalized female survival
train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)


# In[15]:


# Create the column Child, and indicate whether child or not a child. Print the new column.
train["Child"] = float('NaN')
train.Child[train.Age < 5] = 1
train.Child[train.Age >= 5] = 0
print(train.Child[:3])


# In[16]:


# Normalized Survival Rates for under 18
train.Survived[train.Child == 1].value_counts(normalize = True)


# In[17]:


# Normalized Survival Rates for over 18
train.Survived[train.Child == 0].value_counts(normalize = True)


# ## 透视表(pivotTab)
# 透视表就是将指定原有DataFrame的列分别作为行索引和列索引，然后对指定的列应用聚集函数(默认情况下式mean函数)。
# 
# ## 列联表（crossTab）
# 交叉表是用于统计分组频率的特殊透视表
# 
# Compute a simple cross tabulation of two (or more) factors. By default computes a frequency table of the factors unless an array of values and an aggregation function are passed.

# In[4]:


pd.crosstab(train['Sex'],train['Survived'],margins=True)


# In[20]:


pd.crosstab(train['Sex'],train['Survived'],margins=True, normalize='index')


# In[19]:


pd.crosstab(train['Sex'],[train['Survived'], train['Pclass']],margins=True)


# In[22]:


pd.crosstab(train['Sex'],[train['Survived'], train['Pclass']], normalize='index')


# In[26]:


pd.crosstab(train['Sex'],train['Pclass'], values=train['Survived'], aggfunc=np.average)


# In[27]:


pd.crosstab(train['Sex'],train['Pclass'], values=train['Survived'], aggfunc=np.average, margins=True)


# In[6]:


train[['Survived','Sex','Pclass']].pivot_table(index=['Sex','Pclass'])


# In[5]:


train[['Fare','Sex','Pclass']].pivot_table(index=['Sex','Pclass'])


# In[18]:


age = pd.cut(train['Age'], [0, 18, 80])
train.pivot_table('Survived', ['Sex', age], 'Pclass')


# In[19]:


fare = pd.qcut(train['Fare'], 2)
train.pivot_table('Survived', ['Sex', age], [fare, 'Pclass'])


# ## Logistic Regression
# 
# ![image.png](./images/logistic.png)

# 对数几率函数 （一种Sigmoid函数）
# $$y = \frac{1}{1+e^{-z}} = \frac{1}{1+e^{-(w^Tx + b)}}$$
# 
# 对数几率 log odds
# $$logit = ln \frac{y}{1-y} = w^Tx + b$$

# In[4]:


# load data with pandas
import pandas as pd
import statsmodels.api as sm

train = pd.read_csv('../data/tatanic_train.csv',sep = ",", header=0)


# ### Data Cleaning

# In[5]:


# dealing with missing data
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Fare"] = train["Fare"].fillna(train["Fare"].median())


# In[6]:


# Convert the male and female groups to integer form
train['Sex'] = train['Sex'].fillna('ffill')
train['female'] = [1 if i =='female' else 0 for i in train['Sex']]


# In[7]:


#Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna('S')
train['embarked_c'] = [1 if i =='C' else 0 for i in train['Embarked']]
train['embarked_q'] = [1 if i =='Q' else 0 for i in train['Embarked']]


# In[8]:


logit = sm.Logit(train['Survived'],  
                 train[['female', 'Fare', 'Age','Pclass', 'embarked_c', 'embarked_q' ]])
result = logit.fit()
result.summary()


# ![image.png](./images/end.png)
