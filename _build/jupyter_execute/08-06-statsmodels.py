#!/usr/bin/env python
# coding: utf-8

# # Statistical Modeling with Python
# 
# `statsmodels` is better suited for traditional stats

# In[122]:


# the statsmodels.api uses numpy array notation
# statsmodels.formula.api use formula notation (similar to R's formula notation)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ## A minimal OLS example

# Four pairs of points

# In[2]:


x = np.array([1,2,3,4])
y = np.array([2,6,4,8])


# In[3]:


plt.scatter(x,y, marker = '.')
plt.xlim(0,5)
plt.ylim(0,10)
plt.show()


# In[6]:


# make a dataframe of our data
d = pd.DataFrame({'x':x, 'y':y})
print(d)


# Seaborn lmplot

# In[7]:


sns.lmplot(x = 'x', y = 'y', data = d)


# ## Formula notation with statsmodels
# use statsmodels.formula.api (often imported as smf)

# In[8]:


# data is in a dataframe
model = smf.ols('y ~ x', data = d)


# In[10]:


# estimation of coefficients is not done until you call fit() on the model
results = model.fit()


# In[12]:



print(results.summary()) 


# Using the abline_plot function for plotting the results

# In[14]:


sm.graphics.abline_plot(model_results = results)
plt.scatter(d.x, d.y)

plt.xlim(0,5)
plt.ylim(0,10)

plt.show()


# Generating an anova table

# In[17]:


print(sm.stats.anova_lm(results))


# Making predictions

# In[16]:


results.predict({'x' : 2})


# ## numpy array notation
# similar to sklearn's notation

# In[18]:


print(x)


# In[19]:


X = sm.add_constant(x)  
# need to add a constant for the intercept term.
# because we are using the numpy notation, we use sm rather than smf


# In[20]:


print(X)


# $$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$
# 
# $$\mathbf{\hat{Y}} = \boldsymbol{\beta} \mathbf{X}$$
# 

# In[21]:


# OLS is capitalized in the numpy notation
model2 = sm.OLS(y, X)  


# In[22]:


results2 = model2.fit()


# In[23]:


print(results2.summary())


# ## OLS solution
# 
# $$(X^TX)^{-1}X^TY$$

# In[24]:


X


# In[28]:


np.linalg.inv(X.T @ X) @ (X.T @ y)

