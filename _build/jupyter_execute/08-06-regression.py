#!/usr/bin/env python
# coding: utf-8

# 
# # Linear Regression
# 
# 
# ![image.png](./images/author.png)

# ## The Model
# 
# $$y_i = \beta x_i + \alpha + \epsilon_i$$
# 
# 
# where 
# 
# - $y_i$ is the number of minutes user i spends on the site daily, 
# - $x_i$ is the number of friends user i has
# - $\alpha$ is the constant when x = 0.
# - $ε_i$ is a (hopefully small) error term representing the fact that there are other factors not accounted for by this simple model.

# ## Least Squares Fit
# 
# 最小二乘法
# 
# $$ y_i = X_i^T w$$
# 
# The constant could be represent by 1 in X
# 
# The squared error could be written as: 
# 
# $$ \sum_{i = 1}^m (y_i -X_i^T w)^2 $$

# If we know $\alpha$ and $\beta$, then we can make predictions.
# 
# Since we know the actual output $y_i$ we can compute the error for each pair.
# 
# Since the negative errors cancel out with the positive ones, we use squared errors.
# 
# The least squares solution is to choose the $\alpha$ and $\beta$ that make **sum_of_squared_errors** as small as possible.

# 
# 
# $$ y_i = \alpha + \beta x_i + \varepsilon_i $$
# 
# $$ \hat\varepsilon_i =y_i-a -\beta x_i $$
# 
# $$ \text{Find }\min_{\alpha,\, \beta} Q(\alpha, \beta), \quad \text{for } Q(\alpha, \beta) = \sum_{i=1}^n\hat\varepsilon_i^{\,2} = \sum_{i=1}^n (y_i -\alpha - \beta x_i)^2\ $$
# 

# ### 均方误差的几何意义：欧式距离
# 
# 最小二乘法就是试图找到一条直线，是所有样本到直线的**欧氏距离**之和最小。
# 
# ![image.png](./images/least_squares.png)

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf 
import matplotlib
matplotlib.style.use('fivethirtyeight')


# In[3]:


matplotlib.style.available


# In[29]:


num_friends_good = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
daily_minutes_good = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]
alpha, beta = 22.9475, 0.90386


# In[6]:


plt.scatter(num_friends_good, daily_minutes_good)
plt.plot(num_friends_good, [alpha + beta*i for i in num_friends_good], 'b-')
plt.xlabel('# of friends', fontsize = 20)
plt.ylabel('minutes per day', fontsize = 20)
plt.title('linear regression', fontsize = 20)
plt.show()


# Of course, we need a better way to figure out how well we’ve fit the data than staring at the graph. 
# 
# A common measure is the coefficient of determination (or R-squared), which measures the fraction of the total variation in the dependent variable that is captured by the model.

# ## The Matrix Method
# 
# 
# $$ y_i = X_i^T w$$
# 
# The constant could be represent by 1 in X
# 
# The squared error could be written as: 
# 
# $$ \sum_{i = 1}^m (y_i -X_i^T w)^2 $$

# ![image.png](./images/regression.png)

# We can also write this in matrix notation as $(y-Xw)^T(y-Xw)$. 
# 
# If we take the derivative of this with respect to $w$, we’ll get 
# 
# $$X^T(y-Xw)$$ 
# 
# We can set this to zero and solve for w to get the following equation:
# 
# $$\hat w = (X^T X)^{-1}X^T y$$

# In[7]:


# https://github.com/computational-class/machinelearninginaction/blob/master/Ch08/regression.py
import pandas as pd
import random

dat = pd.read_csv('../data/ex0.txt', sep = '\t', names = ['x1', 'x2', 'y'])
dat['x3'] = [yi*.3 + .5*random.random() for yi in dat['y']]
dat.head()


# In[8]:


from numpy import mat, linalg, corrcoef

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


# In[9]:


xs = [[dat.x1[i], dat.x2[i], dat.x3[i]] for i in dat.index]
y = dat.y
print(xs[:2])
ws = standRegres(xs, y)
print(ws)


# In[10]:


xMat=mat(xs)
yMat=mat(y)
#yHat = xMat*ws
xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy*ws


# In[11]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
ax.plot(xCopy[:,1],yHat, 'r-')
plt.ylim(0, 5)
plt.show()


# In[108]:


yHat = xMat*ws
corrcoef(yHat.T, yMat)


# ## Regression with Statsmodels
# 
# http://www.statsmodels.org/stable/index.html
# 
# statsmodels is a Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration.

# In[21]:


dat = pd.read_csv('../data/ex0.txt', sep = '\t', names = ['x1', 'x2', 'y'])
dat['x3'] = [yi*.3 - .1*random.random() for yi in y]
dat.head()


# In[22]:


results = smf.ols('y ~ x2 + x3', data=dat).fit()

results.summary()


# In[23]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(results, fig = fig)
plt.show()


# In[24]:


# regression
import numpy as np
X = np.array(num_friends_good)
X = sm.add_constant(X, prepend=False)
mod = sm.OLS(daily_minutes_good, X)
res = mod.fit()
print(res.summary())


# In[27]:


fig = plt.figure(figsize=(12,6))
fig = sm.graphics.plot_partregress_grid(res, fig = fig)
plt.show()


# ## Regression towards mediocrity
# 
# The concept of regression comes from genetics and was popularized by Sir Francis Galton during the late 19th century with the publication of Regression towards mediocrity in hereditary stature. Galton observed that extreme characteristics (e.g., height) in parents are not passed on completely to their offspring.

# In[17]:


df = pd.read_csv('../data/galton.csv')
df['father_above_average'] = [i-df['father'].mean() for i in df['father']]
df['mother_above_average'] = [i-df['mother'].mean() for i in df['mother']]
df['height_more_than_father'] =df['height'] - df['father']
df['height_more_than_mother'] =df['height'] - df['mother']
df.head()


# In[18]:


import seaborn as sns
sns.set(font_scale=1.5)

g = sns.PairGrid(df, y_vars=["height"], x_vars=["father", "mother"], hue="sex", height=4)
g.map(sns.regplot)
g.add_legend();


# In[98]:


g = sns.PairGrid(df, y_vars=["height_more_than_father", "height_more_than_mother"], 
                 x_vars=["father_above_average", "mother_above_average"], hue="sex", height=4)
g .map(sns.regplot)
g.add_legend();


# In[30]:


results = smf.ols('height ~ father + mother + C(sex) + nkids', data=df).fit()
print(results.summary())


# In[31]:


results2 = smf.ols('height_more_than_father ~ father_above_average + mother_above_average + C(sex) + nkids', data=df).fit()
print(results2.summary())


# In[32]:


fig = plt.figure(figsize=(8,8))
fig = sm.graphics.plot_partregress_grid(results, fig = fig)
plt.show()


# In[33]:


fig = plt.figure(figsize=(12,12))
fig = sm.graphics.plot_partregress_grid(results2, fig = fig)
plt.show()


# ![image.png](./images/end.png)
