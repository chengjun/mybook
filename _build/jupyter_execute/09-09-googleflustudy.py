#!/usr/bin/env python
# coding: utf-8

# # Forecasting and nowcasting with Google Flu Trends
# 
# Rather than predicting the future, nowcasting attempts to use ideas from forecasting to measure the current state of the world; it attempts to “predict the present” (Choi and Varian 2012). Nowcasting has the potential to be especially useful to governments and companies that require timely and accurate measures of the world.
# 
# https://www.bitbybitbook.com/en/1st-ed/observing-behavior/strategies/forecasting/
# 
# ## Predicting the Present with Google Trends
# 
# HYUNYOUNG CHOI and HAL VARIAN
# 
# Google, Inc., California, USA
# 
# > In this paper we show how to use search engine data to forecast near-term values of economic indicators. Examples include automobile sales, unemployment claims, travel destination planning
# and consumer confidence.
# 
# Choi, Hyunyoung, and Hal Varian. 2012. “Predicting the Present with Google Trends.” Economic Record 88 (June):2–9. https://doi.org/10.1111/j.1475-4932.2012.00809.x.
# 
# 

# ## Detecting influenza epidemics using search engine query data
# 
# - Jeremy Ginsberg et al. (2009) Detecting influenza epidemics using search engine query data. Nature. 457, pp:1012–1014 https://www.nature.com/articles/nature07634#Ack1 
# 
# - **Google Query Data** https://static-content.springer.com/esm/art%3A10.1038%2Fnature07634/MediaObjects/41586_2009_BFnature07634_MOESM271_ESM.xls Query fractions for the top 100 search queries, sorted by mean Z-transformed correlation with CDC-provided ILI percentages across the nine regions of the United States. (XLS 5264 kb)
# 
# - **CDC’s ILI Data**. We use the weighted version of CDC’s ILI activity level as the estimation target (available at gis.cdc.gov/grasp/fluview/fluportaldashboard.html). The weekly revisions of CDC’s ILI are available at the CDC website for all recorded seasons (from week 40 of a given year to week 20 of the subsequent year). Click **Download Data** to get the data.
# 
# ![image.png](./img/09CDC.png)
# 
# For example, ILI report revision at week 50 of season 2012–2013 is available at www.cdc.gov/flu/weekly/weeklyarchives2012-2013/data/senAllregt50.htm; ILI report revision at week 9 of season 2014–2015 is available at www.cdc.gov/flu/weekly/weeklyarchives2014-2015/data/senAllregt09.html.
# 

# In[101]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold


# In[42]:


df = pd.read_excel('41586_2009_BFnature07634_MOESM271_ESM.xls', sheet_name=1, header = 1)
df.head()


# In[22]:


plt.plot(df['Date'], df['United States']);


# In[23]:


plt.plot(df['Date'], df['Mid-Atlantic Region']);


# Figure 1: An evaluation of how many top-scoring queries to include in the ILI-related query fraction.
# 
# ![image.png](./img/09gft1.png)
# 
# Maximal performance at estimating out-of-sample points during cross-validation was obtained by summing the top 45 search queries. A steep drop in model performance occurs after adding query 81, which is ‘oscar nominations’.

# In[43]:


# Combine 45 queries
dict = {'date': df['Date'].tolist()}
for i in range(1, 46):
    df = pd.read_excel('41586_2009_BFnature07634_MOESM271_ESM.xls', sheet_name=i, header = 1)
    dict['query'+str(i)] = df['United States'].tolist()
dat = pd.DataFrame.from_dict(dict)
dat.head()


# ## The Parable of Google Flu: Traps in Big Data Analysis
# 
# David Lazer*, Ryan Kennedy, Gary King, Alessandro Vespignani
# 
# 
# Science 14 Mar 2014: Vol. 343, Issue 6176, pp. 1203-1205 DOI: 10.1126/science.1248506
# 
# In February 2013, Google Flu Trends (GFT) made headlines but not for a reason that Google executives or the creators of the flu tracking system would have hoped. Nature reported that GFT was predicting more than double the proportion of doctor visits for influenza-like illness (ILI) than the Centers for Disease Control and Prevention (CDC), which bases its estimates on surveillance reports from laboratories across the United States (1, 2). This happened despite the fact that GFT was built to predict CDC reports. Given that GFT is often held up as an exemplary use of big data (3, 4), what lessons can we draw from this error?
# 
# https://science.sciencemag.org/content/343/6176/1203.summary
# 
# **Data & Code**
# 
# https://science.sciencemag.org/content/sci/suppl/2014/03/12/343.6176.1203.DC1/1248506.Lazer.SM.revision1.pdf
# 
# 
# Lazer, David; Kennedy, Ryan; King, Gary; Vespignani, Alessandro, 2014, "Replication data for: The Parable of Google Flu: Traps in Big Data Analysis", https://doi.org/10.7910/DVN/24823, Harvard Dataverse

# In[51]:


# merge the ILI data
# cflu is CDC % ILI
dat2 = pd.read_csv('../GFT2.0/parable/ParableOfGFT(Replication).csv')
dat3 = dat2[['date', 'cflu']]
data = pd.merge(dat, dat3, how='right', on='date')
data.head()


# In[89]:


#data.to_csv('gft_ili_us.csv', index = False)


# In[81]:


# filter data
data = data[data['query1'].notna()]
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].dt.date


# In[85]:


plt.plot(data['date'], data['query1'], label = 'query1')
plt.plot(data['date'], data['query2'], label = 'query2')
plt.plot(data['date'], data['query3'], label = 'query3')
plt.plot(data['date'], data['cflu'],  label = 'CDC ILI')
plt.legend()
plt.show()


# Using this ILI-related query fraction as the explanatory variable, we fit a final linear model to weekly ILI percentages between 2003 and 2007 for all nine regions together, thus obtaining a single, region-independent coefficient. The model was able to obtain a good fit with CDC-reported ILI percentages, with a mean correlation of 0.90 (min = 0.80, max = 0.96, n = 9 regions; Fig. 2).
# 
# **Figure 2**: A comparison of model estimates for the mid-Atlantic region (black) against CDC-reported ILI percentages (red), including points over which the model was fit and validated.
# 
# ![image.png](./img/09gft2.png)
# 
# A correlation of 0.85 was obtained over 128 points from this region to which the model was fit, whereas a correlation of 0.96 was obtained over 42 validation points. Dotted lines indicate 95% prediction intervals. The region comprises New York, New Jersey and Pennsylvania.

# **Figure 3**: ILI percentages estimated by our model (black) and provided by the CDC (red) in the **mid-Atlantic region**, showing data available at four points in the 2007-2008 influenza season.
# 
# ![image.png](./img/09gft3.png)

# In[90]:


for i in range(1, 8):
    data["lag_{}".format(i)] = data['cflu'].shift(i)
print("done")
data=data.fillna(0)


# In[131]:


y = data['cflu']
date = data['date']
X = data.drop(['cflu', 'date'], axis = 1)


# In[96]:


len(y)


# In[132]:


y


# In[133]:


X


# In[99]:


N = 50
X_train = X.iloc[:N,]
X_test = X.iloc[N:,]
y_train = y[:N]
y_test = y[N:]

# 利用弹性网络
from sklearn.model_selection import cross_val_score
cv_model = ElasticNetCV(l1_ratio=0.5, eps=1e-3, n_alphas=200, fit_intercept=True, 
                        normalize=True, precompute='auto', max_iter=200, tol=0.006, cv=10, 
                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=0)

# 训练模型              
cv_model.fit(X_train, y_train)

# 计算最佳迭代次数、alpha和ratio
print('最佳 alpha: %.8f'%cv_model.alpha_)
print('最佳 l1_ratio: %.3f'%cv_model.l1_ratio_)
print('迭代次数 %d'%cv_model.n_iter_)


# In[102]:


# 输出结果
y_train_pred = cv_model.predict(X_train)
y_pred = cv_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_mse = mean_squared_error(y_train_pred, y_train)
test_mse = mean_squared_error(y_pred, y_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print('Train RMSE: %.4f' % train_rmse)
print('Test RMSE: %.4f' % test_rmse)


# In[103]:


import datetime
plt.style.use('ggplot')

plt.rcParams.update({'figure.figsize': (15, 5)})

plt.plot(date, y)
plt.plot(date[N:], y_pred)

plt.show()


# However, this apparent success story eventually turned into an embarrassment. 
# 
# 1. Google Flu Trends with all its data, machine learning, and powerful computing did not dramatically outperform a simple and easier-to-understand heuristic. This suggests that when evaluating any forecast or nowcast, it is important to compare against a baseline.
# 2. Its ability to predict the CDC flu data was prone to short-term failure and long-term decay because of drift and algorithmic confounding. 
# 
# These two caveats complicate future nowcasting efforts, but they do not doom them. In fact, by using more careful methods, Lazer et al. (2014) and Yang, Santillana, and Kou (2015) were able to avoid these two problems.

# ## References
# 
# 
# - Goel, Sharad, Jake M. Hofman, Sébastien Lahaie, David M. Pennock, and Duncan J. Watts. 2010. “Predicting Consumer Behavior with Web Search.” Proceedings of the National Academy of Sciences of the USA 107 (41):17486–90. https://doi.org/10.1073/pnas.1005962107.
# 
# - Yang, Shihao, Mauricio Santillana, and S. C. Kou. 2015. “Accurate Estimation of Influenza Epidemics Using Google Search Data via ARGO.” Proceedings of the National Academy of Sciences of the USA 112 (47):14473–8. https://doi.org/10.1073/pnas.1515373112.
# 
# - Lazer, David, Ryan Kennedy, Gary King, and Alessandro Vespignani. 2014. “The Parable of Google Flu: Traps in Big Data Analysis.” Science 343 (6176):1203–5. https://doi.org/10.1126/science.1248506.

# ## Learning by Doing
# 
# https://github.com/JEstebanMejiaV/The.Analytics.Edge/blob/352d59a27d2c376f268b1dbdf838e9ee77989d36/Unit%202%20-%20Linear%20Regression/Detecting%20Flu%20Epidemics%20via%20Search%20Engine%20Query%20Data.ipynb

# In[86]:


dat = pd.read_csv('FluTrain.csv')
dat.head()


# In[88]:


dat['Week']


# In[104]:


for i in range(1, 8):
    dat["lag_{}".format(i)] = dat['ILI'].shift(i)
print("done")
dat=dat.fillna(0)


# In[128]:


y = dat['ILI']
week = dat['Week']
week = [i[:10] for i in week.tolist()]
week = pd.to_datetime(week)
X = dat.drop(['ILI', 'Week'], axis = 1)


# In[130]:


y


# In[109]:


X


# In[110]:


N = 100
X_train = X.iloc[:N,]
X_test = X.iloc[N:,]
y_train = y[:N]
y_test = y[N:]

# 利用弹性网络
from sklearn.model_selection import cross_val_score
cv_model = ElasticNetCV(l1_ratio=0.5, eps=1e-3, n_alphas=200, fit_intercept=True, 
                        normalize=True, precompute='auto', max_iter=200, tol=0.006, cv=10, 
                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=0)

# 训练模型              
cv_model.fit(X_train, y_train)

# 计算最佳迭代次数、alpha和ratio
print('最佳 alpha: %.8f'%cv_model.alpha_)
print('最佳 l1_ratio: %.3f'%cv_model.l1_ratio_)
print('迭代次数 %d'%cv_model.n_iter_)


# In[111]:


# 输出结果
y_train_pred = cv_model.predict(X_train)
y_pred = cv_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_mse = mean_squared_error(y_train_pred, y_train)
test_mse = mean_squared_error(y_pred, y_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print('Train RMSE: %.4f' % train_rmse)
print('Test RMSE: %.4f' % test_rmse)


# In[129]:


plt.plot(week, y)
plt.plot(week[N:], y_pred)
plt.show()


# In[ ]:




