#!/usr/bin/env python
# coding: utf-8

# # 使用Surprise构建推荐系统

# In[1]:


import pandas as pd
from surprise import Dataset
from surprise import Reader


# In[2]:


critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
      'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
      'The Night Listener': 3.0},
     'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
      'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
      'You, Me and Dupree': 3.5},
     'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
      'Superman Returns': 3.5, 'The Night Listener': 4.0},
     'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
      'The Night Listener': 4.5, 'Superman Returns': 4.0,
      'You, Me and Dupree': 2.5},
     'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
      'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
      'You, Me and Dupree': 2.0},
     'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
      'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
     'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}


# In[7]:


dat = []
for i in critics:
    for j in critics[i]: 
        dat.append([i, j, critics[i][j]])

df = pd.DataFrame(dat, columns = ['user', 'item', 'rating'])


# In[20]:


df.head()


# In[9]:


from surprise import Dataset
from surprise import Reader

# Loads Pandas dataframe
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)


# In[11]:


from surprise import KNNBasic

sim_options_item = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
                   }
sim_options_user = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
                   }
algo_userCF = KNNBasic(sim_options=sim_options_user)
algo_itemCF = KNNBasic(sim_options=sim_options_item)


# In[12]:


from surprise.model_selection import cross_validate

# Run 5-fold cross-validation and print results.
cross_validate(algo_userCF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[22]:


algo_userCF.predict('Toby', 'The Night Listener')


# In[23]:


algo_userCF.predict('Toby', 'Lady in the Water')


# In[24]:


algo_userCF.predict('Toby', 'Just My Luck')


# In[13]:


# Run 5-fold cross-validation and print results.
cross_validate(algo_itemCF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[ ]:




