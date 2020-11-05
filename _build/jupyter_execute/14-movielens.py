#!/usr/bin/env python
# coding: utf-8

# 
# # 使用Turicreate进行电影推荐
# 
# 

# In[1]:


import turicreate as tc
# set canvas to show sframes and sgraphs in ipython notebook
# import matplotlib.pyplot as plt
# %matplotlib inline


# In[7]:


# download data from: http://files.grouplens.org/datasets/movielens/ml-1m.zip


# In[2]:


data = tc.SFrame.read_csv('/Users/datalab/bigdata/cjc/ml-1m/ratings.dat', delimiter='\n', 
                                header=False)['X1'].apply(lambda x: x.split('::')).unpack()
for col in data.column_names():
    data[col] = data[col].astype(int)
data = data.rename({'X.0': 'user_id', 'X.1': 'movie_id', 'X.2': 'rating', 'X.3': 'timestamp'})
#data.save('ratings')


# In[3]:


users = tc.SFrame.read_csv('/Users/datalab/bigdata/cjc/ml-1m/users.dat', delimiter='\n', 
                                 header=False)['X1'].apply(lambda x: x.split('::')).unpack()
users = users.rename({'X.0': 'user_id', 'X.1': 'gender', 'X.2': 'age', 'X.3': 'occupation', 'X.4': 'zip-code'})
users['user_id'] = users['user_id'].astype(int)
users.save('users')


# In[28]:


#items = tc.SFrame.read_csv('/Users/datalab/bigdata/ml-1m/movies.dat', delimiter='\n', header=False)#['X1'].apply(lambda x: x.split('::')).unpack()
# items = items.rename({'X.0': 'movie_id', 'X.1': 'title', 'X.2': 'genre'})
# items['movie_id'] = items['movie_id'].astype(int)
# items.save('items')


# In[4]:


data


# In[31]:


#items


# In[5]:


users


# In[7]:


#data = data.join(items, on='movie_id')


# In[33]:


#data


# In[6]:


train_set, test_set = data.random_split(0.95, seed=1)


# In[7]:


m = tc.recommender.create(train_set, 'user_id', 'movie_id', 'rating')


# In[36]:


m


# In[8]:


m2 = tc.item_similarity_recommender.create(train_set, 
                                                 'user_id', 'movie_id', 'rating',
                                 similarity_type='pearson')


# In[9]:


m2


# In[10]:


result = tc.recommender.util.compare_models(test_set, 
                                                  [m, m2],
                                            user_sample=.5, skip_set=train_set)


# ## Getting similar items

# In[41]:


m.get_similar_items([1287])  # movie_id is Ben-Hur


# In[19]:


help(m.get_similar_items)


# 'score' gives the similarity score of that item

# In[42]:


# m.get_similar_items([1287]).join(items, on={'similar': 'movie_id'}).sort('rank')


# ## Making recommendations

# In[43]:


recs = m.recommend()


# In[44]:


recs


# In[45]:


data[data['user_id'] == 4]


# In[46]:


# m.recommend(users=[4], k=20).join(items, on='movie_id')


# ## Recommendations for new users

# In[47]:


recent_data = tc.SFrame()
recent_data['movie_id'] = [30, 1000, 900, 883, 251, 200, 199, 180, 120, 991, 1212] 
recent_data['user_id'] = 99999
recent_data['rating'] = [2, 1, 3, 4, 0, 0, 1, 1, 1, 2, 3]
recent_data


# In[48]:


m2.recommend(users=[99999], new_observation_data=recent_data)#.join(items, on='movie_id').sort('rank')


# ## Saving and loading models

# In[ ]:


m.save('my_model')


# In[ ]:


m_again = graphlab.load_model('my_model')


# In[ ]:


m_again


# In[ ]:




