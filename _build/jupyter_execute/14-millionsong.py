#!/usr/bin/env python
# coding: utf-8

# 
# # 使用Turicreate进行音乐推荐
# 
# 

# In[2]:


get_ipython().system('pip install turicreate')


# In[3]:


import turicreate as tc


#  下载数据
# http://s3.amazonaws.com/dato-datasets/millionsong/10000.txt
# 

# In[4]:


#train_file = 'http://s3.amazonaws.com/dato-datasets/millionsong/10000.txt'
train_file = '/Users/datalab/bigdata/cjc/millionsong/song_usage_10000.txt'
sf = tc.SFrame.read_csv(train_file, header=False, delimiter='\t', verbose=False)
sf = sf.rename({'X1':'user_id', 'X2':'music_id', 'X3':'rating'})


# In[5]:


train_set, test_set = sf.random_split(0.8, seed=1)


# In[6]:


popularity_model = tc.popularity_recommender.create(train_set, 
                                                    'user_id', 'music_id', 
                                                    target = 'rating')


# In[7]:


item_sim_model = tc.item_similarity_recommender.create(train_set, 
                                                       'user_id', 'music_id', 
                                                       target = 'rating', 
                                                       similarity_type='cosine')


# In[8]:


factorization_machine_model = tc.recommender.factorization_recommender.create(train_set, 
                                                                              'user_id', 'music_id',
                                                                              target='rating')


# In[9]:


len(train_set)


# In[10]:


result = tc.recommender.util.compare_models(test_set, 
                                            [popularity_model, item_sim_model, factorization_machine_model],
                                            user_sample=.5, skip_set=train_set)


# In[12]:


K = 10
users = gl.SArray(sf['user_id'].unique().head(100))


# In[13]:


recs = item_sim_model.recommend(users=users, k=K)
recs.head()


# In[ ]:




