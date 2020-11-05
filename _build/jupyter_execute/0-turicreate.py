#!/usr/bin/env python
# coding: utf-8

# 
# # Turicreate: Departure from Graphlab
# 
# Turi Create simplifies the development of custom machine learning models. 
# 
# 
# https://github.com/apple/turicreate
# 

# You don't have to be a machine learning expert to add
# - recommendations, 
# - object detection, 
# - image classification, 
# - image similarity  
# - activity classification 
# 
# to your app.
#     
# - https://apple.github.io/turicreate/docs/userguide/
# - https://apple.github.io/turicreate/docs/api/index.html

# 
# ## pip install -U turicreate

# In[1]:


import turicreate as tc
actions = tc.SFrame.read_csv('/Users/datalab/bigdata/cjc/ml-1m/ratings.dat', delimiter='\n', 
                                header=False)['X1'].apply(lambda x: x.split('::')).unpack()
for col in actions.column_names():
    actions[col] = actions[col].astype(int)
actions = actions.rename({'X.0': 'user_id', 'X.1': 'movie_id', 'X.2': 'rating', 'X.3': 'timestamp'})
#actions.save('ratings')

