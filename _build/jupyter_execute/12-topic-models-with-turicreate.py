#!/usr/bin/env python
# coding: utf-8

# 
# # 使用Turicreate建立主题模型
# 

# In[2]:


import turicreate as tc


# Download Data: <del>http://select.cs.cmu.edu/code/graphlab/datasets/wikipedia/wikipedia_raw/w15

# In[6]:


sf = tc.SFrame.read_csv("/Users/datalab/bigdata/cjc/w15", 
                              header=False)


# In[7]:


sf


# ## Transformations

# https://dato.com/learn/userguide/text/analysis.html

# In[8]:


dir(sf['X1']) 


# In[9]:


bow = sf['X1']._count_words() 


# In[10]:


type(sf['X1'])


# In[11]:


type(bow)


# In[12]:


bow.dict_has_any_keys(['limited'])


# In[13]:


bow.dict_values()[0][:20]


# In[14]:


sf


# In[15]:


sf['bow'] = bow


# In[16]:


sf


# In[17]:


type(sf['bow'])


# In[18]:


len(sf['bow'])


# In[21]:


list(sf['bow'][0].items())[:3]


# In[22]:


sf['tfidf'] = tc.text_analytics.tf_idf(sf['X1'])


# In[23]:


sf


# In[24]:


list(sf['tfidf'][0].items())[:5]


# ## Text Cleaning

# In[25]:


docs = sf['bow'].dict_trim_by_values(2)


# In[27]:


docs = docs.dict_trim_by_keys(
    tc.text_analytics.stop_words(),
    exclude=True)


# ## Topic modeling

# In[28]:


help(tc.topic_model.create)


# In[29]:


help(tc.text_analytics.random_split)


# In[30]:


train, test = tc.text_analytics.random_split(docs, .8)


# In[31]:


m = tc.topic_model.create(train, 
                                num_topics=100,       # number of topics
                                num_iterations=100,   # algorithm parameters
                                alpha=None, beta=.1)  # hyperparameters


# In[32]:


results = m.evaluate(test)
print(results['perplexity'])


# In[33]:


m


# In[34]:


m.get_topics()


# In[35]:


help(m.get_topics)


# In[37]:


topics = m.get_topics(num_words=10).unstack(['word','score'],                                 new_column_name='topic_words')['topic_words'].apply(lambda x: x.keys())
for topic in topics:
    print(topic)


# In[40]:


help(m)


# In[41]:


def print_topics(m):
    topics = m.get_topics(num_words=5)
    topics = topics.unstack(['word','score'], new_column_name='topic_words')['topic_words']
    topics = topics.apply(lambda x: x.keys())
    for topic in topics:
        print(topic)
print_topics(m)


# > pred = m.predict(another_data) 
# 
# > pred = m.predict(another_data, output_type='probabilities')

# ### Initializing from other models

# In[46]:


dir(m)


# In[48]:


m.vocabulary


# In[50]:


m.topics


# In[51]:


m2 = tc.topic_model.create(docs,
                                 num_topics=100,
                                 initial_topics=m.topics)


# ### Seeding the model with prior knowledge

# In[52]:


associations = tc.SFrame()
associations['word'] = ['recognition']
associations['topic'] = [0]


# In[53]:


m2 = tc.topic_model.create(docs,
                                 num_topics=20,
                                 num_iterations=50,
                                 associations=associations, 
                                 verbose=False)


# In[54]:


m2.get_topics(num_words=10)


# In[55]:


print_topics(m2)


# ## 阅读材料
# 
# https://apple.github.io/turicreate/docs/userguide/text/
