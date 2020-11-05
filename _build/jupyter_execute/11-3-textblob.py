#!/usr/bin/env python
# coding: utf-8

# # 利用textblob进行情感分析
# 

#  安装textblob
# https://github.com/sloria/TextBlob
#     
# > pip install -U textblob
# 
# > python -m textblob.download_corpora

# In[6]:


from textblob import TextBlob

text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

blob = TextBlob(text)

for sentence in blob.sentences:
    print(sentence, sentence.sentiment.polarity, sentence.sentiment.subjectivity)


# In[7]:


from textblob.classifiers import NaiveBayesClassifier

train=[
        ('I love this car','pos'),
        ('This view is amazing','pos'),
        ('I feel great','pos'),
        ('I am so excited about the concert','pos'),
        ("He is my best friend",'pos'),
        ('I do not like this car','neg'),
        ('This view is horrible','neg'),
        ("I feel tired this morning",'neg'),
        ('I am not looking forward to the concert','neg'),
        ('He is an annoying enemy','neg')
]

test=[
        ('I feel happy this morning','pos'),
        ('Oh I love my friend','pos'),
        ('not like that man','neg'),
        ("this hourse is not great",'neg'),
        ('your song is annoying','neg')
]

cl=NaiveBayesClassifier(train)

for sentence in test:
    print(sentence[0],'：',cl.classify(sentence[0]))


# ## SnowNLP
# 
# https://pypi.org/project/snownlp/
# 
# https://github.com/isnowfy/snownlp
# 
# SnowNLP介绍：是一个python写的类库，可以方便的处理中文文本内容，是受到了TextBlob的启发而写的，由于现在大部分的自然语言处理库基本都是针对英文的，于是写了一个方便处理中文的类库，并且和TextBlob不同的是，这里没有用NLTK，所有的算法都是自己实现的，并且自带了一些训练好的字典。
# 

# In[3]:


pip install snownlp


# In[8]:


from snownlp import SnowNLP

s = SnowNLP(u'这个东西真心很赞')

s.sentiments    # 0.9769663402895832 positive的概率


# In[ ]:




