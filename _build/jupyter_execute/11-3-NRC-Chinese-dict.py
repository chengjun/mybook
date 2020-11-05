#!/usr/bin/env python
# coding: utf-8

# # 基于NRC字典的情感分析
# 
# NRC词典为加拿大国家研究委员会信息技术研究所(Institute for Information Technology, National Research Council Canada. )组织制作的基于众包方式标注出的词典。
# 
# Mohammad, Saif M., and Peter D. Turney. "Crowdsourcing a word–emotion association lexicon." Computational Intelligence 29, no. 3 (2013): 436-465.
# 
# http://sentiment.nrc.ca/lexicons-for-research/ 

# THE SENTIMENT AND EMOTION LEXICONS (click to download the complete bundle, 100Mb)
# 
# Individual lexicons for download:
# - Manually created lexicons:NRC Emotion LexiconNRC Emotion Intensity LexiconNRC Valence, Arousal, and Dominance (VAD) LexiconNRC Sentiment Composition Lexicons (SCL-NMA, SCL-OPP, SemEval-2015 English Twitter, SemEval-2016 Arabic Twitter)NRC Word-Colour Association Lexicon
# - Automatically generated lexicons:NRC Hashtag Emotion LexiconNRC Hashtag Sentiment LexiconNRC Hashtag Affirmative Context Sentiment Lexicon and NRC Hashtag Negated Context Sentiment LexiconNRC Emoticon Lexicon (a.k.a. Sentiment140 Lexicon)NRC Emoticon Affirmative Context Lexicon and NRC Emoticon Negated Context Lexicon
# 

# Vosoughi et al., The spread of true and false news online. Science  359, 1146–1151 (2018) 9 March 2018
# 
# 
# > We categorized the emotion in the replies by using the leading lexicon curated by the National Research Council Canada (NRC), which provides a comprehensive list of ~140,000 English words and their associations with eight emotions 
# 
# False news was more novel than true news, which suggests that people were more likely to share novel information. Whereas false stories inspired fear, disgust, and surprise in replies, true stories inspired anticipation, sadness, joy, and trust. 
# 
# 
# ![image.png](images/NRC.png)

# In[1]:


import pandas as pd


# In[2]:


lexion_df = pd.read_excel('./Textmining/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx')
lexion_df.head()


# In[3]:


lexion_df.columns.tolist()


# In[4]:


chinese_df = lexion_df[['Chinese (Simplified) (zh-CN)', 'Positive', 'Negative', 
                 'Anger','Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']]
chinese_df.head()


# In[5]:


# 构建情感词列表

Positive, Negative, Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust= [[] for i in range(10)]
for idx, row in chinese_df.iterrows():
    if row['Positive']==1:
        Positive.append(row['Chinese (Simplified) (zh-CN)'])
    if row['Negative']==1:
        Negative.append(row['Chinese (Simplified) (zh-CN)'])
    if row['Anger']==1:
        Anger.append(row['Chinese (Simplified) (zh-CN)'])
    if row['Anticipation']==1:
        Anticipation.append(row['Chinese (Simplified) (zh-CN)'])
    if row['Disgust']==1:
        Disgust.append(row['Chinese (Simplified) (zh-CN)'])
    if row['Fear']==1:
        Fear.append(row['Chinese (Simplified) (zh-CN)'])
    if row['Joy']==1:
        Joy.append(row['Chinese (Simplified) (zh-CN)'])
    if row['Sadness']==1:
        Sadness.append(row['Chinese (Simplified) (zh-CN)'])
    if row['Surprise']==1:
        Surprise.append(row['Chinese (Simplified) (zh-CN)'])
    if row['Trust']==1:
        Trust.append(row['Chinese (Simplified) (zh-CN)'])

print('词语列表构建完成')


# In[6]:


Anger[:10]


# In[12]:


positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, trust = [0 for i in range(10)]
[positive, negative]


# In[7]:


import jieba
import time


def emotion_caculate(text):
    positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, trust = [0 for i in range(10)]
    
    wordlist = jieba.lcut(text)
    wordset = set(wordlist)
    wordfreq = []
    for word in wordset:
        freq = wordlist.count(word)
        if word in Positive:
            positive+=freq
        if word in Negative:
            negative+=freq
        if word in Anger:
            anger+=freq  
        if word in Anticipation:
            anticipation+=freq
        if word in Disgust:
            disgust+=freq
        if word in Fear:
            fear+=freq
        if word in Joy:
            joy+=freq
        if word in Sadness:
            sadness+=freq
        if word in Surprise:
            surprise+=freq
        if word in Trust:
            trust+=freq
            
    emotion_info = {
        'positive': positive,
        'negative': negative,
        'anger': anger,
        'anticipation': anticipation,
        'disgust': disgust,
        'fear':fear,
        'joy':joy,
        'sadness':sadness,
        'surprise':surprise,
        'trust':trust,
        'length':len(wordlist)
    }
    indexs = ['length', 'positive', 'negative', 'anger', 'anticipation','disgust','fear','joy','sadness','surprise','trust']
    return pd.Series(emotion_info, index=indexs)
        


# In[8]:


emotion_caculate(text='这个国家再对这些制造假冒伪劣食品药品的人手软的话，那后果真的会相当糟糕。坐牢？从快判个死刑')


# In[ ]:




