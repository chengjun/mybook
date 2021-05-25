#!/usr/bin/env python
# coding: utf-8

# 
# # 第八章 文本挖掘
# 
# 
# ![image.png](images/author.png)

# What can be learned from 5 million books
# 
# https://www.bilibili.com/video/BV1jJ411u7Nd
# 
# This talk by Jean-Baptiste Michel and Erez Lieberman Aiden is phenomenal. 
# 
# 
# Michel, J.-B., et al. (2011). Quantitative Analysis of Culture Using Millions of Digitized Books. Science, 331, 176–182.

# In[1]:


get_ipython().run_cell_magic('html', '', '<iframe src="//player.bilibili.com/player.html?aid=68934891&bvid=BV1jJ411u7Nd&cid=119471774&page=1" \nwidth=1000 height=600\nscrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>')


# ![](./img/books.jpg)

# 试一下谷歌图书的数据: https://books.google.com/ngrams/
#     
# 
# 数据下载： http://www.culturomics.org/home

# ##  Bag-of-words model （BOW)
# 
# Represent text as numerical feature vectors

# - We create a vocabulary of unique tokens—for example, words—from the entire set of documents.
# - We construct a feature vector from each document that contains the counts of how often each word occurs in the particular document.

# Since the unique words in each document represent only a small subset of all the
# words in the bag-of-words vocabulary, the feature vectors will consist of mostly
# zeros, which is why we call them sparse

# ![image.png](images/bow.png)

# “词袋模型”（Bag of words model）假定对于一个文本：
# - 忽略词序、语法、句法；
# - 将其仅仅看做是一个词集合或组合；
# - 每个词的出现都是独立的，不依赖于其他词是否出现。
#     - 文本任意一个位置出现某一个词汇是独立选择的，不受前面句子的影响。
# 
# 这种假设虽然对自然语言进行了简化，便于模型化。
# 
# Document-Term Matrix (DTM)
# 

# 问题：例如在新闻个性化推荐中，用户对“南京醉酒驾车事故”这个短语很感兴趣。词袋模型忽略了顺序和句法，认为用户对“南京”、“醉酒”、“驾车”和“事故”感兴趣，因此可能推荐出和“南京”、“公交车”、“事故”相关的新闻。
# 
# 解决方法: 可抽取出整个短语；或者采用高阶（2阶以上）统计语言模型。例如bigram、trigram来将词序保留下来，相当于bag of bigram和bag of trigram。

# ### Transforming words into feature vectors

# A document-term matrix or term-document matrix is a mathematical matrix that describes the frequency of terms that occur in a collection of documents. 
# 
# In a document-term matrix, rows correspond to documents in the collection and columns correspond to terms. 
# 
# There are various schemes for determining the value that each entry in the matrix should take. One such scheme is tf-idf. They are useful in the field of natural language processing.

# D1 = "I like databases"
# 
# D2 = "I hate databases"
# 
# |          |  I        |    like   |hate        | databases   |
# | -------------|:-------------:|:-------------:|:-------------:|-----:|
# | D1| 1| 1 | 0 |1|
# | D2| 1| 0 | 1 |1|

# In[10]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(ngram_range=(1, 2))
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)


# In[12]:


get_ipython().run_line_magic('pinfo', 'count')


# In[13]:


count.get_feature_names()


# In[14]:


print(count.vocabulary_) # word: position index


# In[15]:


type(bag)


# In[16]:


print(bag.toarray())


# In[17]:


import pandas as pd
pd.DataFrame(bag.toarray(), columns = count.get_feature_names())


# The sequence of items in the bag-of-words model that we just created is also called the 1-gram or unigram model: each item or token in the vocabulary represents a single word. 
# 
# ## n-gram model
# The choice of the number n in the n-gram model depends on the particular application
# 
# - 1-gram: "the", "sun", "is", "shining"
# - 2-gram: "the sun", "sun is", "is shining" 

# The CountVectorizer class in scikit-learn allows us to use different
# n-gram models via its `ngram_range` parameter. 
# 
# While a 1-gram
# representation is used by default
# 
# we could switch to a 2-gram
# representation by initializing a new CountVectorizer instance with
# ngram_range=(2,2).

# ## TF-IDF
# Assessing word relevancy via term frequency-inverse document frequency

# $$tf*idf(t, d) = tf(t, d) \times idf(t)$$
# 
# - $tf(t, d)$ is the term frequency of term t in document d.
# - inverse document frequency $idf(t)$ can be calculated as: $idf(t) = log \frac{n_d}{1 + df(d, t)}$
# 
# 

# Question: Why do we add the constant 1 to the denominator ?
# 
# 

# The tf-idf equation that was implemented in scikit-learn is as follows: $tf*idf(t, d) = tf(t, d) \times （idf(t, d) + 1）$
#  
# [SKlearn use `smooth_idf=True`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer) $idf(t) = log \frac{1+n_d}{1 + df(d, t)} + 1$
# 
# where $n_d$ is the total number of documents, and $df(d, t)$ is the number of documents $d$ that contain the term $t$. 
# 

#  
# ### L2-normalization
# 
# $$l2_{x} = \frac{x} {\sqrt{\sum {x^2}}}$$
# 
# 

# 课堂作业：请根据公式计算'is'这个词在文本2中的tfidf数值？
# 
# ![](./img/ask.jpeg)

# ### TfidfTransformer
# Scikit-learn implements yet another transformer, the TfidfTransformer, that
# takes the raw term frequencies from CountVectorizer as input and transforms
# them into tf-idfs:

# In[18]:


from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision=2)

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


# In[19]:


from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision=2)

tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


# In[20]:


import pandas as pd
bag = tfidf.fit_transform(count.fit_transform(docs))
pd.DataFrame(bag.toarray(), columns = count.get_feature_names())


# In[21]:


# 一个词的tfidf值
import numpy as np
tf_is = 2.0
n_docs = 3.0
# smooth_idf=True & norm = None
idf_is = np.log((1+n_docs) / (1+3)) + 1

tfidf_is = tf_is * idf_is
print('tf-idf of term "is" = %.2f' % tfidf_is)


# In[22]:


# *最后一个文本*里的词的tfidf原始数值（未标准化）
tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf, count.get_feature_names()


# In[23]:


# l2标准化后的tfidf数值
l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf 


# ## 政府工作报告文本挖掘

# ### 0. 读取数据

# In[26]:


with open('./data/gov_reports1954-2021.txt', 'r', encoding = 'utf-8') as f:
    reports = f.readlines()
    


# In[27]:


len(reports)


# In[37]:


print(reports[-7][:1000])


# In[16]:


print(reports[4][:500])


#  pip install jieba
# > https://github.com/fxsjy/jieba
# 
#  pip install wordcloud
# >  https://github.com/amueller/word_cloud
# 
#  pip install gensim
# 

# In[20]:


pip install gensim


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys 
import numpy as np
from collections import defaultdict
import statsmodels.api as sm
from wordcloud import WordCloud
import jieba
import matplotlib
import gensim
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
#matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #指定默认字体 
matplotlib.rc("savefig", dpi=400)


# In[75]:


# 为了确保中文可以在matplotlib里正确显示
#matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #指定默认字体 
# 需要确定系统安装了Microsoft YaHei


# In[76]:


# import matplotlib
# my_font = matplotlib.font_manager.FontProperties(
#     fname='/Users/chengjun/github/cjc/data/msyh.ttf')


# ### 1. 分词

# In[39]:


import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))


# ## 2. 停用词

# In[40]:


filename = './data/stopwords.txt'
stopwords = {}
f = open(filename, 'r')
line = f.readline().rstrip()
while line:
    stopwords.setdefault(line, 0)
    stopwords[line] = 1
    line = f.readline().rstrip()
f.close()


# In[36]:


adding_stopwords = [u'我们', u'要', u'地', u'有', u'这', u'人',
                    u'发展',u'建设',u'加强',u'继续',u'对',u'等',
                    u'推进',u'工作',u'增加']
for s in adding_stopwords: stopwords[s]=10 


# ### 3. 关键词抽取

# #### 基于TF-IDF 算法的关键词抽取

# In[41]:


import jieba.analyse
txt = reports[-1]
tf = jieba.analyse.extract_tags(txt, topK=200, withWeight=True)


# In[42]:


u"、".join([i[0] for i in tf[:50]])


# In[43]:


plt.hist([i[1] for i in tf])
plt.show()


# #### 基于 TextRank 算法的关键词抽取

# In[44]:


tr = jieba.analyse.textrank(txt,topK=200, withWeight=True)
u"、".join([i[0] for i in tr[:50]])


# In[45]:


plt.hist([i[1] for i in tr])
plt.show()


# In[46]:


import pandas as pd

def keywords(index):
    txt = reports[-index]
    tf = jieba.analyse.extract_tags(txt, topK=200, withWeight=True)
    tr = jieba.analyse.textrank(txt,topK=200, withWeight=True)
    tfdata = pd.DataFrame(tf, columns=['word', 'tfidf'])
    trdata = pd.DataFrame(tr, columns=['word', 'textrank'])
    worddata = pd.merge(tfdata, trdata, on='word')
    fig = plt.figure(figsize=(16, 6),facecolor='white')
    plt.plot(worddata.tfidf, worddata.textrank, linestyle='',marker='.')
    for i in range(len(worddata.word)):
        plt.text(worddata.tfidf[i], worddata.textrank[i], worddata.word[i], 
                 fontsize = worddata.textrank[i]*30, 
                 color = 'red', rotation = 0
                )
    plt.title(txt[:4])
    plt.xlabel('Tf-Idf')
    plt.ylabel('TextRank')
    plt.show()


# In[49]:


plt.style.use('ggplot')

keywords(1)


# In[55]:


keywords(-1)


# TextRank: Bringing Order into Texts
# 
# 基本思想:
# 
# * 将待抽取关键词的文本进行分词
# * 以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
# * 计算图中节点的PageRank，注意是无向带权图

# ### 4. 词云

# In[44]:


def wordcloudplot(txt, year):
    wordcloud = WordCloud(font_path='../data/msyh.ttf').generate(txt)
    # Open a plot of the generated image.
    fig = plt.figure(figsize=(16, 6),facecolor='white')
    plt.imshow(wordcloud)
    plt.title(year)
    plt.axis("off")
    #plt.show()


# #### 基于tfidf过滤的词云

# In[45]:


txt = reports[-1]
tfidf200= jieba.analyse.extract_tags(txt, topK=200, withWeight=False)
seg_list = jieba.cut(txt, cut_all=False)
seg_list = [i for i in seg_list if i in tfidf200]
txt200 = r' '.join(seg_list)
wordcloudplot(txt200, txt[:4]) 


# In[24]:


txt = reports[-2]
tfidf200= jieba.analyse.extract_tags(txt, topK=200, withWeight=False)
seg_list = jieba.cut(txt, cut_all=False)
seg_list = [i for i in seg_list if i in tfidf200]
txt200 = r' '.join(seg_list)
wordcloudplot(txt200, txt[:4]) 


# In[59]:


wordfreq = defaultdict(int)
for i in seg_list:
    wordfreq[i] +=1
wordfreq = [[i, wordfreq[i]] for i in wordfreq]

wordfreq.sort(key= lambda x:x[1], reverse = True )
u"、 ".join([ i[0] + u'（' + str(i[1]) +u'）' for i in wordfreq ])


# #### 基于停用词过滤的词云

# In[70]:


#jieba.add_word('股灾', freq=100, tag=None) 

txt = reports[-1]
seg_list = jieba.cut(txt, cut_all=False)
seg_list = [i for i in seg_list if i not in stopwords]
txt = r' '.join(seg_list)
wordcloudplot(txt, txt[:4])  
#file_path = '/Users/chengjun/GitHub/cjc2016/figures/wordcloud-' + txt[:4] + '.png'
#plt.savefig(file_path,dpi = 300, bbox_inches="tight",transparent = True)


# ### 绘制1954-2016政府工作报告词云

# In[113]:


#jieba.add_word('股灾', freq=100, tag=None) 

for txt in reports:
    seg_list = jieba.cut(txt, cut_all=False)
    seg_list = [i for i in seg_list if i not in stopwords]
    txt = r' '.join(seg_list)
    wordcloudplot(txt, txt[:4]) 
    file_path = '../figure/wordcloud-' + txt[:4] + '.png'
    plt.savefig(file_path,dpi = 400, bbox_inches="tight",                transparent = True)


# ## 5. 词向量的时间序列

# In[89]:


reports[0][:500]


# In[90]:


reports[1][:500]


# In[27]:


test = jieba.analyse.textrank(reports[0], topK=200, withWeight=False)


# In[57]:


test = jieba.analyse.extract_tags(reports[1], topK=200, withWeight=False)


# In[56]:


import jieba.analyse

wordset = []
for k, txt in enumerate(reports):
    print(k)
    top200= jieba.analyse.extract_tags(txt, topK=200, withWeight=False)
    for w in top200:
        if w not in wordset:
            wordset.append(w)


# In[57]:


len(wordset)


# In[58]:


print(' '.join(wordset))


# In[60]:


from collections import defaultdict

data = defaultdict(dict)
years = [int(i[:4]) for i in reports]
for i in wordset:
    for year in years:
        data[i][year] = 0 


# In[62]:


for txt in reports:
    year = int(txt[:4])
    print(year)
    top1000= jieba.analyse.extract_tags(txt, topK=1000, withWeight=True)
    for ww in top1000:
        word, weight = ww
        if word in wordset:
            data[word][year]+= weight


# In[63]:


word_weight = []
for i in data:
    word_weight.append([i, np.sum(list(data[i].values()))])


# In[64]:


word_weight.sort(key= lambda x:x[1], reverse = True )
top50 = [i[0] for i in word_weight[:50]]


# In[65]:


' '.join(top50) 


# In[75]:


def plotEvolution(word, color, linestyle, marker):
    cx = data[word]
    plt.plot(list(cx.keys()), list(cx.values()), color = color, 
             linestyle=linestyle, marker=marker, label= word)
    plt.legend(loc=2,fontsize=18)
    plt.ylabel(u'词语重要性')


# In[76]:


plt.figure(figsize=(16, 6),facecolor='white')
plotEvolution(u'民主', 'g', '-', '>')
plotEvolution(u'法制', 'b', '-', 's')


# In[77]:


plt.figure(figsize=(16, 6),facecolor='white')

plotEvolution(u'动能', 'b', '-', 's')
plotEvolution(u'互联网', 'g', '-', '>')


# In[78]:


plt.figure(figsize=(16, 6),facecolor='white')

plotEvolution(u'工业', 'y', '-', '<')
plotEvolution(u'农业', 'r', '-', 'o')
plotEvolution(u'制造业', 'b', '-', 's')
plotEvolution(u'服务业', 'g', '-', '>')


# In[79]:


plt.figure(figsize=(16, 8),facecolor='white')

plotEvolution(u'教育', 'r', '-', 'o')
plotEvolution(u'社会保障', 'b', '-', 's')
plotEvolution(u'医疗', 'g', '-', '>')


# In[80]:


plt.figure(figsize=(16, 8),facecolor='white')

plotEvolution(u'环境', 'b', '-', 's')
plotEvolution(u'住房', 'purple', '-', 'o')


# In[81]:


plt.figure(figsize=(16, 8),facecolor='white')

plotEvolution(u'发展', 'y', '-', '<')
plotEvolution(u'经济', 'r', '-', 'o')
plotEvolution(u'改革', 'b', '-', 's')
plotEvolution(u'创新', 'g', '-', '>')


# In[82]:


plt.figure(figsize=(16, 6),facecolor='white')

plotEvolution(u'社会主义', 'r', '-', 'o')
plotEvolution(u'马克思主义', 'b', '-', 's')


# ![image.png](images/end.png)

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# 10-word2vec
# 11-1-sentiment-analysis-with-dict
# 11-2-emotion-dict
# 11-3-NRC-Chinese-dict
# 11-3-textblob
# 11-4-sentiment-classifier
# 11-5-LIWC
# 12-topic-models-update
# 12-topic-models-with-turicreate
# ```
# 
