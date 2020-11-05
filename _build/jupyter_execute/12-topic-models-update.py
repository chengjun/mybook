#!/usr/bin/env python
# coding: utf-8

# 
# # 主题模型简介
# 
# 
# 
# 
# ![image.png](images/author.png)

# 2014年高考前夕，百度“基于海量作文范文和搜索数据，利用概率主题模型，预测2014年高考作文的命题方向”。共分为了六个主题：时间、生命、民族、教育、心灵、发展。而每个主题下面又包括了一些具体的关键词。比如，生命的主题对应：平凡、自由、美丽、梦想、奋斗、青春、快乐、孤独。[Read more](https://site.douban.com/146782/widget/notes/15462869/note/356806087/)

# <div><img src="images/lda.png" width="1000px"></div>

# ## Latent Dirichlet Allocation (LDA)
# 
# LDA (潜在狄利克雷分配) is a generative model that **infers unobserved meanings** from a large set of observations. 
# - Blei DM, Ng J, Jordan MI. **Latent dirichlet allocation**. J Mach Learn Res. 2003; 3: 993–1022.
# - Blei DM, Lafferty JD. Correction: a correlated topic model of science. Ann Appl Stat. 2007; 1: 634. 
# - Blei DM. **Probabilistic topic models**. Commun ACM. 2012; 55: 55–65.
# - Chandra Y, Jiang LC, Wang C-J (2016) Mining Social Entrepreneurship Strategies Using Topic Modeling. PLoS ONE 11(3): e0151342. 

# 
# ### Topic models assume that each document contains a mixture of topics.
# 
# It is impossible to directly assess the relationships between topics and documents and between topics and terms. 
# 
# - Topics are considered latent/unobserved variables that stand between the documents and terms
# 
# - What can be directly observed is the distribution of terms over documents, which is known as the document term matrix (DTM).
# 
# Topic models algorithmically identify the best set of latent variables (topics) that can best explain the observed distribution of terms in the documents. 

# The DTM is further decomposed into two matrices：
# - a term-topic matrix (TTM) 
# - a topic-document matrix (TDM)
# 
# Each document can be assigned to a primary topic that demonstrates the highest topic-document probability and can then be linked to other topics with declining probabilities.

# ![image.png](images/lda2.png)

# ### LDA（Latent Dirichlet Allocation）是一种**文档主题**生成模型
# - 三层贝叶斯概率模型，包含词、主题和文档三层结构。
# 
# **生成模型**认为一篇文章的每个词都是通过这样一个过程得到:
# 
#  以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语
# 
# - 文档到主题服从**多项式分布**，主题到词服从多项式分布。

# ### 多项式分布（Multinomial Distribution）是二项式分布的推广
# - 二项分布的典型例子是扔硬币，硬币正面朝上概率为p, 重复扔n次硬币，k次为正面的概率即为一个二项分布概率。（严格定义见伯努利实验定义）。
# - 把二项分布公式推广至多种状态，就得到了多项分布。
#     - 例如在上面例子中1出现k1次，2出现k2次，3出现k3次的概率分布情况。

# ## LDA是一种**非监督机器学习技术**
# 
# 可以用来识别大规模文档集（document collection）或语料库（corpus）中潜藏的主题信息。
# - 采用了词袋（bag of words）的方法，将每一篇文档视为一个词频向量，从而将文本信息转化为了易于建模的数字信息。 
# - 但是词袋方法没有考虑词与词之间的顺序，这简化了问题的复杂性，同时也为模型的改进提供了契机。
# - 每一篇文档代表了一些主题所构成的一个概率分布，而每一个主题又代表了很多单词所构成的一个概率分布。

# ![image.png](images/lda3.png)

# 
# 
# ### 多项分布的参数服从Dirichlet分布
# - Dirichlet分布是多项分布的参数的分布， 被认为是“分布上的分布”。
# 
# \begin{equation}
#   \text{Dir}\left(\boldsymbol{\alpha}\right)\rightarrow \mathrm{p}\left(\boldsymbol{\theta}\mid\boldsymbol{\alpha}\right)=\frac{\Gamma\left(\sum_{i=1}^{k}\boldsymbol{\alpha}_{i}\right)}{\prod_{i=1}^{k}\Gamma\left(\boldsymbol{\alpha}_{i}\right)}\prod_{i=1}^{k}\boldsymbol{\theta}_{i}^{\boldsymbol{\alpha}_{i}-1}
# \end{equation}
# 
# 

# In[15]:


# http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

quantiles = np.array([0.2, 0.2, 0.6])  # specify quantiles
alpha = np.array([0.4, 5, 15])  # specify concentration parameters
dirichlet.pdf(quantiles, alpha)


# In[9]:


corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

# Mid-points of triangle sides opposite of each corner
midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0              for i in range(3)]
def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75          for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)


# In[13]:


class Dirichlet(object):
    def __init__(self, alpha):
        self._alpha = np.array(alpha)
        
    def pdf(self, x):
        x=x/x.sum() # enforce simplex constraint
        return dirichlet.pdf(x=x,alpha=self._alpha)


# In[19]:


def draw_pdf_contours(dist, nlevels=200, subdiv=8, **kwargs):
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)

    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')


# In[21]:


draw_pdf_contours(Dirichlet([10., 20., 3.]))


# In[20]:


draw_pdf_contours(Dirichlet([1, 2, 3]))


# ##  LDA的名字由来
# 
# 存在两个隐含的Dirichlet分布。
# 
# - 每篇文档对应一个不同的topic分布，服从多项分布
#     - topic多项分布的参数服从一个Dirichlet分布。 
# - 每个topic下存在一个term的多项分布
#     - term多项分布的参数服从一个Dirichlet分布。
#     
# 

# Assume K topics are in D documents.
# 
# `主题在词语上的分布` Each topic is denoted with $\beta_{1:K}$， 
# 
# - 主题$\beta_K$ 是第k个主题，这个主题表达为一系列的terms。
# - Each topic is a distribution of fixed words. 
# 
# `主题在文本上的分布` The topics proportion in the document *d* is denoted as $\theta_d$
# 
# - e.g., the kth topic's proportion in document d is $\theta_{d, k}$. 

# `主题在文本和词上的分配`
# 
# topic models assign topics to a document and its terms. 
# - The topic assigned to document *d* is denoted as $z_d$, 
# - The topic assigned to the nth term in document *d* is denoted as $z_{d,n}$. 

# `可以观察到的是？`
# 
# 词在文档中的位置，也就是文档-词矩阵（document-term matrix）
# 
# Let $w_{d,n}$ denote the nth term in document d. 

# `联合概率分布` According to Blei et al. the joint distribution of $\beta_{1:K}$,$\theta_{1:D}$, $z_{1:D}$ and $w_{d, n}$ plus the generative process for LDA can be expressed as:
# 
# $$ p(\beta_{1:K}, \theta_{1:D}, z_{1:D}, w_{d, n})  = $$
# 
# $$\prod_{i=1}^{K} p(\beta_i) \prod_{d =1}^D p(\theta_d)(\prod_{n=1}^N p(z_{d,n} \mid \theta_d) \times p(w_{d, n} \mid \beta_{1:K}, Z_{d, n})  ) $$
# 
# <div><img src="images/lda41.png" align="right"></div>

# ![image.png](images/lda5.png)

# 
# **后验分布** Note that $\beta_{1:k},\theta_{1:D},and z_{1:D}$ are latent, unobservable variables. Thus, the computational challenge of LDA is to compute the conditional distribution of them given the observable specific words in the documents $w_{d, n}$. 
# 
# Accordingly, the posterior distribution of LDA can be expressed as:
# 
# $$p(\beta_{1:K}, \theta_{1:D}, z_{1:D} \mid w_{d, n}) = \frac{p(\beta_{1:K}, \theta_{1:D}, z_{1:D}, w_{d, n})}{p(w_{1:D})}$$

# Because the number of possible topic structures is exponentially large, it is impossible to compute the posterior of LDA. 
# 
# Topic models aim to develop efficient algorithms to **approximate** the posterior of LDA. There are two categories of algorithms: 
# - sampling-based algorithms
# - variational algorithms 
# 
# 

# ### Gibbs sampling
# In statistics, Gibbs sampling or a Gibbs sampler is a **Markov chain Monte Carlo (MCMC)** algorithm for obtaining a sequence of observations which are approximated from a specified **multivariate probability distribution**, when direct sampling is difficult. 
# 
# Using the Gibbs sampling method, we can build a Markov chain for the sequence of random variables (see Eq 1). 
# 
# The sampling algorithm is applied to the chain to sample from the limited distribution, and it approximates the **posterior**. 

# 
# ## Gensim: Topic modelling for humans
# 
# 
# 
# Gensim is developed by Radim Řehůřek,who is a machine learning researcher and consultant in the Czech Republic. We must start by installing it. We can achieve this by running the following command:
# 
# > pip install gensim
# 

# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
from gensim import corpora, models, similarities,  matutils
import matplotlib.pyplot as plt
import numpy as np


# **Download data**
# 
# <del>http://www.cs.princeton.edu/~blei/lda-c/ap.tgz</del>
# 
# http://www.cs.columbia.edu/~blei/lda-c/
# 
# Unzip the data and put them into your folder, e.g., /Users/datalab/bigdata/ap/

# In[23]:


# Load the data
corpus = corpora.BleiCorpus('/Users/datalab/bigdata/ap/ap.dat',                            '/Users/datalab/bigdata/ap/vocab.txt')


# **使用help命令理解corpora.BleiCorpus函数**
# 
# > help(corpora.BleiCorpus)
 class BleiCorpus(gensim.corpora.indexedcorpus.IndexedCorpus)
 |  Corpus in Blei's LDA-C format.
 |  
 |  The corpus is represented as two files: 
 |          one describing the documents, 
 |          and another describing the mapping between words and their ids.
# In[24]:


# 使用dir看一下有corpus有哪些子函数？
dir(corpus)[-10:]


# In[25]:


# corpus.id2word is a dict which has keys and values, e.g., 
{0: u'i', 1: u'new', 2: u'percent', 3: u'people', 4: u'year', 5: u'two'}


# In[26]:


# transform the dict to list using items()
corpusList = list(corpus.id2word.items())


# In[27]:


# show the first 5 elements of the list
corpusList[:5]


# ## Build the topic model

# In[28]:


# 设置主题数量
NUM_TOPICS = 100


# In[29]:


model = models.ldamodel.LdaModel(
    corpus, num_topics=NUM_TOPICS, 
    id2word=corpus.id2word, 
    alpha=None)


# **help(models.ldamodel.LdaModel)**
# 
# Help on class LdaModel in module gensim.models.ldamodel:
# 
# class LdaModel(gensim.interfaces.TransformationABC, gensim.models.basemodel.BaseTopicModel)
# - The constructor estimates Latent Dirichlet Allocation model parameters based on a training corpus:
#  
# > lda = LdaModel(corpus, num_topics=10)
#  
# - You can then infer topic distributions on new, unseen documents, with
# 
# > doc_lda = lda[doc_bow]  
# 
# - The model can be updated (trained) with new documents via
# 
# > lda.update(other_corpus)

# In[30]:


# 看一下训练出来的模型有哪些函数？
' '.join(dir(model))


# We can see the list of topics a document refers to 
# 
# by using the model[doc] syntax:

# In[31]:


document_topics = [model[c] for c in corpus]


# In[32]:


# how many topics does one document cover?
# 例如，对于文档2来说，他所覆盖的主题和比例如下：
document_topics[2]


# In[33]:


# The first topic
# 对于主题0而言，它所对应10个词语和比重如下：
model.show_topic(55, 20)


# In[35]:


# 对于主题0而言，它所对应10个词语和比重如下：
words = model.show_topic(0, 10)
words


# In[36]:


for f, w in words[:10]:
    print(f, w)


# In[37]:


# 对于主题99而言，它所对应10个词语和比重如下：

model.show_topic(99, 10)


# In[38]:


# 模型计算出来的所有的主题当中的第1个是？
model.show_topic(0)


# In[39]:


#help(model.show_topics(0))
for w, f in words:
    print(w, f)


# In[46]:


# write out topcis with 10 terms with weights
for ti in range(model.num_topics):
    words = model.show_topic(ti, 10)
    tf = sum(f for w, f in words)
    with open('/Users/chengjun/github/workshop/data/topics_term_weight.txt', 'a') as output:
        for w, f in words:
            line = str(ti) + '\t' +  w + '\t' + str(f/tf) 
            output.write(line + '\n')


# ### Find the most discussed topic
# 
# i.e., the one with the highest total weight

# In[40]:


## Convert corpus into a dense np array 
help(matutils.corpus2dense)


# In[41]:


topics = matutils.corpus2dense(model[corpus], 
                               num_terms=model.num_topics)
topics


# In[42]:


# Return the sum of the array elements 
help(topics.sum)


# In[43]:


# 第一个主题的词语总权重
topics[0].sum()


# In[44]:


# 将每一个主题的词语总权重算出来
weight = topics.sum(1)
weight


# In[45]:


# 找到最大值在哪里

help(weight.argmax)  


# In[46]:


# 找出具有最大权重的主题是哪一个
max_topic = weight.argmax()
print(max_topic)


# In[47]:


# Get the top 64 words for this topic
# Without the argument, show_topic would return only 10 words
words = model.show_topic(max_topic, 64)
words = np.array(words).T
words_freq=[float(i)*10000000 for i in words[1]]
words = list(zip(words[0], words_freq))


# ### 主题词云

# In[48]:


words = {i:j for i, j in words}


# In[49]:


from wordcloud import WordCloud

fig = plt.figure(figsize=(15, 8),facecolor='white')

wordcloud = WordCloud().generate_from_frequencies(words)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# ### 每个文档有多少主题
# 

# In[50]:


# 每个文档有多少主题
num_topics_used = [len(model[doc]) for doc in corpus]


# In[51]:


# 画出来每个文档主题数量的直方图

fig,ax = plt.subplots()
ax.hist(num_topics_used, np.arange(27))
ax.set_ylabel('$Number \;of\; documents$', fontsize = 20)
ax.set_xlabel('$Number \;of \;topics$', fontsize = 20)
fig.tight_layout()
#fig.savefig('Figure_04_01.png')


# We can see that about 150 documents have 5 topics, 
# - while the majority deal with around 10 to 12 of them. 
#     - No document talks about more than 30 topics.

# ### 改变超级参数alpha

# In[52]:


# Now, repeat the same exercise using alpha=1.0
# You can edit the constant below to play around with this parameter
ALPHA = 1.0
model1 = models.ldamodel.LdaModel(
    corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, 
    alpha=ALPHA)

num_topics_used1 = [len(model1[doc]) for doc in corpus]


# In[53]:


fig,ax = plt.subplots()
ax.hist([num_topics_used, num_topics_used1], np.arange(42))
ax.set_ylabel('$Number \;of\; documents$', fontsize = 20)
ax.set_xlabel('$Number \;of \;topics$', fontsize = 20)
# The coordinates below were fit by trial and error to look good
plt.text(9, 223, r'default alpha')
plt.text(26, 156, 'alpha=1.0')
fig.tight_layout()


# **问题**：$\alpha$引起主题数量分布的变化意味着什么？

# ## 从原始文本到主题模型
# 
# 一个完整的例子
# 
# 刚才的例子使用的是一个已经处理好的语料库，已经构建完整的语料和字典，并清洗好了数据。

# In[54]:


with open('/Users/datalab/bigdata/ap/ap.txt', 'r') as f:
    dat = f.readlines()


# In[55]:


# 需要进行文本清洗
dat[:6]


# In[56]:


# 如果包含'<'就去掉这一行
dat[4].strip()[0]


# In[57]:


# 选取前100篇文档
docs = []
for k, i in enumerate(dat): #[:100]:
    #print(k)
    try:
        if i.strip()[0] != '<':
            docs.append(i)
    except Exception as e:
        print(k, e)


# In[58]:


len(docs)


# In[59]:


docs[-1]


# In[60]:


# 定义一个函数，进一步清洗
def clean_doc(doc):
    doc = doc.replace('.', '').replace(',', '')
    doc = doc.replace('``', '').replace('"', '')
    doc = doc.replace('_', '').replace("'", '')
    doc = doc.replace('!', '')
    return doc
docs = [clean_doc(doc) for doc in docs]


# In[61]:


texts = [[i for i in doc.lower().split()] for doc in docs]


# ### 停用词

# In[62]:


import nltk
#nltk.download()
# 会打开一个窗口，选择book，download，待下载完毕就可以使用了。


# In[63]:


from nltk.corpus import stopwords
stop = stopwords.words('english') # 如果此处出错，请执行上一个block的代码
# 停用词stopword：在英语里面会遇到很多a,the,or等使用频率很多的字或词,常为冠词、介词、副词或连词等。
# 人类语言包含很多功能词。与其他词相比，功能词没有什么实际含义。


# In[64]:


' '.join(stop) 


# In[65]:


from gensim.parsing.preprocessing import STOPWORDS

' '.join(STOPWORDS)


# In[66]:


stop.append('said')


# In[67]:


# 计算每一个词的频数
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1


# In[68]:


# 去掉只出现一次的词和
texts = [[token for token in text           if frequency[token] > 1 and token not in stop]
        for text in texts]


# In[69]:


docs[8]


# In[70]:


' '.join(texts[9])


# **help(corpora.Dictionary)**
# 
# Help on class Dictionary in module gensim.corpora.dictionary:
# 
# class Dictionary(gensim.utils.SaveLoad, _abcoll.Mapping)
# - Dictionary encapsulates the mapping between normalized words and their integer ids.
#   
# - The main function is **doc2bow**
#     - which converts a collection of words to its bag-of-words representation: a list of (word_id, word_frequency) 2-tuples.
#  

# In[71]:


dictionary = corpora.Dictionary(texts)
lda_corpus = [dictionary.doc2bow(text) for text in texts]
# The function doc2bow() simply counts the number of occurences of each distinct word, 
# converts the word to its integer word id and returns the result as a sparse vector. 


# In[72]:


NUM_TOPICS = 100
lda_model = models.ldamodel.LdaModel(
    lda_corpus, num_topics=NUM_TOPICS, 
    id2word=dictionary, alpha=None)


# ## 使用pyLDAvis可视化主题模型
# http://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb
# 
# > pip install pyldavis

# In[74]:


pip install pyldavis


# In[75]:


# pyldavis
import pyLDAvis.gensim

ap_data = pyLDAvis.gensim.prepare(lda_model, lda_corpus, dictionary, mds = 'mmds')

pyLDAvis.enable_notebook()
pyLDAvis.display(ap_data)


# In[83]:


import pyLDAvis.gensim

ap_data = pyLDAvis.gensim.prepare(lda_model, lda_corpus, dictionary, mds = 'tsne')

pyLDAvis.enable_notebook()
pyLDAvis.display(ap_data)


# In[28]:


pyLDAvis.show(ap_data)


# In[27]:


pyLDAvis.save_html(ap_data, '../data/ap_ldavis2.html')


# ![image.png](images/end.png)

# ## 对2016年政府工作报告建立主题模型

# pip install jieba
# > https://github.com/fxsjy/jieba
# 
# pip install wordcloud
# >  https://github.com/amueller/word_cloud
# 
# pip install gensim

# In[73]:


import gensim
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import matplotlib
matplotlib.rc("savefig", dpi=400)
#matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #指定默认字体 


# In[40]:


import urllib2
from bs4 import BeautifulSoup
import sys

url2016 = 'http://news.xinhuanet.com/fortune/2016-03/05/c_128775704.htm'
content = urllib2.urlopen(url2016).read()
soup = BeautifulSoup(content) 


# In[41]:


gov_report_2016 = [s.text for s in soup('p')]
for i in gov_report_2016[:10]:
    print(i)


# In[42]:


def clean_txt(txt):
    for i in [u'、', u'，', u'—', u'！', u'。', u'《', u'》', u'（', u'）']:
        txt = txt.replace(i, ' ')
    return txt


# In[43]:


gov_report_2016 = [clean_txt(i) for i in gov_report_2016]


# In[109]:


len(gov_report_2016)


# In[110]:


for i in gov_report_2016[:10]:
    print(i)


# In[111]:


len(gov_report_2016[5:-1])


# In[112]:


# Set the Working Directory 
import os
os.getcwd() 
os.chdir('/Users/chengjun/github/cjc/')
os.getcwd()


# In[113]:


filename = 'data/stopwords.txt'
stopwords = {}
f = open(filename, 'r')
line = f.readline().rstrip()
while line:
    stopwords.setdefault(line, 0)
    stopwords[line.decode('utf-8')] = 1
    line = f.readline().rstrip()
f.close()


# In[114]:


adding_stopwords = [u'我们', u'要', u'地', u'有', u'这', u'人',
                    u'发展',u'建设',u'加强',u'继续',u'对',u'等',
                    u'推进',u'工作',u'增加']
for s in adding_stopwords: stopwords[s]=10


# In[118]:


import jieba.analyse

def cleancntxt(txt, stopwords):
    tfidf1000= jieba.analyse.extract_tags(txt, topK=1000, withWeight=False)
    seg_generator = jieba.cut(txt, cut_all=False)
    seg_list = [i for i in seg_generator if i not in stopwords]
    seg_list = [i for i in seg_list if i != u' ']
    seg_list = [i for i in seg_list if i in tfidf1000]
    return(seg_list)


# In[119]:


def getCorpus(data):
    processed_docs = [tokenize(doc) for doc in data]
    word_count_dict = gensim.corpora.Dictionary(processed_docs)
    print ("In the corpus there are", len(word_count_dict), "unique tokens")
    word_count_dict.filter_extremes(no_below=5, no_above=0.2) # word must appear >5 times, and no more than 10% documents
    print ("After filtering, in the corpus there are only", len(word_count_dict), "unique tokens")
    bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]
    return bag_of_words_corpus, word_count_dict


def getCnCorpus(data):
    processed_docs = [cleancntxt(doc) for doc in data]
    word_count_dict = gensim.corpora.Dictionary(processed_docs)
    print ("In the corpus there are", len(word_count_dict), "unique tokens")
    #word_count_dict.filter_extremes(no_below=5, no_above=0.2) 
    # word must appear >5 times, and no more than 10% documents
    print ("After filtering, in the corpus there are only", len(word_count_dict), "unique tokens")
    bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]
    return bag_of_words_corpus, word_count_dict



# In[120]:


def inferTopicNumber(bag_of_words_corpus, num, word_count_dict):
    lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=num, id2word=word_count_dict, passes=10)
    _ = lda_model.print_topics(-1) #use _ for throwaway variables.
    logperplexity = lda_model.log_perplexity(bag_of_words_corpus)
    return logperplexity

def fastInferTopicNumber(bag_of_words_corpus, num, word_count_dict):
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=bag_of_words_corpus, num_topics=num,                                                         id2word=word_count_dict,                                                        workers=None, chunksize=2000, passes=2,                                                         batch=False, alpha='symmetric', eta=None,                                                         decay=0.5, offset=1.0, eval_every=10,                                                         iterations=50, gamma_threshold=0.001, random_state=None)
    _ = lda_model.print_topics(-1) #use _ for throwaway variables.
    logperplexity = lda_model.log_perplexity(bag_of_words_corpus)
    return logperplexity


# In[116]:


import jieba.analyse

jieba.add_word(u'屠呦呦', freq=None, tag=None)
#del_word(word) 

print (' '.join(cleancntxt(u'屠呦呦获得了诺贝尔医学奖。', stopwords)))


# In[117]:


import gensim

processed_docs = [cleancntxt(doc, stopwords) for doc in gov_report_2016[5:-1]]
word_count_dict = gensim.corpora.Dictionary(processed_docs)
print ("In the corpus there are", len(word_count_dict), "unique tokens")
# word_count_dict.filter_extremes(no_below=5, no_above=0.2) # word must appear >5 times, and no more than 10% documents
# print "After filtering, in the corpus there are only", len(word_count_dict), "unique tokens"
bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]


# In[118]:


tfidf = models.TfidfModel(bag_of_words_corpus )
corpus_tfidf = tfidf[bag_of_words_corpus ]
#lda_model = gensim.models.LdaModel(corpus_tfidf, num_topics=20, id2word=word_count_dict, passes=10)
lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20, id2word=word_count_dict, passes=10)


# In[65]:


perplexity_list = [inferTopicNumber(bag_of_words_corpus, num, word_count_dict) for num in [5, 10, 15, 20, 25, 30 ]]


# In[67]:


plt.plot([5, 10, 15, 20, 25, 30 ], perplexity_list)
plt.show()


# In[119]:


lda_model.print_topics(3)


# In[120]:


topictermlist = lda_model.print_topics(-1)
top_words = [[j.split('*')[1] for j in i[1].split(' + ')] for i in topictermlist] 
for i in top_words: 
    print (" ".join(i) )


# In[121]:


top_words_shares = [[j.split('*')[0] for j in i[1].split(' + ')] for i in topictermlist] 
top_words_shares = [map(float, i) for i in top_words_shares]
def weightvalue(x):
    return (x - np.min(top_words_shares))*40/(np.max(top_words_shares) -np.min(top_words_shares)) + 10
 
top_words_shares = [map(weightvalue, i) for i in top_words_shares]  

def plotTopics(mintopics, maxtopics):
    num_top_words = 10
    plt.rcParams['figure.figsize'] = (20.0, 8.0)  
    n = 0
    for t in range(mintopics , maxtopics):
        plt.subplot(2, 15, n + 1)  # plot numbering starts with 1
        plt.ylim(0, num_top_words)  # stretch the y-axis to accommodate the words
        plt.xticks([])  # remove x-axis markings ('ticks')
        plt.yticks([]) # remove y-axis markings ('ticks')
        plt.title(u'主题 #{}'.format(t+1), size = 15)
        words = top_words[t][0:num_top_words ]
        words_shares = top_words_shares[t][0:num_top_words ]
        for i, (word, share) in enumerate(zip(words, words_shares)):
            plt.text(0.05, num_top_words-i-0.9, word, fontsize= np.log(share*1000))
        n += 1


# In[122]:


plotTopics(0, 10)


# In[123]:


plotTopics(10, 20)


# ## 对宋词进行主题分析初探

# 宋词数据下载 http://cos.name/wp-content/uploads/2011/03/SongPoem.tar.gz

# In[2]:


import pandas as pd


# In[9]:


pdf = pd.read_csv('./data/SongPoem.csv', encoding = 'gb18030')

pdf[:3]


# In[124]:


len(pdf)


# In[10]:


poems = pdf.Sentence


# In[125]:


import gensim

processed_docs = [cleancntxt(doc, stopwords) for doc in poems]
word_count_dict = gensim.corpora.Dictionary(processed_docs)
print ("In the corpus there are", len(word_count_dict), "unique tokens")
# word_count_dict.filter_extremes(no_below=5, no_above=0.2) # word must appear >5 times, and no more than 10% documents
# print "After filtering, in the corpus there are only", len(word_count_dict), "unique tokens"
bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]


# In[126]:


tfidf = models.TfidfModel(bag_of_words_corpus )
corpus_tfidf = tfidf[bag_of_words_corpus ]
lda_model = gensim.models.LdaModel(corpus_tfidf, num_topics=20, id2word=word_count_dict, passes=10)


# In[130]:


# 使用并行LDA加快处理速度。 
lda_model2 = gensim.models.ldamulticore.LdaMulticore(corpus=None, num_topics=20, id2word=word_count_dict,                                        workers=None, chunksize=2000, passes=1,                                         batch=False, alpha='symmetric', eta=None,                                         decay=0.5, offset=1.0, eval_every=10,                                         iterations=50, gamma_threshold=0.001, random_state=None)


# In[132]:


lda_model2.print_topics(3)


# In[133]:


topictermlist = lda_model2.print_topics(-1)
top_words = [[j.split('*')[1] for j in i[1].split(' + ')] for i in topictermlist] 
for k, i in enumerate(top_words): 
    print (k+1, " ".join(i) )


# In[137]:


perplexity_list = [fastInferTopicNumber(bag_of_words_corpus, num, word_count_dict) for num in [5, 15, 20, 25, 30, 35, 40 ]]


# In[138]:


plt.plot([5, 15, 20, 25, 30, 35, 40], perplexity_list)
plt.show()


# In[140]:


import pyLDAvis.gensim

song_data = pyLDAvis.gensim.prepare(lda_model, bag_of_words_corpus, word_count_dict)


# In[141]:


pyLDAvis.enable_notebook()
pyLDAvis.show(song_data)

