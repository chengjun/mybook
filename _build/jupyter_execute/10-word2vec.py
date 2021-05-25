#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# # 词向量模型简介
# 
# Introduction to Word Embeddings: Analyzing Meaning through Word Embeddings
# 
# 

# **Using vectors to represent things**
# - one of the most fascinating ideas in machine learning. 
# - Word2vec is a method to efficiently create word embeddings. 
#     - Mikolov et al. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
#     - Mikolov et al. (2013). [Distributed representations of words and phrases and their compositionality](https://arxiv.org/pdf/1310.4546.pdf)
# 
# 

# 
# 
# ## The Geometry of Culture
# 
# Analyzing Meaning through Word Embeddings
# 
# 
# Austin C. Kozlowski; Matt Taddy; James A. Evans
# 
# https://arxiv.org/abs/1803.09288
# 
# Word embeddings represent **semantic relations** between words as **geometric relationships** between vectors in a high-dimensional space, operationalizing a relational model of meaning consistent with contemporary theories of identity and culture. 

# 
# - Dimensions induced by word differences (e.g. man - woman, rich - poor, black - white, liberal - conservative) in these vector spaces closely correspond to dimensions of cultural meaning, 
# - Macro-cultural investigation with a longitudinal analysis of the coevolution of gender and class associations in the United States over the 20th century 
# 
# The success of these high-dimensional models motivates a move towards "high-dimensional theorizing" of meanings, identities and cultural processes.

# <img src= 'img/word2vec/gender_class.png' width= "700px">
# 

# ## HistWords 
# 
# HistWords is a collection of tools and datasets for analyzing language change using word vector embeddings. 
# 
# - The goal of this project is to facilitate quantitative research in diachronic linguistics, history, and the digital humanities.
# 
# 
# - We used the historical word vectors in HistWords to study the semantic evolution of more than 30,000 words across 4 languages. 
# 
# - This study led us to propose two statistical laws that govern the evolution of word meaning 
# 
# 
# https://nlp.stanford.edu/projects/histwords/
# 
# https://github.com/williamleif/histwords
# 
# 

# **Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change**
# 
# <img src= 'img/word2vec/wordpaths-final.png' width= "900px">

# ## Word embeddings quantify 100 years of gender and ethnic stereotypes
# 
# http://www.pnas.org/content/early/2018/03/30/1720347115
# 
# <img src= 'img/word2vec/sex.png' width= "500px">
# 

# ## The Illustrated Word2vec
# 
# Jay Alammar.  https://jalammar.github.io/illustrated-word2vec/

# ## Personality Embeddings
# 
# > What are you like?
# 
# **Big Five personality traits**: openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism
# - the five-factor model (FFM) 
# - **the OCEAN model**
# 
# 

# - 开放性（openness）：具有想象、审美、情感丰富、求异、创造、智能等特质。
# - 责任心（conscientiousness）：显示胜任、公正、条理、尽职、成就、自律、谨慎、克制等特点。
# - 外倾性（extraversion）：表现出热情、社交、果断、活跃、冒险、乐观等特质。
# - 宜人性（agreeableness）：具有信任、利他、直率、依从、谦虚、移情等特质。
# - 神经质或情绪稳定性（neuroticism）：具有平衡焦虑、敌对、压抑、自我意识、冲动、脆弱等情绪的特质，即具有保持情绪稳定的能力。

# In[1]:


# Personality Embeddings: What are you like?
jay = [-0.4, 0.8, 0.5, -0.2, 0.3]
john = [-0.3, 0.2, 0.3, -0.4, 0.9]
mike = [-0.5, -0.4, -0.2, 0.7, -0.1]


# ## Cosine Similarity
# The cosine of two non-zero vectors can be derived by using the Euclidean dot product formula:
# 
# $$
# \mathbf{A}\cdot\mathbf{B}
# =\left\|\mathbf{A}\right\|\left\|\mathbf{B}\right\|\cos\theta
# $$
# 

# $$
# \text{similarity} = \cos(\theta) = {\mathbf{A} \cdot \mathbf{B} \over \|\mathbf{A}\| \|\mathbf{B}\|} = \frac{ \sum\limits_{i=1}^{n}{A_i  B_i} }{ \sqrt{\sum\limits_{i=1}^{n}{A_i^2}}  \sqrt{\sum\limits_{i=1}^{n}{B_i^2}} },
# $$
# 
# where $A_i$ and $B_i$ are components of vector $A$ and $B$ respectively.

# In[2]:


from numpy import dot
from numpy.linalg import norm

def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

cos_sim([1, 0, -1], [-1,-1, 0])


# In[3]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity([[1, 0, -1]], [[-1,-1, 0]])


# $$CosineDistance = 1- CosineSimilarity$$

# In[6]:


from scipy import spatial
# spatial.distance.cosine computes 
# the Cosine distance between 1-D arrays.
1-spatial.distance.cosine([1, 0, -1], [-1,-1, 0])


# In[7]:


cos_sim(jay, john)


# In[8]:


cos_sim(jay, mike)


# Cosine similarity works for any number of dimensions. 
# - We can represent people (and things) as vectors of numbers (which is great for machines!).
# - We can easily calculate how similar vectors are to each other.

# ## Word Embeddings
# 

# ### Google News Word2Vec
# 
# You can download Google’s pre-trained model here.
# 
# - It’s 1.5GB! 
# - It includes word vectors for a vocabulary of 3 million words and phrases 
# - It is trained on roughly 100 billion words from a Google News dataset. 
# - The vector length is 300 features.
# 
# http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

# Using the **Gensim** library in python, we can 
# - find the most similar words to the resulting vector. 
# - add and subtract word vectors, 
# 

# In[9]:


import gensim
# Load Google's pre-trained Word2Vec model.
filepath = '/Users/datalab/bigdata/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True) 


# In[11]:


model['woman'][:3]


# In[12]:


model.most_similar('woman')


# In[13]:


model.similarity('woman', 'man')


# In[14]:


cos_sim(model['woman'], model['man'])


# In[15]:


model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)


# $$King- Queen = Man - Woman$$

# 
# <img src= 'img/word2vec/word2vec.png' width= "700px">

# Now that we’ve looked at trained word embeddings, 
# 
# - let’s learn more about the training process. 
# - But before we get to word2vec, we need to look at a conceptual parent of word embeddings: **the neural language model**.
# 
# 

# ## The neural language model
# 
# “You shall know a word by the company it keeps” J.R. Firth
# 
# 
# 
# > Bengio 2003 A Neural Probabilistic Language Model. Journal of Machine Learning Research. 3:1137–1155
# 
# After being trained, early neural language models (Bengio 2003) would calculate a prediction in three steps:
# 
# 

# <img src= 'img/word2vec/neural-language-model-prediction.png' width= "700px">
# 

# <img src= 'img/word2vec/bengio.png' width= "400px">

# The output of the neural language model is a probability score for all the words the model knows. 
# - We're referring to the probability as a percentage here, 
# - but 40% would actually be represented as 0.4 in the output vector.
# 

# ### Language Model Training
# 
# - We get a lot of text data (say, all Wikipedia articles, for example). then
# - We have a window (say, of three words) that we slide against all of that text.
# - The sliding window generates training samples for our model

# 
# <img src= 'img/word2vec/lm-sliding-window-4.png' width= "700px">

# As this window slides against the text, we (virtually) generate a dataset that we use to train a model. 

# Instead of only looking two words before the target word, we can also look at two words after it.
# 
# 
# 
# 
# <img src= 'img/word2vec/continuous-bag-of-words-example.png' width= "700px">
# 
# 

# If we do this, the dataset we’re virtually building and training the model against would look like this:
# 
# <img src= 'img/word2vec/continuous-bag-of-words-dataset.png' width= "700px">
# 
# This is called a **Continuous Bag of Words** (CBOW) https://arxiv.org/pdf/1301.3781.pdf

# ### Skip-gram
# Instead of guessing a word based on its context (the words before and after it), this other architecture tries to guess neighboring words using the current word. 
# 
# <img src= 'img/word2vec/skipgram.png' width= "700px">
# 
# https://arxiv.org/pdf/1301.3781.pdf

# 
# <img src= 'img/word2vec/cbow.png' width= "700px">

# 
# 
# <img src= 'img/word2vec/skipgram-sliding-window-samples.png' width= "700px">

# The pink boxes are in different shades because this sliding window actually creates four separate samples in our training dataset.
# 
# 
# - We then slide our window to the next position:
# - Which generates our next four examples:
# 
# 

# 
# <img src= 'img/word2vec/skipgram-language-model-training.png' width= "700px">

# 
# 
# <img src= 'img/word2vec/skipgram-language-model-training-4.png' width= "700px">

# 
# 
# <img src= 'img/word2vec/skipgram-language-model-training-5.png' width= "700px">

# 
# 
# <img src= 'img/word2vec/language-model-expensive.png' width= "700px">

# ### Negative Sampling
# 

# And switch it to a model that takes the input and output word, and outputs a score indicating **if they’re neighbors or not** 
# - 0 for “not neighbors”, 1 for “neighbors”.
# 
# 
# 
# <img src= 'img/word2vec/are-the-words-neighbors.png' width= "700px">

# 
# <img src= 'img/word2vec/word2vec-negative-sampling-2.png' width= "700px">

# we need to introduce negative samples to our dataset
# - samples of words that are not neighbors. 
# - Our model needs to return 0 for those samples.
# - This leads to a great tradeoff of computational and statistical efficiency.
# 
# **Skipgram with Negative Sampling (SGNS)**
# 

# ### Word2vec Training Process
# 
# 
# <img src= 'img/word2vec/word2vec-training-update.png' width= "700px">

# ## Pytorch word2vec 
# https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.ipynb
# 
# https://github.com/bamtercelboo/pytorch_word2vec/blob/master/model.py

# In[118]:


# see http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


# In[202]:


text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""

text = text.replace(',', '').replace('.', '').lower().split()


# In[203]:


# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(text)
vocab_size = len(vocab)
print('vocab_size:', vocab_size)

w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}


# In[204]:


# context window size is two
def create_cbow_dataset(text):
    data = []
    for i in range(2, len(text) - 2):
        context = [text[i - 2], text[i - 1],
                   text[i + 1], text[i + 2]]
        target = text[i]
        data.append((context, target))
    return data

cbow_train = create_cbow_dataset(text)
print('cbow sample', cbow_train[0])


# In[205]:


def create_skipgram_dataset(text):
    import random
    data = []
    for i in range(2, len(text) - 2):
        data.append((text[i], text[i-2], 1))
        data.append((text[i], text[i-1], 1))
        data.append((text[i], text[i+1], 1))
        data.append((text[i], text[i+2], 1))
        # negative sampling
        for _ in range(4):
            if random.random() < 0.5 or i >= len(text) - 3:
                rand_id = random.randint(0, i-1)
            else:
                rand_id = random.randint(i+3, len(text)-1)
            data.append((text[i], text[rand_id], 0))
    return data


skipgram_train = create_skipgram_dataset(text)
print('skipgram sample', skipgram_train[0])


# In[206]:


class CBOW(nn.Module):
    def __init__(self, vocab_size, embd_size, context_size, hidden_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.linear1 = nn.Linear(2*context_size*embd_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, inputs):
        embedded = self.embeddings(inputs).view((1, -1))
        hid = F.relu(self.linear1(embedded))
        out = self.linear2(hid)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs
    
    def extract(self, inputs):
        embeds = self.embeddings(inputs)
        return embeds


# In[207]:


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
    
    def forward(self, focus, context):
        embed_focus = self.embeddings(focus).view((1, -1)) # input
        embed_ctx = self.embeddings(context).view((1, -1)) # output
        score = torch.mm(embed_focus, torch.t(embed_ctx)) # input*output
        log_probs = F.logsigmoid(score) # sigmoid
        return log_probs
    
    def extract(self, focus):
        embed_focus = self.embeddings(focus)
        return embed_focus


# `torch.mm` Performs a matrix multiplication of the matrices 
# 
# `torch.t` Expects :attr:`input` to be a matrix (2-D tensor) and transposes dimensions 0
# and 1. Can be seen as a short-hand function for ``transpose(input, 0, 1)``.

# In[208]:


embd_size = 100
learning_rate = 0.001
n_epoch = 30
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right


# In[209]:


def train_cbow():
    hidden_size = 64
    losses = []
    loss_fn = nn.NLLLoss()
    model = CBOW(vocab_size, embd_size, CONTEXT_SIZE, hidden_size)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(n_epoch):
        total_loss = .0
        for context, target in cbow_train:
            ctx_idxs = [w2i[w] for w in context]
            ctx_var = Variable(torch.LongTensor(ctx_idxs))

            model.zero_grad()
            log_probs = model(ctx_var)

            loss = loss_fn(log_probs, Variable(torch.LongTensor([w2i[target]])))

            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
        losses.append(total_loss)
    return model, losses 


# In[210]:


def train_skipgram():
    losses = []
    loss_fn = nn.MSELoss()
    model = SkipGram(vocab_size, embd_size)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epoch):
        total_loss = .0
        for in_w, out_w, target in skipgram_train:
            in_w_var = Variable(torch.LongTensor([w2i[in_w]]))
            out_w_var = Variable(torch.LongTensor([w2i[out_w]]))
            
            model.zero_grad()
            log_probs = model(in_w_var, out_w_var)
            loss = loss_fn(log_probs[0], Variable(torch.Tensor([target])))
            
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
        losses.append(total_loss)
    return model, losses


# In[211]:


cbow_model, cbow_losses = train_cbow()
sg_model, sg_losses = train_skipgram()


# In[212]:


plt.figure(figsize= (10, 4))
plt.subplot(121)
plt.plot(range(n_epoch), cbow_losses, 'r-o', label = 'CBOW Losses')
plt.legend()
plt.subplot(122)
plt.plot(range(n_epoch), sg_losses, 'g-s', label = 'SkipGram Losses')
plt.legend()
plt.tight_layout()


# In[213]:


cbow_vec = cbow_model.extract(Variable(torch.LongTensor([v for v in w2i.values()])))
cbow_vec = cbow_vec.data.numpy()
len(cbow_vec[0])


# In[214]:


sg_vec = sg_model.extract(Variable(torch.LongTensor([v for v in w2i.values()])))
sg_vec = sg_vec.data.numpy()
len(sg_vec[0])


# In[217]:


# 利用PCA算法进行降维
from sklearn.decomposition import PCA
X_reduced = PCA(n_components=2).fit_transform(sg_vec)

# 绘制所有单词向量的二维空间投影
import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure(figsize = (20, 10))
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.4, color = 'white')
# 绘制几个特殊单词的向量
words = list(w2i.keys())
# 设置中文字体，否则无法在图形上显示中文
for w in words:
    if w in w2i:
        ind = w2i[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'red')
        plt.text(xy[0], xy[1], w, alpha = 1, color = 'white', fontsize = 20)


# ## NGram词向量模型
# 
# 本文件是集智AI学园http://campus.swarma.org 出品的“火炬上的深度学习”第VI课的配套源代码
# 
# 原理：利用一个人工神经网络来根据前N个单词来预测下一个单词，从而得到每个单词的词向量
# 
# 以刘慈欣著名的科幻小说《三体》为例，来展示利用NGram模型训练词向量的方法
# - 预处理分为两个步骤：1、读取文件、2、分词、3、将语料划分为N＋1元组，准备好训练用数据
# - 在这里，我们并没有去除标点符号，一是为了编程简洁，而是考虑到分词会自动将标点符号当作一个单词处理，因此不需要额外考虑。

# In[35]:


with open("../data/3body.txt", 'r') as f:
    text = str(f.read())


# In[37]:


import jieba, re
temp = jieba.lcut(text)
words = []
for i in temp:
    #过滤掉所有的标点符号
    i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：]+", "", i)
    if len(i) > 0:
        words.append(i)
print(len(words))


# In[36]:


text[:100]


# In[44]:


print(*words[:50])


# In[38]:


trigrams = [([words[i], words[i + 1]], words[i + 2]) for i in range(len(words) - 2)]
# 打印出前三个元素看看
print(trigrams[:3])


# In[84]:


# 得到词汇表
vocab = set(words)
print(len(vocab))
word_to_idx = {i:[k, 0] for k, i in enumerate(vocab)} 
idx_to_word = {k:i for k, i in enumerate(vocab)}
for w in words:
     word_to_idx[w][1] +=1


# 构造NGram神经网络模型 (三层的网络)
# 
# 1. 输入层：embedding层，这一层的作用是：先将输入单词的编号映射为一个one hot编码的向量，形如：001000，维度为单词表大小。
# 然后，embedding会通过一个线性的神经网络层映射到这个词的向量表示，输出为embedding_dim
# 2. 线性层，从embedding_dim维度到128维度，然后经过非线性ReLU函数
# 3. 线性层：从128维度到单词表大小维度，然后log softmax函数，给出预测每个单词的概率

# In[87]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch

class NGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  #嵌入层
        self.linear1 = nn.Linear(context_size * embedding_dim, 128) #线性层
        self.linear2 = nn.Linear(128, vocab_size) #线性层

    def forward(self, inputs):
        #嵌入运算，嵌入运算在内部分为两步：将输入的单词编码映射为one hot向量表示，然后经过一个线性层得到单词的词向量
        embeds = self.embeddings(inputs).view(1, -1)
        # 线性层加ReLU
        out = F.relu(self.linear1(embeds))
        
        # 线性层加Softmax
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs
    def extract(self, inputs):
        embeds = self.embeddings(inputs)
        return embeds


# In[89]:


losses = [] #纪录每一步的损失函数
criterion = nn.NLLLoss() #运用负对数似然函数作为目标函数（常用于多分类问题的目标函数）
model = NGram(len(vocab), 10, 2) #定义NGram模型，向量嵌入维数为10维，N（窗口大小）为2
optimizer = optim.SGD(model.parameters(), lr=0.001) #使用随机梯度下降算法作为优化器 
#循环100个周期
for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        # 准备好输入模型的数据，将词汇映射为编码
        context_idxs = [word_to_idx[w][0] for w in context]
        # 包装成PyTorch的Variable
        context_var = Variable(torch.LongTensor(context_idxs))
        # 清空梯度：注意PyTorch会在调用backward的时候自动积累梯度信息，故而每隔周期要清空梯度信息一次。
        optimizer.zero_grad()
        # 用神经网络做计算，计算得到输出的每个单词的可能概率对数值
        log_probs = model(context_var)
        # 计算损失函数，同样需要把目标数据转化为编码，并包装为Variable
        loss = criterion(log_probs, Variable(torch.LongTensor([word_to_idx[target][0]])))
        # 梯度反传
        loss.backward()
        # 对网络进行优化
        optimizer.step()
        # 累加损失函数值
        total_loss += loss.data
    losses.append(total_loss)
    print('第{}轮，损失函数为：{:.2f}'.format(epoch, total_loss.numpy()[0]))


#  12m 24s!!!

# In[91]:


# 从训练好的模型中提取每个单词的向量
vec = model.extract(Variable(torch.LongTensor([v[0] for v in word_to_idx.values()])))
vec = vec.data.numpy()

# 利用PCA算法进行降维
from sklearn.decomposition import PCA
X_reduced = PCA(n_components=2).fit_transform(vec)


# In[107]:


# 绘制所有单词向量的二维空间投影
import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure(figsize = (20, 10))
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.4, color = 'white')
# 绘制几个特殊单词的向量
words = ['智子', '地球', '三体', '质子', '科学', '世界', '文明', '太空', '加速器', '平面', '宇宙', '信息']
# 设置中文字体，否则无法在图形上显示中文
zhfont1 = matplotlib.font_manager.FontProperties(fname='/Library/Fonts/华文仿宋.ttf', size = 35)
for w in words:
    if w in word_to_idx:
        ind = word_to_idx[w][0]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'red')
        plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'white')


# In[109]:


# 定义计算cosine相似度的函数
import numpy as np
def cos_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    norm = norm1 * norm2
    dot = np.dot(vec1, vec2)
    result = dot / norm if norm > 0 else 0
    return result
    
# 在所有的词向量中寻找到与目标词（word）相近的向量，并按相似度进行排列
def find_most_similar(word, vectors, word_idx):
    vector = vectors[word_to_idx[word][0]]
    simi = [[cos_similarity(vector, vectors[num]), key] for num, key in enumerate(word_idx.keys())]
    sort = sorted(simi)[::-1]
    words = [i[1] for i in sort]
    return words

# 与智子靠近的词汇
find_most_similar('智子', vec, word_to_idx)[:10]


# ## Gensim Word2vec 

# In[16]:


import gensim as gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence


# In[20]:


f = open("./data/三体.txt", 'r')
lines = []

import jieba
import re

for line in f:
    temp = jieba.lcut(line)
    words = []
    for i in temp:
        #过滤掉所有的标点符号
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)


# In[21]:


# 调用gensim Word2Vec的算法进行训练。
# 参数分别为：size: 嵌入后的词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(lines, size = 20, window = 2 , min_count = 0)


# In[23]:


model.wv.most_similar('三体', topn = 100)


# In[26]:


# 将词向量投影到二维空间
import numpy as np
from sklearn.decomposition import PCA

rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.vocab):
    rawWordVec.append(model[w])
    word2ind[w] = i
rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)


# In[34]:


# 绘制星空图
# 绘制所有单词向量的二维空间投影
import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure(figsize = (15, 10))
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.5, color = 'white')
# 绘制几个特殊单词的向量
words = ['智子', '地球', '三体', '质子', '科学', '世界', '文明', '太空', '加速器', '平面', '宇宙', '进展','的']
# 设置中文字体，否则无法在图形上显示中文
#zhfont1 = matplotlib.font_manager.FontProperties(fname='/Library/Fonts/华文仿宋.ttf', size=26)
for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'red')
        #plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'yellow')
        plt.text(xy[0], xy[1], w, alpha = 1, color = 'yellow', fontsize = 16)


# In[117]:


# 绘制星空图
# 绘制所有单词向量的二维空间投影
fig = plt.figure(figsize = (15, 10))
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.3, color = 'white')
# 绘制几个特殊单词的向量
words = ['智子', '地球', '三体', '质子', '科学', '世界', '文明', '太空', '加速器', '平面', '宇宙', '进展','的']
# 设置中文字体，否则无法在图形上显示中文
zhfont1 = matplotlib.font_manager.FontProperties(fname='/Library/Fonts/华文仿宋.ttf', size=26)
for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'red')
        plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'yellow')


# ![](images/end.png)
