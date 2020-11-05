#!/usr/bin/env python
# coding: utf-8

# # 基于字典的情感分析
# 
# 以下内容来自**邓旭东HIT** https://zhuanlan.zhihu.com/p/23225934
# 
# 情感分析就是分析一句话说得是很主观还是客观描述，分析这句话表达的是积极的情绪还是消极的情绪。

# ## 原理
# 比如这么一句话：
# 
# > “这手机的画面极好，操作也比较流畅。不过拍照真的太烂了！系统也不好。”
# 
# 

# ① 情感词
# 
# 要分析一句话是积极的还是消极的，最简单最基础的方法就是找出句子里面的情感词，积极的情感词比如：赞，好，顺手，华丽等，消极情感词比如：差，烂，坏，坑爹等。出现一个积极词就+1，出现一个消极词就-1。
# 里面就有“好”，“流畅”两个积极情感词，“烂”一个消极情感词。那它的情感分值就是1+1-1+1=2. 很明显这个分值是不合理的，下面一步步修改它。
# 
# 

# 
# 
# ② 程度词
# 
# “好”，“流畅”和‘烂“前面都有一个程度修饰词。”极好“就比”较好“或者”好“的情感更强，”太烂“也比”有点烂“情感强得多。所以需要在找到情感词后往前找一下有没有程度修饰，并给不同的程度一个权值。比如”极“，”无比“，”太“就要把情感分值*4，”较“，”还算“就情感分值*2，”只算“，”仅仅“这些就*0.5了。那么这句话的情感分值就是：4*1+1*2-1*4+1=3
# 
# 

# 
# 
# ③ 感叹号
# 
# 可以发现太烂了后面有感叹号，叹号意味着情感强烈。因此发现叹号可以为情感值+2. 那么这句话的情感分值就变成了：4*1+1*2-1*4-2+1 = 1
# 

# 
# 
# ④ 否定词
# 
# 明眼人一眼就看出最后面那个”好“并不是表示”好“，因为前面还有一个”不“字。所以在找到情感词的时候，需要往前找否定词。比如”不“，”不能“这些词。而且还要数这些否定词出现的次数，如果是单数，情感分值就*-1，但如果是偶数，那情感就没有反转，还是*1。在这句话里面，可以看出”好“前面只有一个”不“，所以”好“的情感值应该反转，*-1。
# 因此这句话的准确情感分值是：4*1+1*2-1*4-2+1*-1 = -1
# 
# 

# 
# ⑤ 积极和消极分开来
# 
# 再接下来，很明显就可以看出，这句话里面有褒有贬，不能用一个分值来表示它的情感倾向。而且这个权值的设置也会影响最终的情感分值，敏感度太高了。因此对这句话的最终的正确的处理，是得出这句话的一个积极分值，一个消极分值（这样消极分值也是正数，无需使用负数了）。它们同时代表了这句话的情感倾向。所以这句评论应该是”积极分值：6，消极分值：7“
# 

# 
# ⑥ 以分句的情感为基础
# 
# 再仔细一步，详细一点，一条评论的情感分值是由不同的分句加起来的，因此要得到一条评论的情感分值，就要先计算出评论中每个句子的情感分值。这条例子评论有四个分句，因此其结构如下（[积极分值, 消极分值]）：[[4, 0], [2, 0], [0, 6], [0, 1]] 
# 
# 以上就是使用情感词典来进行情感分析的主要流程了，算法的设计也会按照这个思路来实现。
# 

# ### 算法设计
# - 第一步：读取评论数据，对评论进行分句。
# - 第二步：查找对分句的情感词，记录积极还是消极，以及位置。
# - 第三步：往情感词前查找程度词，找到就停止搜寻。为程度词设权值，乘以情感值。
# - 第四步：往情感词前查找否定词，找完全部否定词，若数量为奇数，乘以-1，若为偶数，乘以1。
# - 第五步：判断分句结尾是否有感叹号，有叹号则往前寻找情感词，有则相应的情感值+2。
# - 第六步：计算完一条评论所有分句的情感值，用数组（list）记录起来。
# - 第七步：计算并记录所有评论的情感值。
# - 第八步：通过分句计算每条评论的积极情感均值，消极情感均值，积极情感方差，消极情感方差。

# In[1]:


import csv
import jieba
import numpy as np


# In[2]:


#打开词典文件，返回列表
def open_dict(Dict, path):
    path = path + '%s.txt' % Dict
    dictionary = open(path, 'r', encoding='utf-8')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict

def judgeodd(num):
    if (num % 2) == 0:
        return 'even'
    else:
        return 'odd'


# In[3]:


#修改成自己的path路径。
deny_word = open_dict(Dict = '否定词', path= r'./Textmining/')
posdict = open_dict(Dict = 'positive', path= r'./Textmining/')
negdict = open_dict(Dict = 'negative', path= r'./Textmining/')
degree_word = open_dict(Dict = '程度级别词语', path= r'./Textmining/')


# In[4]:


len(degree_word), degree_word[:3]


# In[5]:


mostdict = degree_word[degree_word.index('extreme')+1 : degree_word.index('very')]#权重4，即在情感词前乘以3
verydict = degree_word[degree_word.index('very')+1 : degree_word.index('more')]#权重3
moredict = degree_word[degree_word.index('more')+1 : degree_word.index('ish')]#权重2
ishdict = degree_word[degree_word.index('ish')+1 : degree_word.index('last')]#权重0.5


# In[6]:


mostdict[:5]


# In[7]:


def sentiment_score_list(dataset):
    seg_sentence = dataset.split('。')
    count1 = []
    count2 = []
    for sen in seg_sentence: #循环遍历每一个评论
        segtmp = jieba.lcut(sen, cut_all=False)  #把句子进行分词，以列表的形式返回
        i = 0 #记录扫描到的词的位置
        a = 0 #记录情感词的位置
        poscount = 0 #积极词的第一次分值
        poscount2 = 0 #积极词反转后的分值
        poscount3 = 0 #积极词的最后分值（包括叹号的分值）
        negcount = 0
        negcount2 = 0
        negcount3 = 0
        for word in segtmp:
            if word in posdict:  # 判断词语是否是情感词
                poscount += 1
                c = 0
                for w in segtmp[a:i]:  # 扫描情感词前的程度词
                    if w in mostdict:
                        poscount *= 4.0
                    elif w in verydict:
                        poscount *= 3.0
                    elif w in moredict:
                        poscount *= 2.0
                    elif w in ishdict:
                        poscount *= 0.5
                    elif w in deny_word:
                        c += 1
                if judgeodd(c) == 'odd':  # 扫描情感词前的否定词数
                    poscount *= -1.0
                    poscount2 += poscount
                    poscount = 0
                    poscount3 = poscount + poscount2 + poscount3
                    poscount2 = 0
                else:
                    poscount3 = poscount + poscount2 + poscount3
                    poscount = 0
                a = i + 1  # 情感词的位置变化

            elif word in negdict:  # 消极情感的分析，与上面一致
                negcount += 1
                d = 0
                for w in segtmp[a:i]:
                    if w in mostdict:
                        negcount *= 4.0
                    elif w in verydict:
                        negcount *= 3.0
                    elif w in moredict:
                        negcount *= 2.0
                    elif w in ishdict:
                        negcount *= 0.5
                    elif w in degree_word:
                        d += 1
                if judgeodd(d) == 'odd':
                    negcount *= -1.0
                    negcount2 += negcount
                    negcount = 0
                    negcount3 = negcount + negcount2 + negcount3
                    negcount2 = 0
                else:
                    negcount3 = negcount + negcount2 + negcount3
                    negcount = 0
                a = i + 1
            elif word == '！' or word == '!':  ##判断句子是否有感叹号
                for w2 in segtmp[::-1]:  # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    if w2 in posdict or negdict:
                        poscount3 += 2
                        negcount3 += 2
                        break
            i += 1 # 扫描词位置前移


            # 防止出现负数的情况
            pos_count = 0
            neg_count = 0
            if poscount3 < 0 and negcount3 > 0:
                neg_count += negcount3 - poscount3
                pos_count = 0
            elif negcount3 < 0 and poscount3 > 0:
                pos_count = poscount3 - negcount3
                neg_count = 0
            elif poscount3 < 0 and negcount3 < 0:
                neg_count = -poscount3
                pos_count = -negcount3
            else:
                pos_count = poscount3
                neg_count = negcount3

            count1.append([pos_count, neg_count])
        count2.append(count1)
        count1 = []

    return count2


# In[8]:


def sentiment_score(senti_score_list):
    score = []
    for review in senti_score_list:
        Pos_list, Neg_list = np.array(review).T
        Pos = np.sum(Pos_list)
        Neg = np.sum(Neg_list)
        score.append([Pos, Neg])
    return score


# In[9]:


data = '今天心情不好！股票又跌了，损失惨重,和女朋友也分手了！非常生气，我非常郁闷！！！！'
data2= '我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心!!!'

print(sentiment_score(sentiment_score_list(data)))
print(sentiment_score(sentiment_score_list(data2)))


# In[ ]:




