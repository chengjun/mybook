#!/usr/bin/env python
# coding: utf-8

# 
# # 第九章 推荐系统简介
# 

# ![image.png](images/recsys.png)

# ## 集体智慧编程
# 
# > 集体智慧是指为了创造新想法，将一群人的行为、偏好或思想组合在一起。一般基于聪明的算法（Netflix, Google）或者提供内容的用户(Wikipedia)。
# 
# 集体智慧编程所强调的是前者，即通过编写计算机程序、构造具有智能的算法收集并分析用户的数据，发现新的信息甚至是知识。
# 
# Toby Segaran, 2007, Programming Collective Intelligence. O'Reilly. 
# 
# https://github.com/computational-class/programming-collective-intelligence-code/blob/master/chapter2/recommendations.py

# ## 推荐系统
# 
# - 目前互联网世界最常见的智能产品形式。
# - 从信息时代过渡到注意力时代：
#     - 信息过载（information overload）
#     - 注意力稀缺
# - 推荐系统的基本任务是联系用户和物品，帮助用户快速发现有用信息，解决信息过载的问题。
#     - 针对长尾分布问题，找到个性化需求，优化资源配置
# 
# 

# ## 推荐系统的类型
# - 基于流行度的推荐
# - 社会化推荐（Social Recommendation）
#     - 让朋友帮助推荐物品
# - 基于内容的推荐 （Content-based filtering）
#     - 基于用户已经消费的物品内容，推荐新的物品。例如根据看过的电影的导演和演员，推荐新影片。
# - 基于协同过滤的推荐（collaborative filtering）
#     - 找到和某用户的历史兴趣一致的用户，根据这些用户之间的相似性或者他们所消费物品的相似性，为该用户推荐物品
# - 隐语义模型（latent factor model）
# - 基于图的随机游走算法（random walk on graphs）

# ## 协同过滤算法
# 
# - 基于邻域的方法（neighborhood-based method）
#     - 基于用户的协同过滤（user-based filtering）
#     - 基于物品的协同过滤 （item-based filtering）
# 

# ## UserCF和ItemCF的比较
# 
# - UserCF较为古老， 1992年应用于电子邮件个性化推荐系统Tapestry, 1994年应用于Grouplens新闻个性化推荐， 后来被Digg采用
#     - 推荐那些与个体有共同兴趣爱好的用户所喜欢的物品（群体热点，社会化）
#         - 反映用户所在小型群体中物品的热门程度
# - ItemCF相对较新，应用于电子商务网站Amazon和DVD租赁网站Netflix
#     - 推荐那些和用户之前喜欢的物品相似的物品 （历史兴趣，个性化）
#         - 反映了用户自己的兴趣传承
# - 新闻更新快，物品数量庞大，相似度变化很快，不利于维护一张物品相似度的表格，电影、音乐、图书则可以。
# 
# 

# ## 推荐系统评测
# - 用户满意度
# - 预测准确度
# 
#     $r_{ui}$用户实际打分， $\hat{r_{ui}}$推荐算法预测打分, T为测量次数
# 
#     - 均方根误差RMSE
#     
#         $RMSE = \sqrt{\frac{\sum_{u, i \in T} (r_{ui} - \hat{r_{ui}})}{ T }^2} $
#         
#     - 平均绝对误差MAE
#     
#         $  MAE = \frac{\sum_{u, i \in T} \left | r_{ui} - \hat{r_{ui}} \right|}{ T}$

# In[1]:


# A dictionary of movie critics and their ratings of a small
# set of movies
critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
      'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
      'The Night Listener': 3.0},
     'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
      'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
      'You, Me and Dupree': 3.5},
     'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
      'Superman Returns': 3.5, 'The Night Listener': 4.0},
     'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
      'The Night Listener': 4.5, 'Superman Returns': 4.0,
      'You, Me and Dupree': 2.5},
     'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
      'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
      'You, Me and Dupree': 2.0},
     'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
      'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
     'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}


# In[3]:


critics['Lisa Rose']['Lady in the Water']


# In[4]:


critics['Toby']['Snakes on a Plane']


# In[5]:


critics['Toby']


# 
# ## 1. User-based filtering
# 
# 
# ### 1.0 Finding similar users

# In[6]:


# 欧几里得距离
import numpy as np
np.sqrt(np.power(5-4, 2) + np.power(4-1, 2))


# - This formula calculates the distance, which will be smaller for people who are more similar. 
# - However, you need a function that gives higher values for people who are similar. 
# - This can be done by adding 1 to the function (so you don’t get a division-by-zero error) and inverting it:

# In[7]:


1.0 /(1 + np.sqrt(np.power(5-4, 2) + np.power(4-1, 2)) )


# In[8]:


# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):
    # Get the list of shared_items
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1
    # if they have no ratings in common, return 0
    if len(si)==0: return 0
    # Add up the squares of all the differences
    sum_of_squares=np.sum([np.power(prefs[person1][item]-prefs[person2][item],2) for item in si])
    return 1/(1+np.sqrt(sum_of_squares) )


# In[9]:


sim_distance(critics, 'Lisa Rose','Toby')


# ### Pearson Correlation Coefficient

# In[10]:


# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item]=1
    # Find the number of elements
    n=len(si)
    # if they are no ratings in common, return 0
    if n==0: return 0
    # Add up all the preferences
    sum1=np.sum([prefs[p1][it] for it in si])
    sum2=np.sum([prefs[p2][it] for it in si])
    # Sum up the squares
    sum1Sq=np.sum([np.power(prefs[p1][it],2) for it in si])
    sum2Sq=np.sum([np.power(prefs[p2][it],2) for it in si])
    # Sum up the products
    pSum=np.sum([prefs[p1][it]*prefs[p2][it] for it in si])
    # Calculate Pearson score
    num=pSum-(sum1*sum2/n)
    den=np.sqrt((sum1Sq-np.power(sum1,2)/n)*(sum2Sq-np.power(sum2,2)/n))
    if den==0: return 0
    return num/den


# In[22]:


sim_pearson(critics, 'Lisa Rose','Toby')


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
si = [item for item in critics['Lisa Rose'] if item in critics['Toby']]
score1 = [critics['Lisa Rose'][i] for i in si]
score2 = [critics['Toby'][i] for i in si]

plt.plot(score1, score2, 'ro');


# In[23]:


# Returns the best matches for person from the prefs dictionary.
# Number of results and similarity function are optional params.
def topMatches(prefs,person,n=5,similarity=sim_pearson):
    scores=[(similarity(prefs,person,other),other)
        for other in prefs if other!=person]
    # Sort the list so the highest scores appear at the top 
    scores.sort( )
    scores.reverse( )
    return scores[0:n]


# In[24]:


topMatches(critics,'Toby',n=3) # topN


# ## 1.1 Recommending Items
# 
# <div><img src="images/recsys2.png" width =1000></div>

#  Toby相似的五个用户（Rose, Reymour, Puig, LaSalle, Matthews）及相似度（依次为0.99， 0.38， 0.89， 0.92, 0.66）
# - 这五个用户看过的三个电影（Night,Lady, Luck）及其评分
#     - 例如，Rose对Night评分是3.0
# - S.xNight是用户相似度与电影评分的乘积
#     - 例如，Toby与Rose相似度(0.99) * Rose对Night评分是3.0 = 2.97
# - 可以得到每部电影的得分
#     - 例如，Night的得分是12.89 = 2.97+1.14+4.02+2.77+1.99
# - 电影得分需要使用用户相似度之和进行加权
#     - 例如，Night电影的预测得分是3.35 = 12.89/3.84
# 

# In[25]:


# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in prefs:
        # don't compare me to myself
        if other==person: continue
        sim=similarity(prefs,person,other)
        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:   
            # only score movies I haven't seen yet
            if item not in prefs[person]:# or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


# In[27]:


# Now you can find out what movies I should watch next:
getRecommendations(critics,'Toby')


# In[28]:


# You’ll find that the results are only affected very slightly by the choice of similarity metric.
getRecommendations(critics,'Toby',similarity=sim_distance)


# 
# ## 2. Item-based filtering
# 
# 
# Now you know how to find similar people and recommend products for a given person
# 
# **But what if you want to see which products are similar to each other?**
# 
# This is actually the same method we used earlier to determine similarity between people—

# ### 将item-user字典的键值翻转

# In[29]:


# you just need to swap the people and the items. 
def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

movies = transformPrefs(critics)


# ### 计算item的相似性

# In[30]:


topMatches(movies,'Superman Returns')


# ### 给item推荐user

# In[31]:


def calculateSimilarItems(prefs,n=10):
    # Create a dictionary of items showing which other items they
    # are most similar to.
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0: 
            print("%d / %d" % (c,len(itemPrefs)))
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,n=n,similarity=sim_distance)
        result[item]=scores
    return result

itemsim=calculateSimilarItems(critics) 
itemsim['Superman Returns']


# <div><img src="images/recsys3.png" width = 1200></div>

# Toby看过三个电影（snakes、Superman、dupree）和评分（依次是4.5、4.0、1.0）
# - 表格2-3给出这三部电影与另外三部电影的相似度
#     - 例如superman与night的相似度是0.103
# - R.xNight表示Toby对自己看过的三部定影的评分与Night这部电影相似度的乘积
#     - 例如，0.818 = 4.5*0.182
#     
#     
# - 那么Toby对于Night的评分可以表达为0.818+0.412+0.148 = 1.378
#     - 已经知道Night相似度之和是0.182+0.103+0.148 = 0.433
#         - 那么Toby对Night的最终评分可以表达为：1.378/0.433 = 3.183
# 

# In[32]:


def getRecommendedItems(prefs,itemMatch,user):
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
        # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
    # Divide each total score by total weighting to get an average
    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings

getRecommendedItems(critics,itemsim,'Toby')


# In[33]:


getRecommendations(movies,'Just My Luck')


# In[34]:


getRecommendations(movies, 'You, Me and Dupree')


# <img src = './img/itemcfNetwork.png' width = 700px>
# 
# **基于物品的协同过滤算法的网络表示方法**

# ## 基于图的模型
# 
# 使用二分图表示用户行为，因此基于图的算法可以应用到推荐系统当中。
# 
# <img src = './img/graphrec.png' width = 500px>

# In[35]:


# https://github.com/ParticleWave/RecommendationSystemStudy/blob/d1960056b96cfaad62afbfe39225ff680240d37e/PersonalRank.py
import os
import random

class Graph:
    def __init__(self):
        self.G = dict()
    
    def addEdge(self, p, q):
        if p not in self.G: self.G[p] = dict()
        if q not in self.G: self.G[q] = dict()
        self.G[p][q] = 1
        self.G[q][p] = 1

    def getGraphMatrix(self):
        return self.G


# In[36]:


graph = Graph()
graph.addEdge('A', 'a')
graph.addEdge('A', 'c')
graph.addEdge('B', 'a')
graph.addEdge('B', 'b')
graph.addEdge('B', 'c')
graph.addEdge('B', 'd')
graph.addEdge('C', 'c')
graph.addEdge('C', 'd')
G = graph.getGraphMatrix()
print(G.keys())


# In[37]:


G


# In[38]:


for i, ri in G.items():
    for j, wij in ri.items():
        print(i, j, wij) 


# In[39]:


def PersonalRank(G, alpha, root, max_step):
    # G is the biparitite graph of users' ratings on items
    # alpha is the probability of random walk forward
    # root is the studied User
    # max_step if the steps of iterations.
    rank = dict()
    rank = {x:0.0 for x in G.keys()}
    rank[root] = 1.0
    for k in range(max_step):
        tmp = {x:0.0 for x in G.keys()}
        for i,ri in G.items():
            for j,wij in ri.items():
                if j not in tmp: tmp[j] = 0.0 #
                tmp[j] += alpha * rank[i] / (len(ri)*1.0)
                if j == root: tmp[j] += 1.0 - alpha
        rank = tmp
        print(k, rank)
    return rank


# In[40]:


PersonalRank(G, 0.8, 'A', 20)
#    print(PersonalRank(G, 0.8, 'B', 20))
#    print(PersonalRank(G, 0.8, 'C', 20))


# 
# ## 3. MovieLens Recommender
# 
# MovieLens是一个电影评价的真实数据，由明尼苏达州立大学的GroupLens项目组开发。
# 
# ### 数据下载
# http://grouplens.org/datasets/movielens/1m/
# 
# > These files contain 1,000,209 anonymous ratings of approximately 3,900 movies 
# made by 6,040 MovieLens users who joined MovieLens in 2000.
# 

# **数据格式**
# 
# All ratings are contained in the file "ratings.dat" and are in the following format:
# 
# UserID::MovieID::Rating::Timestamp
# 
# 1::1193::5::978300760
# 
# 1::661::3::978302109
# 
# 1::914::3::978301968
# 

# In[41]:


def loadMovieLens(path='/Users/datalab/bigdata/cjc/ml-1m/'):
    # Get movie titles
    movies={}
    for line in open(path+'movies.dat', encoding = 'iso-8859-15'):
        (id,title)=line.split('::')[0:2]
        movies[id]=title
  
    # Load data
    prefs={}
    for line in open(path+'/ratings.dat'):
        (user,movieid,rating,ts)=line.split('::')
        prefs.setdefault(user,{})
        prefs[user][movies[movieid]]=float(rating)
    return prefs


# In[42]:


prefs=loadMovieLens()
prefs['87']


# ### user-based filtering

# In[43]:


getRecommendations(prefs,'87')[0:30]


# ### Item-based filtering

# In[44]:


itemsim=calculateSimilarItems(prefs,n=50)


# In[95]:


getRecommendedItems(prefs,itemsim,'87')[0:30]


# <div><img src=attachment:image.png width = 500 align='right'></div>
# 
# Libraries:
# - [Surprise](https://github.com/NicolasHug/Surprise): a Python scikit building and analyzing recommender systems that deal with explicit rating data.
# - [LightFM](https://github.com/lyst/lightfm): a hybrid recommendation algorithm in Python
# - [Python-recsys](https://github.com/ocelma/python-recsys): a Python library for implementing a recommender system
# 
# https://realpython.com/build-recommendation-engine-collaborative-filtering/
# 

# ![image.png](images/end.png)

# 
# ## Buiding Recommendation System with Turicreate
# 

# In this notebook we will import Turicreate and use it to
# 
# - train two models that can be used for recommending new songs to users 
# - compare the performance of the two models
# 

# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
import turicreate as tc
import matplotlib.pyplot as plt


# In[46]:


sf = tc.SFrame({'user_id': ["0", "0", "0", "1", "1", "2", "2", "2"],
                       'item_id': ["a", "b", "c", "a", "b", "b", "c", "d"],
                       'rating': [1, 3, 2, 5, 4, 1, 4, 3]})
sf


# In[47]:


m = tc.recommender.create(sf, target='rating')
recs = m.recommend()
recs


# ### The CourseTalk dataset: loading and first look
# 
# Loading of the CourseTalk database.

# In[48]:


train_file = '../data/ratings.dat'
sf = tc.SFrame.read_csv(train_file, header=False, 
                        delimiter='|', verbose=False)
sf = sf.rename({'X1':'user_id', 'X2':'course_id', 'X3':'rating'})
sf.show()


# In order to evaluate the performance of our model, we randomly split the observations in our data set into two partitions: we will use `train_set` when creating our model and `test_set` for evaluating its performance.

# In[49]:


sf


# In[50]:


train_set, test_set = sf.random_split(0.8, seed=1)


# ### Popularity model

# Create a model that makes recommendations using item popularity. When no target column is provided, the popularity is determined by the number of observations involving each item. When a target is provided, popularity is computed using the item’s mean target value. When the target column contains ratings, for example, the model computes the mean rating for each item and uses this to rank items for recommendations.
# 
# One typically wants to initially create a simple recommendation system that can be used as a baseline and to verify that the rest of the pipeline works as expected. The `recommender` package has several models available for this purpose. For example, we can create a model that predicts songs based on their overall popularity across all users.
# 

# In[51]:


popularity_model = tc.popularity_recommender.create(train_set, 'user_id', 'course_id', target = 'rating')


# ### Item similarity Model

# * [Collaborative filtering](http://en.wikipedia.org/wiki/Collaborative_filtering) methods make predictions for a given user based on the patterns of other users' activities. One common technique is to compare items based on their [Jaccard](http://en.wikipedia.org/wiki/Jaccard_index) similarity.This measurement is a ratio: the number of items they have in common, over the total number of distinct items in both sets.
# * We could also have used another slightly more complicated similarity measurement, called [Cosine Similarity](http://en.wikipedia.org/wiki/Cosine_similarity). 
# 
# If your data is implicit, i.e., you only observe interactions between users and items, without a rating, then use ItemSimilarityModel with Jaccard similarity.  
# 
# If your data is explicit, i.e., the observations include an actual rating given by the user, then you have a wide array of options.  ItemSimilarityModel with cosine or Pearson similarity can incorporate ratings.  In addition, MatrixFactorizationModel, FactorizationModel, as well as LinearRegressionModel all support rating prediction.  
# 
# Now data contains three columns: ‘user_id’, ‘item_id’, and ‘rating’.
# 
# itemsim_cosine_model = graphlab.recommender.create(data, 
#        target=’rating’, 
#        method=’item_similarity’, 
#        similarity_type=’cosine’)
#        
# factorization_machine_model = graphlab.recommender.create(data, 
#        target=’rating’, 
#        method=’factorization_model’)
# 
# 
# In the following code block, we compute all the item-item similarities and create an object that can be used for recommendations.

# In[52]:


item_sim_model = tc.item_similarity_recommender.create(
    train_set, 'user_id', 'course_id', target = 'rating', 
    similarity_type='cosine')


# ### Factorization Recommender Model
# Create a FactorizationRecommender that learns latent factors for each user and item and uses them to make rating predictions. This includes both standard matrix factorization as well as factorization machines models (in the situation where side data is available for users and/or items). [link](https://dato.com/products/create/docs/generated/graphlab.recommender.factorization_recommender.create.html#graphlab.recommender.factorization_recommender.create)

# In[53]:


factorization_machine_model = tc.recommender.factorization_recommender.create(
    train_set, 'user_id', 'course_id',                                                                    
    target='rating')


# ### Model Evaluation

# It's straightforward to use GraphLab to compare models on a small subset of users in the `test_set`. The [precision-recall](http://en.wikipedia.org/wiki/Precision_and_recall) plot that is computed shows the benefits of using the similarity-based model instead of the baseline `popularity_model`: better curves tend toward the upper-right hand corner of the plot. 
# 
# The following command finds the top-ranked items for all users in the first 500 rows of `test_set`. The observations in `train_set` are not included in the predicted items.

# In[54]:


result = tc.recommender.util.compare_models(
    test_set, [popularity_model, item_sim_model, factorization_machine_model],
    user_sample=.5, skip_set=train_set)


# Now let's ask the item similarity model for song recommendations on several users. We first create a list of users and create a subset of observations, `users_ratings`, that pertain to these users.

# In[41]:


K = 10
users = tc.SArray(sf['user_id'].unique().head(100))
users


# Next we use the `recommend()` function to query the model we created for recommendations. The returned object has four columns: `user_id`, `song_id`, the `score` that the algorithm gave this user for this song, and the song's rank (an integer from 0 to K-1). To see this we can grab the top few rows of `recs`:

# In[24]:


recs = item_sim_model.recommend(users=users, k=K)
recs.head()


# To learn what songs these ids pertain to, we can merge in metadata about each song.

# In[42]:


# Get the meta data of the courses
courses = tc.SFrame.read_csv('../data/cursos.dat', header=False, delimiter='|', verbose=False)
courses =courses.rename({'X1':'course_id', 'X2':'title', 'X3':'avg_rating', 
              'X4':'workload', 'X5':'university', 'X6':'difficulty', 'X7':'provider'})
courses.show()

courses = courses[['course_id', 'title', 'provider']]
results = recs.join(courses, on='course_id', how='inner')

#Populate observed user-course data with course info
userset = frozenset(users)
ix = sf['user_id'].apply(lambda x: x in userset, int)  
user_data = sf[ix]
user_data = user_data.join(courses, on='course_id')[['user_id', 'title', 'provider']]


# In[27]:


# Print out some recommendations 
for i in range(5):
    user = list(users)[i]
    print("User: " + str(i + 1))
    user_obs = user_data[user_data['user_id'] == user].head(K)
    del user_obs['user_id']
    user_recs = results[results['user_id'] == str(user)][['title', 'provider']]

    print("We were told that the user liked these courses: ")
    print (user_obs.head(K))

    print ("We recommend these other courses:")
    print (user_recs.head(K))

    print ("")


# ## Readings
# - (Looking for more details about the modules and functions? Check out the <a href="https://dato.com/products/create/docs/">API docs</a>.)
# - Toby Segaran, 2007, Programming Collective Intelligence. O'Reilly. Chapter 2 Making Recommendations
#     - programming-collective-intelligence-code/blob/master/chapter2/recommendations.py
# - 项亮 2012 推荐系统实践 人民邮电出版社

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# 13-recsys-latent-factor-model
# 13-recsys-intro-surprise
# 14-millionsong
# 14-movielens
# ```
# 
