#!/usr/bin/env python
# coding: utf-8

# 
# # 基于机器学习的情感分析
# 
# 
# ![image.png](images/author.png)

# <div><img src="images/emotion.png" align="right"></div>
# 
# ## Emotion
# Different types of emotion: anger, disgust, fear, joy, sadness, and surprise. The classification can be performed using different algorithms: e.g., naive Bayes classiﬁer trained on Carlo Strapparava and Alessandro Valitutti’s emotions lexicon.
# 
# 
# ## Polarity
# 
# To classify some text as positive or negative. In this case, the classification can be done by using a naive Bayes algorithm trained on Janyce Wiebe’s subjectivity lexicon.

# ![image.png](images/tweet.png)

# ## Sentiment Analysis with Sklearn
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


pos_tweets = [('I love this car', 'positive'),
    ('This view is amazing', 'positive'),
    ('I feel great this morning', 'positive'),
    ('I am so excited about the concert', 'positive'),
    ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
    ('This view is horrible', 'negative'),
    ('I feel tired this morning', 'negative'),
    ('I am not looking forward to the concert', 'negative'),
    ('He is my enemy', 'negative')]

test_tweets = [
    ('feel happy this morning', 'positive'),
    ('larry is my friend', 'positive'),
    ('I do not like that man', 'negative'),
    ('house is not great', 'negative'),
    ('your song is annoying', 'negative')]


# In[3]:


dat = []
for i in pos_tweets+neg_tweets+test_tweets:
    dat.append(i)
    
X = np.array(dat).T[0]
y = np.array(dat).T[1]


# In[4]:


get_ipython().run_line_magic('pinfo', 'TfidfVectorizer')


# In[5]:


vec = TfidfVectorizer(stop_words='english', ngram_range = (1, 1), lowercase = True)
X_vec = vec.fit_transform(X)
Xtrain = X_vec[:10]
Xtest = X_vec[10:]
ytrain = y[:10]
ytest= y[10:] 


# In[6]:


pd.DataFrame(X_vec.toarray(), columns=vec.get_feature_names())


# In[7]:


from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain.toarray(), ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest.toarray())              # 4. predict on new data
y_model


# In[8]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)


# In[9]:


from sklearn.svm import SVC
svc=SVC(kernel='rbf', gamma=1) # 超级参数 
svc.fit(Xtrain.toarray(), ytrain)                  # 3. fit model to data
y_model=svc.predict(Xtest.toarray())
accuracy_score(ytest, y_model)


# In[10]:


y_model 


# In[11]:


y_model=svc.predict(Xtest.toarray())


# In[14]:


# Don’t be too positive, let’s try another example:
vocabulary = vec.get_feature_names()

def classify_sentiment(str_list, model, vocabulary):
    # str_list = ['a str']
    vec_pred = TfidfVectorizer(stop_words='english', ngram_range = (1, 1), lowercase = True, vocabulary = vocabulary)
    return model.predict(vec_pred.fit_transform(str_list).toarray())

classify_sentiment(['Your song is annoying','larry is horrible'], model, vocabulary)


# In[15]:


classify_sentiment(['I do not like larry', 'larry is my friend'], svc, vocabulary)


# 作业
# 
# - 使用另外一种sklearn的分类器来对tweet_negative2进行情感分析
# 
# - 使用https://github.com/victorneo/Twitter-Sentimental-Analysis 所提供的推特数据进行情感分析，可以使用其代码 https://github.com/victorneo/Twitter-Sentimental-Analysis/blob/master/classification.py
# 
# - Sentiment Analysis of IMDb movie review Dataset Using Sklearn https://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch08/ch08.ipynb

# ## PaddlePaddle
#  
# <div><img src="images/paddlepaddle.png" align="right"></div>  飞桨（PaddlePaddle）以百度多年的深度学习技术研究和业务应用为基础，集深度学习核心框架、基础模型库、端到端开发套件、工具组件和服务平台于一体，2016 年正式开源，是全面开源开放、技术领先、功能完备的产业级深度学习平台。 
# 
# http://paddlepaddle.org
# 
# https://github.com/PaddlePaddle/book/tree/develop/06.understand_sentiment

# ## Turicreate
# 
# https://github.com/apple/turicreate
# 
# <div><img src="images/turicreate.png" align = "right"></div>
# Turi Create simplifies the development of custom machine learning models. You don't have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.
# 
# 
# https://apple.github.io/turicreate/docs/userguide/text_classifier/
# 
# https://www.kaggle.com/prakharrathi25/updated-turicreate-sentiment-analysis

# ![image.png](images/end.png)

# ## Creating Sentiment Classifier with Turicreate

# In this notebook, I will explain how to develop sentiment analysis classifiers that are based on a bag-of-words model. 
# Then, I will demonstrate how these classifiers can be utilized to solve Kaggle's "When Bag of Words Meets Bags of Popcorn" challenge.

# Using <del>GraphLab</del> Turicreate it is very easy and straight foward to create a sentiment classifier based on bag-of-words model. Given a dataset stored as a CSV file, you can construct your sentiment classifier using the following code: 

# In[ ]:


# toy code, do not run it
import turicreate as tc
train_data = tc.SFrame.read_csv(traindata_path,header=True, 
                                delimiter='\t',quote_char='"', 
                                column_type_hints = {'id':str, 
                                                     'sentiment' : int, 
                                                     'review':str } )
train_data['1grams features'] = tc.text_analytics.count_ngrams(
    train_data['review'],1)
train_data['2grams features'] = tc.text_analytics.count_ngrams(
    train_data['review'],2)
cls = tc.classifier.create(train_data, target='sentiment', 
                           features=['1grams features',
                                     '2grams features'])


# In the rest of this notebook, we will explain this code recipe in details, by demonstrating how this recipe can used to create IMDB movie reviews sentiment classifier.

# Before we begin constructing the classifiers, we need to import some Python libraries: turicreate (tc), and IPython display utilities.

# In[2]:


import turicreate as tc
from IPython.display import display
from IPython.display import Image


# ### IMDB movies reviews Dataset 
# 
# > Bag of Words Meets Bags of Popcorn
# 
# 

# Throughout this notebook, I will use Kaggle's IMDB movies reviews datasets that is available to download from the following link: https://www.kaggle.com/c/word2vec-nlp-tutorial/data. I downloaded labeledTrainData.tsv and testData.tsv files, and unzipped them to the following local files.
# 
# ###  DeepLearningMovies
# 
# Kaggle's competition for using Google's word2vec package for sentiment analysis
# 
# https://github.com/wendykan/DeepLearningMovies

# In[5]:


traindata_path = "/Users/datalab/bigdata/cjc/kaggle_popcorn_data/labeledTrainData.tsv"
testdata_path = "/Users/datalab/bigdata/cjc/kaggle_popcorn_data/testData.tsv"


# ### Loading Data

# We will load the data with IMDB movie reviews to an SFrame using SFrame.read_csv function.

# In[6]:


movies_reviews_data = tc.SFrame.read_csv(traindata_path,header=True, 
                                         delimiter='\t',quote_char='"', 
                                         column_type_hints = {'id':str, 
                                                              'sentiment' : str, 
                                                              'review':str } )


# By using the SFrame show function, we can visualize the data and notice that the train dataset consists of 12,500 positive and 12,500 negative, and overall 24,932 unique reviews.

# In[7]:


movies_reviews_data


# ### Constructing Bag-of-Words Classifier 

# One of the common techniques to perform document classification (and reviews classification) is using Bag-of-Words model, in which the frequency of each word in the document is used as a feature for training a classifier. GraphLab's text analytics toolkit makes it easy to calculate the frequency of each word in each review. Namely, by using the count_ngrams function with n=1, we can calculate the frequency of each word in each review. By running the following command:

# In[8]:


movies_reviews_data['1grams features'] = tc.text_analytics.count_ngrams(movies_reviews_data ['review'],1)


# By running the last command, we created a new column in movies_reviews_data SFrame object. In this column each value is a dictionary object, where each dictionary's keys are the different words which appear in the corresponding review, and the dictionary's values are the frequency of each word.
# We can view the values of this new column using the following command.

# In[9]:


movies_reviews_data#[['review','1grams features']]


# We are now ready to construct and evaluate the movie reviews sentiment classifier using the calculated above features. But first, to be able to perform a quick evaluation of the constructed classifier, we need to create labeled train and test datasets. We will create train and test datasets by randomly splitting the train dataset into two parts. The first part will contain 80% of the labeled train dataset and will be used as the training dataset, while the second part will contain 20% of the labeled train dataset and will be used as the testing dataset. We will create these two dataset by using the following command:  

# In[10]:


train_set, test_set = movies_reviews_data.random_split(0.8, seed=5)


# We are now ready to create a classifier using the following command:

# In[11]:


model_1 = tc.classifier.create(train_set, target='sentiment',                                features=['1grams features'])


# We can evaluate the performence of the classifier by evaluating it on the test dataset

# In[13]:


result1 = model_1.evaluate(test_set)


# In order to get an easy view of the classifier's prediction result, we define and use the following function

# In[14]:


def print_statistics(result):
    print( "*" * 30)
    print( "Accuracy        : ", result["accuracy"])
    print( "Confusion Matrix: \n", result["confusion_matrix"])
print_statistics(result1)


# As can be seen in the results above, in just a few relatively straight foward lines of code, we have developed a sentiment classifier that has accuracy of about ~0.88. Next, we demonstrate how we can improve the classifier accuracy even more.

# ### Improving The Classifier

# One way to improve the movie reviews sentiment classifier is to extract more meaningful features from the reviews. One method to add additional features, which might be meaningful, is to calculate the frequency of every two consecutive words in each review. To calculate the frequency of each two consecutive words in each review, as before, we will use turicreate's count_ngrams function only this time we will set n to be equal 2 (n=2) to create new column named '2grams features'.  

# In[15]:


movies_reviews_data['2grams features'] = tc.text_analytics.count_ngrams(movies_reviews_data['review'],2)


# In[16]:


movies_reviews_data


# As before, we will construct and evaluate a movie reviews sentiment classifier. However, this time we will use both the '1grams features' and the '2grams features' features

# In[17]:


train_set, test_set = movies_reviews_data.random_split(0.8, seed=5)
model_2 = tc.classifier.create(train_set, target='sentiment', features=['1grams features','2grams features'])
result2 = model_2.evaluate(test_set)


# In[18]:


print_statistics(result2)


# Indeed, the new constructed classifier seems to be more accurate with an accuracy of about ~0.9.

# ### Unlabeled Test File

# To test how well the presented method works, we will use all the 25,000 labeled IMDB movie reviews in the train dataset to construct a classifier. Afterwards, we will utilize the constructed classifier to predict sentiment for each review in the unlabeled dataset. Lastly, we will create a submission file according to Kaggle's guidelines and submit it. 

# In[19]:


traindata_path = "/Users/datalab/bigdata/cjc/kaggle_popcorn_data/labeledTrainData.tsv"
testdata_path = "/Users/datalab/bigdata/cjc/kaggle_popcorn_data/testData.tsv"
#creating classifier using all 25,000 reviews
train_data = tc.SFrame.read_csv(traindata_path,header=True, delimiter='\t',quote_char='"', 
                                column_type_hints = {'id':str, 'sentiment' : int, 'review':str } )
train_data['1grams features'] = tc.text_analytics.count_ngrams(train_data['review'],1)
train_data['2grams features'] = tc.text_analytics.count_ngrams(train_data['review'],2)

cls = tc.classifier.create(train_data, target='sentiment', features=['1grams features','2grams features'])
#creating the test dataset
test_data = tc.SFrame.read_csv(testdata_path,header=True, delimiter='\t',quote_char='"', 
                               column_type_hints = {'id':str, 'review':str } )
test_data['1grams features'] = tc.text_analytics.count_ngrams(test_data['review'],1)
test_data['2grams features'] = tc.text_analytics.count_ngrams(test_data['review'],2)

#predicting the sentiment of each review in the test dataset
test_data['sentiment'] = cls.classify(test_data)['class'].astype(int)

#saving the prediction to a CSV for submission
test_data[['id','sentiment']].save("/Users/datalab/bigdata/cjc/kaggle_popcorn_data/predictions.csv", format="csv")


# We then submitted the predictions.csv file to the Kaggle challange website and scored AUC of about 0.88.

# ### Further Readings

# Further reading materials can be found in the following links:
# 
# http://en.wikipedia.org/wiki/Bag-of-words_model
# 
# https://dato.com/products/create/docs/generated/graphlab.SFrame.html
# 
# https://dato.com/products/create/docs/graphlab.toolkits.classifier.html
# 
# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
# 
# Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). "Learning Word Vectors for Sentiment Analysis." The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
# 
