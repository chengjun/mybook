#!/usr/bin/env python
# coding: utf-8

# 
# #  Latent Factor Recommender System
# 
# 
# ![image.png](images/author.png)

# The NetFlix Challenge
# 
# **Training data**  100 million ratings, 480,000 users, 17,770 movies. 6 years of data: 2000-2005
# 
# **Test data**  Last few ratings of each user (2.8 million)
# 
# **Competition**  2,700+ teams, $1 million prize for 10% improvement on Netflix

# - Evaluation criterion: Root Mean Square Error (RMSE) 
# 
# \begin{equation}
#   RMSE =  \frac{1}{|R|} \sqrt{\sum_{(i, x)\in R}(\hat{r}_{xi} - r_{xi})^2}
# \end{equation}
# - Netflix’s system RMSE: 0.9514

# ![image.png](images/latent.png)

# ![image.png](images/latent2.png)

# ![image.png](images/latent3.png)

# ![image.png](images/latent4.png)

# ![image.png](images/latent5.png)
# 
# U (m x m) , $\Sigma$(m x n),   $V^T$ (n x n)

# In[74]:


import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# Singular-value decomposition
U, s, VT = np.linalg.svd(A)
# create n x n Sigma matrix
Sigma = np.diag(s)
# reconstruct matrix
PT = Sigma.dot(VT)
#B = U.dot(Sigma.dot(VT))
print(PT)


# $\Sigma$本来应该跟A矩阵的大小一样，但linalg.svd()只返回了一个行向量的$\Sigma$，并且舍弃值为0的奇异值。因此，必须先将$\Sigma$转化为矩阵。
# 
# ![image.png](images/latent6.png)

# In[91]:


# Singular-value decomposition 
A = np.array([[1, 2], [3, 4], [5, 6]])
U, s, VT = np.linalg.svd(A)
# create n x n Sigma matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = np.diag(s)
# reconstruct matrix
PT = Sigma.dot(VT)
B = U.dot(PT)

print('A = \n', A, '\n')
print('U = \n', U, '\n')
print('Sigma = \n', Sigma, '\n')
print('VT = \n', VT, '\n')
print('PT = \n', PT, '\n')
print('B = \n', B, '\n') 


# In[107]:


# Singular-value decomposition
A = np.array([[1, 2, 3], 
              [4, 5, 6]])
U,S,VT = np.linalg.svd(A)
# create n x n Sigma matrix
Sigma = np.zeros((A.shape[1], A.shape[1]))
# populate Sigma with n x n diagonal matrix
if A.shape[1] > S.shape[0]:
    S = np.append(S, 0)
Sigma[:A.shape[1], :A.shape[1]] = np.diag(S)

PT= Sigma.dot(VT)
PT = PT[0:A.shape[0]]
B = U.dot(PT)
print('A = \n', A, '\n')
print('U = \n', U, '\n')
print('Sigma = \n', Sigma, '\n')
print('VT = \n', VT, '\n')
print('PT = \n', PT, '\n')
print('B = \n', B, '\n')


# ![image.png](images/latent7.png)

# ![image.png](images/latent8.png)

# SVD gives minimum reconstruction error (Sum of Squared Errors, **SSE**)
# 
# SSE and RMSE are monotonically related:$RMSE=\frac{1}{c}\sqrt{SSE}$
#  
# Great news: SVD is minimizing RMSE

# In[226]:


# https://beckernick.github.io/matrix-factorization-recommender/
import pandas as pd
import numpy as np

ratings_list = [i.strip().split("::") for i in open('/Users/datalab/bigdata/cjc/ml-1m/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('/Users/datalab/bigdata/cjc/ml-1m/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('/Users/datalab/bigdata/cjc/ml-1m/movies.dat', 'r', encoding = 'iso-8859-15').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])

movies_df['MovieID'] = movies_df['MovieID'].astype('int64')
ratings_df['UserID'] = ratings_df['UserID'].astype('int64')
ratings_df['MovieID'] = ratings_df['MovieID'].astype('int64')


# In[219]:


movies_df.head()


# In[241]:


ratings_df.head()


# In[227]:


# 注意：使用0填充缺失值
R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
R_df.head()


# In[228]:


R = R_df.to_numpy(dtype=np.int16)
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


# In[229]:


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)


# In[230]:


sigma = np.diag(sigma)

all_user_predicted_ratings = U.dot( sigma.dot(Vt)) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)


# In[239]:


preds_df
# each row is a user
# each column is a movie


# In[246]:


def recommend_movies(preds_df, user_row_number, movies_df, ratings_df, num_recommendations=5):
    # Get and sort the user's predictions
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    userID = user_row_number + 1
    user_data = ratings_df[ratings_df.UserID == userID]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )

    print('UserID {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    potential_movie_df= movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])]
    predicted_movie_df = pd.DataFrame(sorted_user_predictions).reset_index()
    predicted_movie_df['MovieID'] = predicted_movie_df['MovieID'].astype('int64')
    recommendations = (
        potential_movie_df.merge(predicted_movie_df, how = 'left', on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations 


# In[247]:


already_rated, predictions = recommend_movies(preds_df, 0, movies_df, ratings_df, 10)


# In[238]:


already_rated[:3]


# In[237]:


predictions


# 比较三种矩阵分解的方法
# - 特征值分解 Eigen value decomposition
#     - 只能用于方阵
# - 奇异值分解 Singular value decomposition
#     - 需要填充稀疏矩阵中的缺失元素
#     - 计算复杂度高 $O(mn^2)$
# - 梯度下降 Gradient Descent
#     - 广泛使用！

# ![image.png](images/latent9.png)

# ## Including bias
# 
# ![image.png](images/latent10.png)
# 
# \begin{equation}
#   \hat{r}_{xi}= u + b_x + b_i + q_i p_x^{T}
# \end{equation}
# 
# - $u$ is the global bias, measured by the overall mean rating
# - $b_x$ is the bias for user x, measured by the mean rating given by user x.
# - $b_i$ is the bias for movie i, measured by the mean ratings of movie i.
# - $q_i p_{x}^{T}$ is the user-movie interaction

# ![image.png](images/latent11.png)

# ![image.png](images/latent12.png)

# ![image.png](images/latent13.png)

# <div><img src=attachment:image.png width = 800></div>

# ## Further reading:
# Y. Koren, Collaborative filtering with temporal dynamics, KDD ’09
# - http://www2.research.att.com/~volinsky/netflix/bpc.html
# - http://www.the-ensemble.com/
# 

# ![](images/recsys14.png)

# ![image.png](images/end.png)
