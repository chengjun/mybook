#!/usr/bin/env python
# coding: utf-8

# 
# 
# # 使用Jupyter制作Slides的介绍
# 
# 王成军 
# 
# wangchengjun@nju.edu.cn
# 
# 计算传播网 http://computational-communication.com
# 
# 
# 
# 
# 
# 

# ## 使用Nbviewer打开Slides
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/03-who-runs-China.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/0-jupyter-notebook.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/0-matplotlib-chinese.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/0-slides.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/0-turicreate.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/01-intro2cjc.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/02-bigdata.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/03-python-intro.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/03-UK-MPS-Scandal.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/03-umbrella-of-love.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-13chambers.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-beautifulsoup.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-cppcc.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-douban.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-fact-checking.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-gov-report.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-netease-music.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-pyppeteer.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-selenium-music-history.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-selenium-people-com-search.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-selenium.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-tripadvisor.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-wechat.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/04-crawler-weibo.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/06-data-cleaning-intro.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/06-data-cleaning-music-list.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/06-data-cleaning-occupy-central-news.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/06-data-cleaning-pandas.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/06-data-cleaning-preprocessing.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/06-data-cleaning-tweets.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-01-statistics-thinking.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-02-kl-divergence.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-02-linear-algebra.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-03-distributions.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-03-probability.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-04-hypothesis-inference.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-05-gradient-descent.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-06-regression.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-06-statsmodels.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-07-analyzing-titanic-dataset.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-07-covid19-pew-survey.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-08-covid19-grangercausality.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-09-survival-analysis.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/08-10-dowhy-estimation-methods.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-01-machine-learning-with-sklearn.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-03-hyperparameters-and-model-validation.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-04-feature-engineering.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-05-naive-bayes.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-06-linear-regression.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-07-support-vector-machines.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-08-random-forests.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-09-googleflustudy.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-10-future-employment.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-11-neural-network-intro.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-12-hand-written-digits.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-13-cnn.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-14-rnn.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-15-cifar10.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-16-pytorch_vgg_pretrained.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/09-grf.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/10-doc2vec.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/10-text-minning-gov-report.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/10-word2vec.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/11-1-sentiment-analysis-with-dict.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/11-2-emotion-dict.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/11-3-NRC-Chinese-dict.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/11-3-textblob.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/11-4-sentiment-classifier.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/11-5-LIWC.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/12-topic-models-update.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/12-topic-models-with-turicreate.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/13-recsys-intro-surprise.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/13-recsys-intro.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/13-recsys-latent-factor-model.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/14-millionsong.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/14-movielens.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/15-network-science-intro.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/16-network-science-models.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/17-networkx.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/18-02-network-diffusion.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/18-03-network-epidemics.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/18-04-seir-hcd-model.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/18-network-analysis-of-tianya-bbs.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/18-network-ecomplexity.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/18-network-ergm-siena.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/18-network-weibo-hot-search.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/19-facebook-ego-netwrok-visualization.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/19-visualization-datapane.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/19-visualization-datashader.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/19-visualization-maps-using-folium.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/19-visualization-pantheon.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/19-visualization-plotly-express.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/19-visualization-scientific-plot.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/19-visualization-with-pyecharts.ipynb
# - https://nbviewer.org/format/slides/github/chengjun/mybook/blob/main/19-visualization-with-seaborn.ipynb

# ## RISE: "Live" Reveal.js Jupyter/IPython Slideshow Extension
# https://github.com/damianavila/RISE

# ## Installation
# - Downnload from https://github.com/damianavila/RISE
# - open your teminal, cd to the RISE folder, e.g., 
# 
#     >## cd  /github/RISE/
# 
# - To install this nbextension, simply run 
# 
#     >## python setup.py install
# 
# from the RISE repository.

# In the notebook toolbar, a new button ("Enter/Exit Live Reveal Slideshow") will be available.
# 
# The notebook toolbar also contains a "Cell Toolbar" dropdown menu that gives you access to metadata for each cell. If you select the Slideshow preset, you will see in the right corner of each cell a little box where you can select the cell type (similar as for the static reveal slides with nbconvert).

# ## 将ipynb文件转为slides.html
# - download the reveal.js from Github https://github.com/hakimel/reveal.js
# - generate html using the following code
# - put the generated html into the reveal.js folder
# - open the html using chrome
# 

#     chengjuns-MacBook-Pro:~ chengjun$ cd github/cjc/code/
# 
#     chengjuns-MacBook-Pro:code chengjun$ jupyter nbconvert slides.ipynb --to slides --post serve

# ## 批量生成slides.html¶
#     chengjuns-MacBook-Pro:~ chengjun$ cd github/cjc/code/
#     
#     chengjuns-MacBook-Pro:code chengjun$ jupyter nbconvert *.ipynb --to slides

# ## 数学公式
# $E = MC^2$

# In[10]:


get_ipython().run_cell_magic('latex', '', '\\begin{align}\na = \\frac{1}{2}\\\\\n\\end{align}')


# ## 程序代码

# In[1]:


print 'hello world'


# In[2]:


for i in range(10):
    print i


# In[ ]:


# get a list of all the available magics


# In[21]:


get_ipython().run_line_magic('lsmagic', '')


# In[20]:


get_ipython().run_line_magic('env', '')
# to list your environment variables.


# In[11]:


get_ipython().run_line_magic('prun', '')


# In[15]:


get_ipython().run_line_magic('time', 'range(10)')


# In[14]:


get_ipython().run_line_magic('timeit', 'range(100)')


# !: to run a shell command. E.g., ! pip freeze | grep pandas to see what version of pandas is installed.
# 

# In[17]:


get_ipython().system(' cd /Users/chengjun/github/')


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
# to show matplotlib plots inline the notebook.
import matplotlib.pyplot as plt

plt.plot(range(10), range(10), 'r-o')
plt.show()


# ![](./images/end.png) 
