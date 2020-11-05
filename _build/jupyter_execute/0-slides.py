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
