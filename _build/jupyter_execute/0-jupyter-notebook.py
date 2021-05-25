#!/usr/bin/env python
# coding: utf-8

# # Jupyter Notebook 
# 
# It is a **```web application```** that allows you to create and share documents that contain 
# - live code
# - equations
# - visualizations
# - explanatory text
# 
# 

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Jupyter-Notebook" data-toc-modified-id="Jupyter-Notebook-1">Jupyter Notebook</a></span><ul class="toc-item"><li><span><a href="#Copy-and-Paste-images" data-toc-modified-id="Copy-and-Paste-images-1.1">Copy and Paste images</a></span></li><li><span><a href="#Set-the-layout-of-figures" data-toc-modified-id="Set-the-layout-of-figures-1.2">Set the layout of figures</a></span></li><li><span><a href="#Publish-on-Github-Pages" data-toc-modified-id="Publish-on-Github-Pages-1.3">Publish on Github Pages</a></span></li><li><span><a href="#Workflow" data-toc-modified-id="Workflow-1.4">Workflow</a></span></li><li><span><a href="#引用" data-toc-modified-id="引用-1.5">引用</a></span><ul class="toc-item"><li><span><a href="#搜狗输入法表情和符号" data-toc-modified-id="搜狗输入法表情和符号-1.5.1">搜狗输入法表情和符号</a></span></li></ul></li><li><span><a href="#运行C代码" data-toc-modified-id="运行C代码-1.6">运行C代码</a></span></li><li><span><a href="#Jupyter-魔术命令" data-toc-modified-id="Jupyter-魔术命令-1.7">Jupyter 魔术命令</a></span></li><li><span><a href="#References" data-toc-modified-id="References-1.8">References</a></span></li></ul></li></ul></div>

# ## Copy and Paste images
# 
#     ![image.png](attachment:image.png)

# ## Set the layout of figures
# 
# - first copy and paste a figure into notebook, and you will get this line of script: 
# 
# ```
# ![image.png](attachment:image.png)
# ```
# 
# - Second, replace it with the following script:
# 
# ```        
# <div>
# <img src="attachment:image.png"
#      align="right" width = "200px"/>
# </div>
# ```
# 
# 

# ```
# <div>
# <img src="attachment:image.png" align="right" width = "200px"/>
# </div>
# ```
# > “Know thyself.” ― Socrates
# 
# > “The unexamined life is not worth living.” ― Socrates 
# 
# > “Wonder is the beginning of wisdom.” ― Socrates
# 
# >  “True wisdom comes to each of us when we realize how little we understand about life, ourselves, and the world around us.” 
# ― Socrates
# 

# ## Publish on Github Pages
# 
# 1. Confirm that your book's HTML is built in the **_site** folder.
# 2. Install the ghp-import tool.
# 
# ```bash
# pip install ghp-import
# ```
# 
# 3. Use ghp-import to push your book's HTML onto the gh-pages branch of your repository.
# 
# ```bash
# ghp-import -n -p -f _site
# ```
# 
# https://jupyterbook.org/guide/publish/github-pages.html
# 

# ## Workflow
# 
# ```bash
# #!/bin/bash
# # open github master branch
# # open atom master branch
# cd ..
# jupyter-book build ccbook/
# cd ccbook
# make serve
# 
# # Publish your book's HTML manually to GitHub-pages
# # publish the _site folder of master 
# # branch to gh-pages branch
# 
# ghp-import -n -p -f _site
# ```

# In[1]:


# my first python script
print("hello world! \n I am Cheng-Jun Wang.")


# Uses include: 
# - data cleaning and transformation, 
# - numerical simulation, 
# - statistical modeling, 
# - machine learning 
# - and much more.
# 

# In[2]:


print('hello world')


# In[3]:


1 + 1


# $E = MC^2$

# \begin{align}
# \dot{x} & = \sigma(y-x) \\
# \dot{y} & = \rho x - y - xz \\
# \dot{z} & = -\beta z + xy
# \end{align}

# ## 引用
# 
# Because jupyter-book is built on top of Jekyll, we can use the excellent `jekyll-scholar` plugin to include citations and a bibliography with your book.
# 
# Note: It only works if you're building your book HTML locally and hosting the HTML files online somewhere. 
# 
# 
# 
# 
# {% cite holdgraf_evidence_2014 %}
# 
# For example, this text: {% cite holdgraf_evidence_2014 %} generates this citation:
# 
# 
# https://jupyterbook.org/features/citations.html
# 

# ```
# # 一级标题
# ## 二级标题
# 
# [复旦大学](http://www.fdu.edu.cn)是一个*非常棒*的大学！
# 
# 1. point 1
# 1. point 2
# 1. point 3
# 
# ```

# > Here's an example of an epigraph quote. Note that in this case, the quote itself is a bit larger and italicized. You probably shouldn't make this too long so that they don't stand out too much. - Jo the Jovyan

# 
# For example, here's some popout content! It was created by adding the popout tag to a cell in the notebook. Jupyter Book automatically converts these cells into helpful side content.
# 
# To enable the cell tag editor, go click View -> Cell Toolbar -> Tags. This will enable the tags UI. Here's what the menu looks like.

# ### 搜狗输入法表情和符号
# 
# - ∫ ∑ ※ ➕➖✖️➗ ❎ √ ×
# - 😪😠😡😎☺️😁📚🌲
# - 👌👍👎👂👃👀✋❌💰🌂
# - 0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣②🔟 
# - 🐶🐱🐔🐷🐖🐴🐎🐂🐑🐯🐧🐺🐒🐵🐻🐦🐲
# - 💻 🌈🌎☁️❄️🏃♀👩👱✨
# - 🆚🔥🌹✈️🌉🎄
# 
# (✿◡‿◡)害羞 ⁄(⁄ ⁄•⁄ω⁄•⁄ ⁄)⁄ d=====(￣▽￣*)b厉害 
# 
# 我是我，不一样花火。～(￣▽￣～)(～￣▽￣)～ 矜持 

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

xi = [1, 2, 3, 4, 5]
y = [3, 5, 9, 13, 16]

plt.plot(xi, y, 'g-s')
plt.xlabel('$x_i$', fontsize = 20)
plt.ylabel('$y$', fontsize = 20)
plt.title('$Scatter\,Plot$', fontsize = 20)
plt.show()


# In[4]:


import matplotlib.pyplot as plt
plt.style.use('ggplot') 
xi = [1, 2, 3, 4, 5]
y = [3, 5, 9, 13, 16]
plt.plot(xi, y, 'g-s')
plt.xlabel('$x_i$', fontsize = 20)
plt.ylabel('$y$', fontsize = 20)
plt.title('$Scatter\,Plot$', fontsize = 20)
plt.show()


# ## 运行C代码
# 
# C functions are typically split into header files (.h) where things are declared but not defined, and implementation files (.c) where they are defined. http://people.duke.edu/~ccc14/sta-663/CrashCourseInC.html#a-tutorial-example-coding-a-fibonacci-function-in-c

# In[5]:


get_ipython().run_cell_magic('file', 'hello.c', '#include <stdio.h>\n\nint main() {\n    printf("Hello, world!");\n}')


# In[6]:


get_ipython().system(' gcc hello.c -o hello # 编译')


# In[3]:


get_ipython().system(' ./hello # 执行')


# ## Jupyter 魔术命令 

# In[14]:


get_ipython().run_line_magic('lsmagic', '')


# >  pip install version_information

# In[8]:


get_ipython().system('pip install version_information')


# In[9]:


# install version_information in the terminal first.
get_ipython().run_line_magic('reload_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'numpy, matplotlib, pandas, scipy, statsmodels')


# ## References
# 
# {% bibliography %}
# 
# 
# 
