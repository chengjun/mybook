#!/usr/bin/env python
# coding: utf-8

# # 解决Matplotlib绘图显示中文问题
# 

# In[18]:


import pylab as plt
plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

x = range(100)
y = [i**2 for i in x] 
plt.plot(x, y, 'r--', label = '人啊，认识你自己！')
plt.xlabel('我一无所知！', fontsize = 16)
plt.ylabel('未经省察的人生不值得度过。', fontsize = 16)
plt.title('苏格拉底', fontsize = 20)
plt.legend(fontsize = 16)
plt.show()


# ## 1. 下载微软雅黑字体
# 
# https://github.com/computational-class/ccbook/blob/master/data/msyh.ttf

# ## 2. 找到字体文件夹

# In[4]:


import matplotlib
print(matplotlib.matplotlib_fname())


# ## 3. 将字体文件放到ttf目录
# 
# - 打开matplotlibrc所在这个文件夹
# - 进入fonts\ttf目录
# - 把第一步下载的msyh.ttf放到该目录下面

# ## 4. 修改matplotlibrc文件
# 使用任何一个文件编辑器(推荐sublime Text2),修改该文件,通过ctrl+f搜索找到
# 
# ```
# #axes.unicode_minus  : True    ## use unicode for the minus symbol
# #font.family         : sans-serif
# #font.sans-serif     : DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
# ```

# 分别修改为以下三行
# 
# ```
# axes.unicode_minus  : False    ## use unicode for the minus symbol
# font.family         : Microsoft YaHei
# font.sans-serif     : Microsoft YaHei, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
# ```    
#     
#     

# - 首先, 三行都需要删除第一个#,取消注释
# - 第一行,修改True为False,是为了正常显示负号
# - 第二行和第三行是为了使用微软雅黑作为默认字体

# ## 5. 删除缓存

# In[5]:


import matplotlib
print(matplotlib.get_cachedir())


# - 一般在用户`.matplotlib`文件夹📂
# - 删除该目录下的所有文件

# ## 6. 重启Jupyter Notebook
# 
# - 刷新页面即可
# - 或者点击 `服务`-`重启`

# In[6]:


# test
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(2, 2)
plt.text(2, 2, '汉字', fontsize = 300)
plt.show()

