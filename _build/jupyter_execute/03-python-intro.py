#!/usr/bin/env python
# coding: utf-8

# 
# # 第二章 数据科学的编程工具
# Python使用简介
# 
# 
# 王成军
# 
# ![](./images/author.png)

# ## 人生苦短，我用Python。
# 
# Python（/ˈpaɪθən/）是一种面向对象、解释型计算机程序设计语言
# - 由Guido van Rossum于1989年底发明
# - 第一个公开发行版发行于1991年
# - Python语法简洁而清晰
# - 具有强大的标准库和丰富的第三方模块
# - 它常被昵称为胶水语言
# - TIOBE编程语言排行榜“2010年度编程语言”
# 
# 

# ## 特点
# - 免费、功能强大、使用者众多
# - 与R和MATLAB相比，Python是一门更易学、更严谨的程序设计语言。使用Python编写的脚本更易于理解和维护。
# - 如同其它编程语言一样，Python语言的基础知识包括：类型、列表（list）和元组（tuple）、字典（dictionary）、条件、循环、异常处理等。
# - 关于这些，初阶读者可以阅读《Beginning Python》一书（Hetland, 2005)。
# 

# ## Python中包含了丰富的类库。
# 众多开源的科学计算软件包都提供了Python的调用接口，例如著名的计算机视觉库OpenCV。
# Python本身的科学计算类库发展也十分完善，例如NumPy、SciPy和matplotlib等。
# 就社会网络分析而言，igraph, networkx, graph-tool, Snap.py等类库提供了丰富的网络分析工具

# ## Python软件与IDE
# 目前最新的Python版本为3.0，更稳定的2.7版本。
# 编译器是编写程序的重要工具。
# 免费的Python编译器有Spyder、PyCharm(免费社区版)、Ipython、Vim、 Emacs、 Eclipse(加上PyDev插件)。
# 

# ## Installing Anaconda Python
# - Use the Anaconda Python
#     - http://anaconda.com/

# ## 第三方包可以使用pip install的方法安装。
# - 可以点击ToolsOpen command prompt
# - 然后在打开的命令窗口中输入：
#     - <del>pip install beautifulsoup4 
# 

# > pip install beautifulsoup4

# - NumPy /SciPy for scientific computing
# - pandas to make Python usable for data analysis
# - matplotlib to make graphics
# - scikit-learn for machine learning
# 

# In[1]:


pip install flownetwork


# In[2]:


from flownetwork import flownetwork as fn
import networkx as nx
import pylab as plt
import numpy as np

print(fn.__version__)


# In[3]:


help(fn.constructFlowNetwork)


# In[3]:


# constructing a flow network
demo = fn.attention_data
gd = fn.constructFlowNetwork(demo)


# In[5]:


# drawing a demo network
fig = plt.figure(figsize=(12, 8),facecolor='white')
pos={0: np.array([ 0.2 ,  0.8]),
 2: np.array([ 0.2,  0.2]),
 1: np.array([ 0.4,  0.6]),
 6: np.array([ 0.4,  0.4]),
 4: np.array([ 0.7,  0.8]),
 5: np.array([ 0.7,  0.5]),
 3: np.array([ 0.7,  0.2 ]),
 'sink': np.array([ 1,  0.5]),
 'source': np.array([ 0,  0.5])}

width=[float(d['weight']*1.2) for (u,v,d) in gd.edges(data=True)]
edge_labels=dict([((u,v,),d['weight']) for u,v,d in gd.edges(data=True)])

nx.draw_networkx_edge_labels(gd,pos,edge_labels=edge_labels, font_size = 15, alpha = .5)
nx.draw(gd, pos, node_size = 3000, node_color = 'orange',
        alpha = 0.2, width = width, edge_color='orange',style='solid')
nx.draw_networkx_labels(gd,pos,font_size=18)
plt.show()


# In[4]:


nx.info(gd)


# In[7]:


# flow matrix
m = fn.getFlowMatrix(gd)
m


# In[8]:


fn.networkDissipate(gd)


# In[5]:


import random, datetime
import numpy as np
import pylab as plt
import matplotlib
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats.stats import pearsonr


# In[7]:


with open('./data/the_republic_plato_gutenberg_pg1497.txt', 'r') as f:
    lines = f.readlines()


# In[8]:


len(lines)


# In[9]:


type(lines)


# In[10]:


lines[8520:8530]


# ## Variable Type
# 

# In[15]:


# str, int, float, bool
str(3)


# In[16]:


"chengjun wang"


# In[17]:


# int
int('5') 


# In[18]:


# float
float(str(7.1))


# In[32]:


range(10) 


# In[20]:


# for i in range(1, 10):
#     print(i)
range(1, 10)


# ## dir & help
# 
# 当你想要了解对象的详细信息时使用

# In[21]:


dir(str) [-10:]


# In[22]:


'cheng'.capitalize()


# In[23]:


#dir(str)[-10:]


# In[63]:


help(str)


# In[45]:


'   '.isspace()


# In[35]:


'socrates the king'.__add__(' is the greatest.')


# In[24]:


x = ' Hello WorlD  '
dir(x)[-10:] 


# In[25]:


# lower
x.lower() 


# In[26]:


# upper
x.upper()


# In[28]:


# rstrip
x.rstrip()


# In[29]:


# strip
x.strip()


# In[33]:


# replace
x.replace('lo', 'l')


# In[38]:


# split
# x.lower().strip().split(' ')
x.split('lo')


# In[39]:


# join 
', '.join(['a', 'b'])


# ## type
# 当你想要了解变量类型时使用type

# In[40]:


x = 'hello world'
type(x)


# ## Data Structure
# list, tuple, set, dictionary, array
# 

# In[41]:


l = [1,2,3,3] # list
t = (1, 2, 3, 3) # tuple
s = {1, 2, 3, 3} # set([1,2,3,3]) # set
d = {'a':1,'b':2,'c':3} # dict
a = np.array(l) # array
print(l, t, s, d, a)


# In[42]:


l = [1,2,3,3] # list
l.append(4)
l


# In[43]:


d = {'a':1,'b':2,'c':3} # dict
d.keys()


# In[44]:


d = {'a':1,'b':2,'c':3} # dict
d.values()


# In[48]:


d = {'a':1,'b':3,'c':3} # dict
d['b']


# In[64]:


d = {'a':1,'b':2,'c':3} # dict
d.items() 


# ## 定义函数

# In[49]:


def devidePlus(m, n): # 结尾是冒号
    y = m/n + 1 # 注意：空格
    return y          # 注意：return


# ## For 循环

# In[50]:


range(10)


# In[51]:


range(1, 10)  


# In[52]:


for i in range(10):
    print(i, i*10, i**2)


# In[53]:


for i in range(10):
    print(i*10) 


# In[54]:


for i in range(10):
    print(devidePlus(i, 2))


# In[55]:


# 列表内部的for循环
r = [devidePlus(i, 2) for i in range(10)]
r 


# ## map函数

# In[58]:


def fahrenheit(T):
    return (9/5)*T + 32

temp = [0, 22.5, 40,100]
F_temps = map(fahrenheit, temp)

print(*F_temps)


# In[59]:


m1 = map(devidePlus, [4,3,2], [2, 1, 5])
print(*m1)
#print(*map(devidePlus, [4,3,2], [2, 1, 5]))
# 注意： 将（4， 2)作为一个组合进行计算，将（3， 1）作为一个组合进行计算


# In[60]:


m2 = map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
print(*m2)


# In[78]:


m3 = map(lambda x, y, z: x + y - z, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [3, 3, 2, 2, 5])
print(*m3)


# ## if elif else

# In[61]:


j = 5.5
if j%2 == 1:
    print(r'余数是1')
elif j%2 ==0:
    print(r'余数是0')
else:
    print(r'余数既不是1也不是0')


# In[62]:


x = 5
if x < 5:
    y = -1
    z = 5
elif x > 5:
    y = 1
    z = 11
else:
    y = 0
    z = 10
print(x, y, z)


# ## while循环

# In[63]:


j = 0
while j <10:
    print(j)
    j+=1 # avoid dead loop
    


# In[64]:


j = 0
while j <10:
    if j%2 != 0: 
        print(j**2)
    j+=1 # avoid dead loop 


# In[65]:


j = 0
while j <50:
    if j == 30:
        break
    if j%2 != 0: 
        print(j**2)
    j+=1 # avoid dead loop
    


# In[66]:


a = 4
while a: # 0, None, False
    print(a) 
    a -= 1
    if a < 2:
        a = None # []


# ## try except 

# In[70]:


def devidePlus(m, n): # 结尾是冒号
    return m/n+ 1 # 注意：空格
for i in [2, 0, 5]:
    try:
        print(devidePlus(4, i))
    except Exception as e:
        print(i, e)
        pass


# In[73]:


alist = [[1,1], [0, 0, 1]]
for i in alist:
    try:
        for j in i:
            print(10 / j)
    except Exception as e:
        print(e)
        pass


# In[74]:


alist = [[1,1], [0, 0, 1]]
for i in alist:
    for j in i:
        try:
            print(10 / j)
        except Exception as e:
            print(j, e)
            pass


# ## Write and Read data

# In[75]:


data =[[i, i**2, i**3] for i in range(10)] 
data


# In[76]:


for i in data:
    print('\t'.join(map(str, i)))  


# In[77]:


type(data)


# In[78]:


len(data)


# In[79]:


data[0]


# In[87]:


help(f.write)  


# In[88]:


# 保存数据
data =[[i, i**2, i**3] for i in range(10000)] 

f = open("data_write_to_file1.txt", "w")
for i in data:
    f.write('\t'.join(map(str,i)) + '\n')
f.close() 


# In[89]:


with open('data_write_to_file1.txt','r') as f:
    data = f.readlines()
data[:5]


# In[82]:


with open('./data/data_write_to_file1.txt','r') as f:
    data = f.readlines(1000) #bytes 
len(data) 


# In[84]:


with open('./data/data_write_to_file1.txt','r') as f:
    print(f.readline())


# In[85]:


f = [1, 2, 3, 4, 5]
for k, i in enumerate(f):
    print(k, i)

# with open('data_write_to_file1.txt', 'r') as f:
#      for i in f:
#             print(i)


# In[88]:


from time import sleep

total = 0
with open('./data/data_write_to_file1.txt','r') as f:
    for k, i in enumerate(f):
        if k % 1000 ==0:
            sleep(1)
            print(k)
        total += sum([int(j) for j in i.strip().split('\t')])

print(total)


# In[32]:


with open('../data/data_write_to_file.txt','r') as f:
    for k, i in enumerate(f):
        if k%2000 == 0:
            print(i)


# In[89]:


data = []
line = '0\t0\t0\n'
line = line.replace('\n', '')
line = line.split('\t')
line = [int(i) for i in line] # convert str to int
data.append(line) 
data


# In[96]:


# 读取数据
data = []
with open('./data/data_write_to_file1.txt','r') as f:
    for line in f:
        line = line.replace('\n', '').split('\t')
        line = [int(i) for i in line]
        data.append(line)
len(data)
#data[-5:]


# In[98]:


# 读取数据
data = []
with open('./data/data_write_to_file.txt','r') as f:
    for line in f:
        line = line.replace('\n', '').split('\t')
        line = [int(i) for i in line]
        data.append(line)
len(data)


# In[99]:


import pandas as pd


# In[4]:


help(pd.read_csv)


# In[108]:


df = pd.read_csv('./data/data_write_to_file1.txt', 
                 sep = '\t', names = ['a', 'b', 'c'])
df.head()


# ## 保存中间步骤产生的字典数据

# In[109]:


import json
data_dict = {'a':1, 'b':2, 'c':3}
with open('./data/save_dict.json', 'w') as f:
    json.dump(data_dict, f)


# In[109]:


import json
dd = json.load(open("./data/save_dict.json"))
dd


# ## 重新读入json

# ## 保存中间步骤产生的列表数据

# In[71]:


data_list = list(range(10))
with open('./data/save_list.json', 'w') as f:
    json.dump(data_list, f)


# In[110]:


dl = json.load(open("./data/save_list.json"))
dl


# ## 使用matplotlib绘图

# In[41]:


#
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
x = range(1, 100)
y = [i**-3 for i in x]
plt.plot(x, y, 'b-s')
plt.ylabel('$p(k)$', fontsize = 20)
plt.xlabel('$k$', fontsize = 20)
plt.xscale('log')
plt.yscale('log')
plt.title('Degree Distribution')
plt.show()


# In[42]:


import numpy as np
# red dashes, blue squares and green triangles
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--')
plt.plot(t, t**2, 'bs')
plt.plot(t, t**3, 'g^')
plt.show()


# In[43]:


# red dashes, blue squares and green triangles
t = np.arange(0., 5., 0.2)
plt.plot(t, t**2, 'b-s', label = '1')
plt.plot(t, t**2.5, 'r-o', label = '2')
plt.plot(t, t**3, 'g-^', label = '3')
plt.annotate(r'$\alpha = 3$', xy=(3.5, 40), xytext=(2, 80),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize = 20)
plt.ylabel('$f(t)$', fontsize = 20)
plt.xlabel('$t$', fontsize = 20)
plt.legend(loc=2,numpoints=1,fontsize=10)
plt.show()
# plt.savefig('/Users/chengjun/GitHub/cjc/figure/save_figure.png',
#             dpi = 300, bbox_inches="tight",transparent = True)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(5,5))
sns.set(style="whitegrid")


# In[44]:


plt.figure(1)
plt.subplot(221)
plt.plot(t, t, 'r--')
plt.text(2, 0.8*np.max(t), r'$\alpha = 1$', fontsize = 20)
plt.subplot(222)
plt.plot(t, t**2, 'bs')
plt.text(2, 0.8*np.max(t**2), r'$\alpha = 2$', fontsize = 20)
plt.subplot(223)
plt.plot(t, t**3, 'g^')
plt.text(2, 0.8*np.max(t**3), r'$\alpha = 3$', fontsize = 20)
plt.subplot(224)
plt.plot(t, t**4, 'r-o')
plt.text(2, 0.8*np.max(t**4), r'$\alpha = 4$', fontsize = 20)
plt.show()


# In[4]:


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)


# In[5]:


plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo')
plt.plot(t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()


# In[4]:


import matplotlib.gridspec as gridspec
import numpy as np

t = np.arange(0., 5., 0.2)

gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
plt.plot(t, t**2, 'b-s')
ax2 = plt.subplot(gs[1,:-1])
plt.plot(t, t**2, 'g-s')
ax3 = plt.subplot(gs[1:, -1])
plt.plot(t, t**2, 'r-o')
ax4 = plt.subplot(gs[-1,0])
plt.plot(t, t**2, 'g-^')
ax5 = plt.subplot(gs[-1,1])
plt.plot(t, t**2, 'b-<')
plt.tight_layout()


# In[45]:



def OLSRegressPlot(x,y,col,xlab,ylab):
    xx = sm.add_constant(x, prepend=True)
    res = sm.OLS(y,xx).fit()
    constant, beta = res.params
    r2 = res.rsquared
    lab = r'$\beta = %.2f, \,R^2 = %.2f$' %(beta,r2)
    plt.scatter(x,y,s=60,facecolors='none', edgecolors=col)
    plt.plot(x,constant + x*beta,"red",label=lab)
    plt.legend(loc = 'upper left',fontsize=16)
    plt.xlabel(xlab,fontsize=26)
    plt.ylabel(ylab,fontsize=26)


# In[46]:


x = np.random.randn(50)
y = np.random.randn(50) + 3*x
pearsonr(x, y)
fig = plt.figure(figsize=(10, 4),facecolor='white')
OLSRegressPlot(x,y,'RoyalBlue',r'$x$',r'$y$')
plt.show()


# In[206]:


fig = plt.figure(figsize=(7, 4),facecolor='white')
data = norm.rvs(10.0, 2.5, size=5000)
mu, std = norm.fit(data)
plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2)
title = r"$\mu = %.2f, \,  \sigma = %.2f$" % (mu, std)
plt.title(title,size=16)
plt.show()


# In[47]:


import pandas as pd
df = pd.read_csv('../data/data_write_to_file.txt', sep = '\t', names = ['a', 'b', 'c'])
df[:5]


# In[7]:


df.plot.line()
plt.yscale('log')
plt.ylabel('$values$', fontsize = 20)
plt.xlabel('$index$', fontsize = 20)
plt.show()


# In[27]:


df.plot.scatter(x='a', y='b')
plt.show()


# In[18]:


df.plot.hexbin(x='a', y='b', gridsize=25)
plt.show()


# In[22]:


df['a'].plot.kde()
plt.show()


# In[32]:


bp = df.boxplot()
plt.yscale('log')
plt.show()


# In[41]:


df['c'].diff().hist()
plt.show()


# In[45]:


df.plot.hist(stacked=True, bins=20)
# plt.yscale('log')
plt.show()


# > To be a programmer is to develop a carefully managed relationship with error. There's no getting around it. You either make your accommodations with failure, or the work will become intolerable.
# 
# Ellen Ullman  （an American computer programmer and author）
# 

# ## This is the end.
# > Thank you for your attention.
