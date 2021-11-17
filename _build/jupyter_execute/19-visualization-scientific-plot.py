#!/usr/bin/env python
# coding: utf-8

# # Matplotlib的科学绘图样式
# 
# Matplotlib styles for scientific plotting
# 
# https://github.com/garrettj403/SciencePlots
# 
# 

# In[1]:


get_ipython().system('pip3 install SciencePlots')


# SciencePlots库需要电脑安装LaTex，其中
# 
# - MacOS电脑安装MacTex https://www.tug.org/mactex/
# - Windows电脑安装MikTex https://miktex.org/

# In[4]:


import matplotlib.pyplot as plt
import numpy as np

plt.style.use('science')


# In[5]:


def function(x,p):
    return x**(2*p+1)/(1+x**(2*p))

pparam=dict(xlabel='Voltage(mV)',ylabel='Current($\mu$A)')

x=np.linspace(0.75,1.25,201)


# In[11]:


with plt.style.context(['science']):
    fig,ax=plt.subplots(figsize=(3, 3), dpi=150)
    for p in[10,15,20,30,50,100]:
        ax.plot(x,function(x,p),label=p)
        ax.legend(title='Order')
        ax.autoscale(tight=True)
        ax.set(**pparam)
    #fig.savefig('figures/fig1.pdf')
    #fig.savefig('figures/fig1.jpg',dpi=300)


# In[12]:


with plt.style.context(['science','ieee']):
    fig,ax=plt.subplots(figsize=(3, 3), dpi=150)
    for p in[10,15,20,30,50,100]:
        ax.plot(x,function(x,p),label=p)
        ax.legend(title='Order')
        ax.autoscale(tight=True)
        ax.set(**pparam)


# In[18]:


with plt.style.context(['science','nature']):
    fig,ax=plt.subplots(figsize=(3, 3), dpi=150)
    for p in[10,15,20,30,50,100]:
        ax.plot(x,function(x,p),label=p)
        ax.legend(title='Order')
        ax.autoscale(tight=True)
        ax.set(**pparam)


# In[19]:


with plt.style.context(['science','notebook']):
    fig,ax=plt.subplots(figsize=(3, 3), dpi=150)
    for p in[10,15,20,30,50,100]:
        ax.plot(x,function(x,p),label=p)
        ax.legend(title='Order')
        ax.autoscale(tight=True)
        ax.set(**pparam)


# In[20]:


with plt.style.context(['science','bright']):
    fig,ax=plt.subplots(figsize=(3, 3), dpi=150)
    for p in[10,15,20,30,50,100]:
        ax.plot(x,function(x,p),label=p)
        ax.legend(title='Order')
        ax.autoscale(tight=True)
        ax.set(**pparam)


# In[21]:


with plt.style.context(['science','high-vis']):
    fig,ax=plt.subplots(figsize=(3, 3), dpi=150)
    for p in[10,15,20,30,50,100]:
        ax.plot(x,function(x,p),label=p)
        ax.legend(title='Order')
        ax.autoscale(tight=True)
        ax.set(**pparam)


# In[17]:


with plt.style.context(['dark_background','science','high-vis']):
    fig,ax=plt.subplots(figsize = (3, 3), dpi = 200)
    for p in[10,15,20,30,50,100]:
        ax.plot(x,function(x,p),label=p)
    ax.legend(title='Order')
    ax.autoscale(tight=True)
    ax.set(**pparam)


# In[15]:


with plt.style.context(['science','scatter']):
    fig,ax=plt.subplots(figsize=(4,4), dpi=150)
    ax.plot([-2,2],[-2,2],'k--')
    ax.fill_between([-2,2],[-2.2,1.8],[-1.8,2.2],
    color='dodgerblue',alpha=0.2,lw=0)
    for i in range(7):
        x1=np.random.normal(0,0.5,10)
        y1=x1+np.random.normal(0,0.2,10)
        ax.plot(x1,y1,label=r"$^\#${}".format(i+1))
    ax.legend(title='Sample',loc=2)
    xlbl=r"$\log_{10}\left(\frac{L_\mathrm{IR}}{\mathrm{L}_\odot}\right)$"
    ylbl=r"$\log_{10}\left(\frac{L_\mathrm{6.2}}{\mathrm{L}_\odot}\right)$"
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])


# ## Common xlabel/ylabel for matplotlib subplots
# 
# https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
# 
# **New** in matplotlib 3.4.0
# 
# We can now use `supxlabel` and `supylabel` to set a common xlabel and ylabel.
# 
# Note that these are `FigureBase` methods, so they can be used with either `Figure` and `SubFigure`.
# 
# 

# In[2]:


pip install --upgrade matplotlib


# In[4]:


import pylab as plt
import numpy as np

x = np.arange(0.01, 10.01, 0.01)
y = 2 ** x

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.loglog(y, x)
ax2.loglog(x, y)

# subplot titles
ax1.set_title('A')
ax2.set_title('B')

# common labels
fig.supxlabel('fig.supxlabel')
fig.supylabel('fig.supylabel')

plt.tight_layout()


# In[3]:


import pylab as plt
import numpy as np

x = np.arange(0.01, 10.01, 0.01)
y = 2 ** x

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.loglog(y, x)
plt.title('A')
plt.subplot(2, 1, 2)
plt.loglog(x, y)
plt.title('B')
# common labels
fig.supxlabel('fig.supxlabel')
fig.supylabel('fig.supylabel')

plt.tight_layout()


# In[6]:


with plt.style.context(['science','nature']):
    fig = plt.figure(figsize=(4,4), dpi=150)
    plt.subplot(2, 1, 1)
    plt.loglog(y, x)
    plt.title('A')
    plt.subplot(2, 1, 2)
    plt.loglog(x, y)
    plt.title('B')
    # common labels
    fig.supxlabel('fig.supxlabel')
    fig.supylabel('fig.supylabel')

    plt.tight_layout()


# In[ ]:




