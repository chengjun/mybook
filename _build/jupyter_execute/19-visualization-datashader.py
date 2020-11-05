#!/usr/bin/env python
# coding: utf-8

# 
# # 使用Datashader可视化地理信息
# 
# 
# Datashader is part of the PyViz initiative for making Python-based visualization tools work well together. 
# 
# - http://datashader.org/index.html 
# - Datashader is supported and mantained by Anaconda 
# > !conda install datashader
# 

# 
# PyViz 
# 
# http://pyviz.org/
# 
# 
# 
# 

# ## US Census

# In[2]:


import datashader as ds
import datashader.transfer_functions as tf
import dask.dataframe as dd
import numpy as np


# In[3]:


df = dd.io.parquet.read_parquet('/Users/datalab/bigdata/census.snappy.parq')
df = df.persist()


# In[4]:


df.head()


# In[6]:


USA           = ((-124.72,  -66.95), (23.55, 50.06))
LakeMichigan  = (( -91.68,  -83.97), (40.75, 44.08))
Chicago       = (( -88.29,  -87.30), (41.57, 42.00))
Chinatown     = (( -87.67,  -87.63), (41.84, 41.86))
NewYorkCity   = (( -74.39,  -73.44), (40.51, 40.91))
LosAngeles    = ((-118.53, -117.81), (33.63, 33.96))
Houston       = (( -96.05,  -94.68), (29.45, 30.11))
Austin        = (( -97.91,  -97.52), (30.17, 30.37))
NewOrleans    = (( -90.37,  -89.89), (29.82, 30.05))
Atlanta       = (( -84.88,  -84.04), (33.45, 33.84))

from datashader.utils import lnglat_to_meters as webm
x_range,y_range = [list(r) for r in webm(*USA)]

plot_width  = int(900)
plot_height = int(plot_width*7.0/12)

background = "black"


# In[7]:


from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9
from IPython.core.display import HTML, display

export = partial(export_image, background = background, export_path="export")
cm = partial(colormap_select, reverse=(background!="black"))

display(HTML("<style>.container { width:100% !important; }</style>"))


# In[8]:


cvs = ds.Canvas(plot_width, plot_height, *webm(*USA))
agg = cvs.points(df, 'easting', 'northing')


# In[11]:


export(tf.shade(agg, cmap = cm(Greys9, 0.2), how='log'),"census_gray_linear")


# In[12]:


from colorcet import fire
export(tf.shade(agg, cmap = cm(fire,0.2), how='eq_hist'),"census_ds_fire_eq_hist")


# In[13]:


from datashader.colors import viridis
export(tf.shade(agg, cmap=cm(viridis), how='eq_hist'),"census_viridis_eq_hist")


# In[15]:


if background == "black":
      color_key = {'w':'aqua', 'b':'lime',  'a':'red', 'h':'fuchsia', 'o':'yellow' }
else: color_key = {'w':'blue', 'b':'green', 'a':'red', 'h':'orange',  'o':'saddlebrown'}
def create_image(longitude_range, latitude_range, w=plot_width, h=plot_height):
    x_range,y_range=webm(longitude_range,latitude_range)
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'easting', 'northing', ds.count_cat('race'))
    img = tf.shade(agg, color_key=color_key, how='eq_hist')
    return img

export(create_image(*USA),"Zoom 0 - USA")


# In[16]:


export(create_image(*NewYorkCity),"NYC")


# In[17]:


cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
aggc = cvs.points(df, 'easting', 'northing', ds.count_cat('race'))

export(tf.shade(aggc.sel(race='b'), cmap=cm(Greys9,0.25), how='eq_hist'),"USA blacks")


# In[18]:


agg2 = aggc.where((aggc.sel(race=['w', 'b', 'a', 'h']) > 0).all(dim='race')).fillna(0)
export(tf.shade(agg2, color_key=color_key, how='eq_hist'),"USA all")


# ## NYC Crime

# In[170]:


# https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i
import pandas as pd
df = pd.read_csv('/Users/datalab/bigdata/NYPD_Complaint_Data_Historic.csv', #nrows = 1000, 
                usecols= ['Latitude', 'Longitude', 'SUSP_SEX', 'SUSP_RACE', 'OFNS_DESC'])
df.head()


# In[171]:


from datashader.utils import lnglat_to_meters as webm

df['Lon'], df['Lat'] = webm(df['Longitude'].tolist(), df['Latitude'].tolist())


# In[119]:


df.head()


# In[18]:


df.columns


# In[104]:


df.head()


# In[38]:


df.groupby('SUSP_SEX').size()


# In[39]:


df.groupby('SUSP_RACE').size()


# In[172]:


import datashader as ds
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, inferno
import datashader.transfer_functions as tf


# In[173]:


# http://datashader.org/topics/census.html
# Initial datashader / visualization configuration
background = 'black'
export = partial(export_image, background = background, export_path="export")
cm = partial(colormap_select, reverse=(background!="black"))
# Create a color key for VIOLATION, MISDEMEANOR, and FELONY
# color_key = {'F':'white', 'M':'yellow',  'U':'red'}
# Convert OFFENSE_LEVEL column to type 'category'
# df['SUSP_SEX'] = df['SUSP_SEX'].astype('category')
# Create function to re-generate canvas, grid, and map based on data category provided
from datashader.utils import lnglat_to_meters as webm

NewYorkCity   = (( -74.39,  -73.44), (40.51, 40.91))
x_range,y_range = [list(r) for r in webm(*NewYorkCity)]

plot_width  = int(900)
plot_height = int(plot_width*7.0/12)

cvs = ds.Canvas(plot_width, plot_height, *webm(*NewYorkCity))
agg = cvs.points(df, 'Lon', 'Lat')#, ds.count_cat('SUSP_SEX'))


# In[176]:


export(tf.shade(agg, cmap = cm(Greys9,0.25), how='log'),"census_gray_linear")*2


# In[178]:


from datashader.colors import viridis

# Show map with 'viridis' color map
export(tf.shade(agg, cmap = cm(viridis, 0.1), how = 'eq_hist'), "export")*2


# In[179]:


from colorcet import fire
export(tf.shade(agg, cmap = cm(fire,0.2), how='eq_hist'),"census_ds_fire_eq_hist")*3


# ![](images/end.png)
