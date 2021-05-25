#!/usr/bin/env python
# coding: utf-8

# 
# # 对大数据进行预处理
# 
# 以占领华尔街推特数据为例
# 
# 
# 
# ![image.png](./images/author.png)

# ## 字节（Byte /bait/）
# 
# 计算机信息技术用于计量存储容量的一种计量单位，通常情况下一字节等于有八位， [1]  也表示一些计算机编程语言中的数据类型和语言字符。
# - 1B（byte，字节）= 8 bit；
# - 1KB=1000B；1MB=1000KB=1000×1000B。其中1000=10^3。
# - 1KB（kilobyte，千字节）=1000B= 10^3 B；
# - 1MB（Megabyte，兆字节，百万字节，简称“兆”）=1000KB= 10^6 B；
# - 1GB（Gigabyte，吉字节，十亿字节，又称“千兆”）=1000MB= 10^9 B；

# ## 按照Chunk读取数据并进行处理
# 
# Lazy Method for Reading Big File in Python?

# In[1]:


# 按行读取数据
line_num = 0
cops_num = 0
with open('/Users/datalab/bigdata/cjc/ows-raw.txt', 'r') as f:
    for i in f:
        line_num += 1
        if 'cops' in i:
            cops_num += 1
        if line_num % 100000 ==0:
            print(line_num)


# In[2]:


line_num


# In[4]:


cops_num/line_num


# In[1]:


bigfile = open('/Users/datalab/bigdata/cjc/ows-raw.txt', 'r')
chunkSize = 1000000
chunk = bigfile.readlines(chunkSize)
print(len(chunk))
# with open("../data/ows_tweets_sample.txt", 'w') as f:
#     for i in chunk:
#         f.write(i)  


# In[5]:


get_ipython().run_line_magic('pinfo', 'bigfile.readlines')


# In[8]:


5%5


# In[5]:


# https://stackoverflow.com/questions/519633/lazy-method-for-reading-big-file-in-python?lq=1
import csv
bigfile = open('/Users/datalab/bigdata/cjc/ows-raw.txt', 'r')
chunkSize = 10**8
chunk = bigfile.readlines(chunkSize)
num_chunk, num_lines = 0, 0
while chunk:
    lines = csv.reader((line.replace('\x00','') for line in chunk), 
                       delimiter=',', quotechar='"')
    #do sth.
    num_lines += len(list(lines))
    if num_chunk % 5 ==0:
        print(num_chunk, num_lines)
    num_chunk += 1
    chunk = bigfile.readlines(chunkSize) # read another chunk


# In[4]:


num_lines


# ## 用Pandas的get_chunk功能来处理亿级数据
# 
# > 只有在超过5TB数据量的规模下，Hadoop才是一个合理的技术选择。

# In[14]:


import pandas as pd

f = open('/Users/datalab/bigdata/cjc/ows-raw.txt',encoding='utf-8')
reader = pd.read_table(f,  sep=',',  quotechar='"', iterator=True, error_bad_lines=False) #跳过报错行
chunkSize = 100000
chunk = reader.get_chunk(chunkSize)
len(chunk)

#pd.read_table?


# In[15]:


chunk.head()


# In[2]:


import pandas as pd

f = open('/Users/datalab/bigdata/cjc/ows-raw.txt',encoding='utf-8')
reader = pd.read_table(f,  sep=',',  quotechar='"', 
                       iterator=True, error_bad_lines=False) #跳过报错行
chunkSize = 100000
loop = True
#data = []
num_chunk, num_lines = 0, 0
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        # dat = data_cleaning_funtion(chunk) # do sth.
        num_lines += len(chunk)
        print(num_chunk, num_lines)
        num_chunk +=1
        #data.append(dat) 
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
#df = pd.concat(data, ignore_index=True)


# ![image.png](./images/end.png)
