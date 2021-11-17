#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# # 抓取历届政府工作报告
# 
# 

# In[1]:


import requests
from bs4 import BeautifulSoup


# www.hprc.org.cn/wxzl/wxysl/lczf/
# 
# 另外一个镜像
# 
# http://hprc.cssn.cn/wxzl/wxysl/lczf/

# ## Inspect
# 
# <td width="274" class="bl">·&nbsp;<a href="./d12qgrdzfbg/201603/t20160318_369509.html" target="_blank" title="2016年政府工作报告">2016年政府工作报告</a></td>
# 
#     <td width="274" class="bl">·&nbsp;<a href="./d12qgrdzfbg/201603/t20160318_369509.html" target="_blank" title="2016年政府工作报告">2016年政府工作报告</a></td>
# 
# 

# In[3]:


# get the link for each year
url = "http://www.hprc.org.cn/wxzl/wxysl/lczf/" 
content = requests.get(url)
content.encoding


# ### Encoding
# 
# - ASCII
#     - 7位字符集
#     - 美国标准信息交换代码（American Standard Code for Information Interchange）的缩写, 为美国英语通信所设计。
#     - 它由128个字符组成，包括大小写字母、数字0-9、标点符号、非打印字符（换行符、制表符等4个）以及控制字符（退格、响铃等）组成。
# - iso8859-1 通常叫做Latin-1。
#     - 和ascii编码相似。
#     - 属于单字节编码，最多能表示的字符范围是0-255，应用于英文系列。比如，字母a的编码为0x61=97。 
#     - 无法表示中文字符。
#     - 单字节编码，和计算机最基础的表示单位一致，所以很多时候，仍旧使用iso8859-1编码来表示。在很多协议上，默认使用该编码。

# - gb2312/gbk/gb18030
#     - 是汉字的国标码，专门用来表示汉字，是双字节编码，而英文字母和iso8859-1一致（兼容iso8859-1编码）。
#     - 其中gbk编码能够用来同时表示繁体字和简体字,K 为汉语拼音 Kuo Zhan（扩展）中“扩”字的声母
#     - gb2312只能表示简体字，gbk是兼容gb2312编码的。 
#     - gb18030，全称：国家标准 GB 18030-2005《信息技术中文编码字符集》，是中华人民共和国现时最新的内码字集

# - unicode 
#     - 最统一的编码，用来表示所有语言的字符。
#     - 占用更多的空间，定长双字节（也有四字节的）编码，包括英文字母在内。
#     - 不兼容iso8859-1等其它编码。相对于iso8859-1编码来说，uniocode编码只是在前面增加了一个0字节，比如字母a为"00 61"。 
#     - 定长编码便于计算机处理（注意GB2312/GBK不是定长编码），unicode又可以用来表示所有字符，所以在很多软件内部是使用unicode编码来处理的，比如java。 
# - UTF 
#     - unicode不便于传输和存储，产生了utf编码
#     - utf编码兼容iso8859-1编码，同时也可以用来表示所有语言的字符
#     - utf编码是不定长编码，每一个字符的长度从1-6个字节不等。
#     - 其中，utf8（8-bit Unicode Transformation Format）是一种针对Unicode的可变长度字符编码，又称万国码。
#         - 由Ken Thompson于1992年创建。现在已经标准化为RFC 3629。

# ### decode
# <del>urllib2.urlopen(url).read().decode('gb18030') </del>
# 
#     content.encoding = 'gb18030'
#     
#     content = content.text
#   
# Or
# 
#     content = content.text.encode(content.encoding).decode('gb18030')
# 
# 
# 
# ### html.parser
# BeautifulSoup(content, 'html.parser')

# In[4]:


# Specify the encoding
content.encoding = 'utf8' # 'gb18030'
content = content.text


# In[5]:


soup = BeautifulSoup(content, 'html.parser') 
# links = soup.find_all('td', {'class', 'bl'})   
links = soup.select('.bl a')
print(links[0])


# In[8]:


len(links)


# In[9]:


links[-1]['href']


# In[10]:


links[0]['href'].split('./')[1]


# In[12]:


print(url + links[0]['href'].split('./')[1])


# In[13]:


hyperlinks = [url + i['href'].split('./')[1] for i in links]
hyperlinks[:5]


# In[14]:


hyperlinks[-5:]


# In[15]:


print(hyperlinks[14]) # 2007年有分页


# ### Inspect 下一页
# 
# <a href="t20090818_27775_1.html"><span style="color:#0033FF;font-weight:bold">下一页</span></a>
# 
#     <a href="t20090818_27775_1.html"><span style="color:#0033FF;font-weight:bold">下一页</span></a>
#     
# - a
#     - script
#         - td

# In[16]:


url_i = 'http://www.hprc.org.cn/wxzl/wxysl/lczf/dishiyijie_1/200908/t20090818_3955570.html'
content = requests.get(url_i)
content.encoding = 'utf8'
content = content.text
#content = content.text.encode(content.encoding).decode('gb18030')
soup = BeautifulSoup(content, 'html.parser') 
#scripts = soup.find_all('script')
#scripts[0]
scripts = soup.select('td script')[0]


# In[17]:


scripts


# In[18]:


scripts.text


# In[19]:


# countPage = int(''.join(scripts).split('countPage = ')\
#                 [1].split('//')[0])
# countPage

countPage = int(scripts.text.split('countPage = ')[1].split('//')[0])
countPage


# In[30]:


import sys
def flushPrint(s):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % s)
    sys.stdout.flush()
    
def crawler(url_i):
    content = requests.get(url_i)
    content.encoding = 'utf8'  
    content = content.text
    soup = BeautifulSoup(content, 'html.parser') 
    year = soup.find('span', {'class', 'huang16c'}).text[:4]
    year = int(year)
    report = ''.join(s.text for s in soup('p'))
    # 找到分页信息
    scripts = soup.find_all('script')
    countPage = int(''.join(scripts[1]).split('countPage = ')[1].split('//')[0])
    if countPage == 1:
        pass
    else:
        for i in range(1, countPage):
            url_child = url_i.split('.html')[0] +'_'+str(i)+'.html'
            content = requests.get(url_child)
            content.encoding = 'utf8'
            content = content.text
            soup = BeautifulSoup(content, 'html.parser') 
            report_child = ''.join(s.text for s in soup('p'))
            report = report + report_child
    return year, report


# In[26]:


hyperlinks[14]


# In[31]:


year, report = crawler(hyperlinks[14])


# In[32]:


year


# In[34]:


report[:30]


# In[35]:


# 抓取52年政府工作报告内容
reports = {}
for link in hyperlinks:
    year, report = crawler(link)
    flushPrint(year)
    reports[year] = report 


# In[36]:


with open('./data/gov_reports1954-2021.txt', 'w', encoding = 'utf8') as f:
    for r in reports:
        line = str(r)+'\t'+reports[r].replace('\n', '\t') +'\n'
        f.write(line)


# In[37]:


import pandas as pd

df = pd.read_table('./data/gov_reports1954-2021.txt', names = ['year', 'report'])


# In[38]:


df[:5]


# ![](./images/end.png)
