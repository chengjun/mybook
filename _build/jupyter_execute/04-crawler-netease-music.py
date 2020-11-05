#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# # 抓取网易云音乐热门评论
# 
# 

# https://github.com/RitterHou/music-163
# 
# 爬取网易云音乐的所有的歌曲的评论数。以下为主要思路：
# 
# - 爬取所有的歌手信息（artists.py）；
# - 根据上一步爬取到的歌手信息去爬取所有的专辑信息（album_by _artist.py）；
# - 根据专辑信息爬取所有的歌曲信息（music_by _album.py）；
# - 根据歌曲信息爬取其评论条数（comments_by _music.py）

# ## 爬取所有的歌手信息（artists.py）

# 观察网易云音乐官网页面HTML结构
# 
# http://music.163.com/

# 
# 
# http://music.163.com/#/discover/artist/cat

# http://music.163.com/#/discover/artist/cat?id=4003&initial=0
# 
# 
# ![](./images/netease.png)
# 

# In[1]:


import requests
from bs4 import BeautifulSoup


# In[2]:


headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Cookie': '_ntes_nnid=7eced19b27ffae35dad3f8f2bf5885cd,1476521011210; _ntes_nuid=7eced19b27ffae35dad3f8f2bf5885cd; usertrack=c+5+hlgB7TgnsAmACnXtAg==; Province=025; City=025; NTES_PASSPORT=6n9ihXhbWKPi8yAqG.i2kETSCRa.ug06Txh8EMrrRsliVQXFV_orx5HffqhQjuGHkNQrLOIRLLotGohL9s10wcYSPiQfI2wiPacKlJ3nYAXgM; P_INFO=hourui93@163.com|1476523293|1|study|11&12|jis&1476511733&mail163#jis&320100#10#0#0|151889&0|g37_client_check&mailsettings&mail163&study&blog|hourui93@163.com; NTES_SESS=Fa2uk.YZsGoj59AgD6tRjTXGaJ8_1_4YvGfXUkS7C1NwtMe.tG1Vzr255TXM6yj2mKqTZzqFtoEKQrgewi9ZK60ylIqq5puaG6QIaNQ7EK5MTcRgHLOhqttDHfaI_vsBzB4bibfamzx1.fhlpqZh_FcnXUYQFw5F5KIBUmGJg7xdasvGf_EgfICWV; S_INFO=1476597594|1|0&80##|hourui93; NETEASE_AUTH_SOURCE=space; NETEASE_AUTH_USERNAME=hourui93; _ga=GA1.2.1405085820.1476521280; JSESSIONID-WYYY=cbd082d2ce2cffbcd5c085d8bf565a95aee3173ddbbb00bfa270950f93f1d8bb4cb55a56a4049fa8c828373f630c78f4a43d6c3d252c4c44f44b098a9434a7d8fc110670a6e1e9af992c78092936b1e19351435ecff76a181993780035547fa5241a5afb96e8c665182d0d5b911663281967d675ff2658015887a94b3ee1575fa1956a5a%3A1476607977016; _iuqxldmzr_=25; __utma=94650624.1038096298.1476521011.1476595468.1476606177.8; __utmb=94650624.20.10.1476606177; __utmc=94650624; __utmz=94650624.1476521011.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)',
    'DNT': '1',
    'Host': 'music.163.com',
    'Pragma': 'no-cache',
    'Referer': 'http://music.163.com/',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'
}


# In[7]:


group_id = 1001
initial = 67
params = {'id': group_id, 'initial': initial}
r = requests.get('http://music.163.com/discover/artist/cat', params=params, headers=headers)

# 网页解析
soup = BeautifulSoup(r.content.decode(), 'html.parser')
body = soup.body


# In[8]:


hotartist_dic = {}
hot_artists = body.find_all('a', attrs={'class': 'msk'})
for artist in hot_artists:
    artist_id = artist['href'].replace('/artist?id=', '').strip()
    artist_name = artist['title'].replace('的音乐', '')
    try:
        hotartist_dic[artist_id] = artist_name
    except Exception as e:
        # 打印错误日志
        print(e)


# In[9]:


artist_dic = {}
artists = body.find_all('a', attrs={'class': 'nm nm-icn f-thide s-fc0'})
for artist in artists:
        artist_id = artist['href'].replace('/artist?id=', '').strip()
        artist_name = artist['title'].replace('的音乐', '')
        try:
            artist_dic[artist_id] = artist_name
        except Exception as e:
            # 打印错误日志
            print(e)


# In[10]:


artist_dic


# In[11]:


def save_artist(group_id, initial, hot_artist_dic, artisti_dic):
    params = {'id': group_id, 'initial': initial}
    r = requests.get('http://music.163.com/discover/artist/cat', params=params)

    # 网页解析
    soup = BeautifulSoup(r.content.decode(), 'html.parser')
    body = soup.body

    hot_artists = body.find_all('a', attrs={'class': 'msk'})
    artists = body.find_all('a', attrs={'class': 'nm nm-icn f-thide s-fc0'})
    for artist in hot_artists:
        artist_id = artist['href'].replace('/artist?id=', '').strip()
        artist_name = artist['title'].replace('的音乐', '')
        try:
            hot_artist_dic[artist_id] = artist_name
        except Exception as e:
            # 打印错误日志
            print(e)

    for artist in artists:
        artist_id = artist['href'].replace('/artist?id=', '').strip()
        artist_name = artist['title'].replace('的音乐', '')
        try:
            artist_dic[artist_id] = artist_name
        except Exception as e:
            # 打印错误日志
            print(e)
    #return artist_dic, hot_artist_dic


# In[12]:


gg = 1001
initial = 67
artist_dic = {}
hot_artist_dic = {} 
save_artist(gg, initial, hot_artist_dic, artist_dic  )


# In[13]:


artist_dic


# In[14]:


artist_dic = {}
hot_artist_dic = {} 
for i in range(65, 91):
    print(i)
    save_artist(gg, i, hot_artist_dic, artist_dic  )


# In[15]:


len(hot_artist_dic)


# In[16]:


len(artist_dic)


# ## 爬取所有的专辑信息（album_by _artist.py）

# In[68]:


list(hot_artist_dic.keys())[0]


# http://music.163.com/#/artist/album?id=89659&limit=400

# In[ ]:


headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Cookie': '_ntes_nnid=7eced19b27ffae35dad3f8f2bf5885cd,1476521011210; _ntes_nuid=7eced19b27ffae35dad3f8f2bf5885cd; usertrack=c+5+hlgB7TgnsAmACnXtAg==; Province=025; City=025; _ga=GA1.2.1405085820.1476521280; NTES_PASSPORT=6n9ihXhbWKPi8yAqG.i2kETSCRa.ug06Txh8EMrrRsliVQXFV_orx5HffqhQjuGHkNQrLOIRLLotGohL9s10wcYSPiQfI2wiPacKlJ3nYAXgM; P_INFO=hourui93@163.com|1476523293|1|study|11&12|jis&1476511733&mail163#jis&320100#10#0#0|151889&0|g37_client_check&mailsettings&mail163&study&blog|hourui93@163.com; JSESSIONID-WYYY=189f31767098c3bd9d03d9b968c065daf43cbd4c1596732e4dcb471beafe2bf0605b85e969f92600064a977e0b64a24f0af7894ca898b696bd58ad5f39c8fce821ec2f81f826ea967215de4d10469e9bd672e75d25f116a9d309d360582a79620b250625859bc039161c78ab125a1e9bf5d291f6d4e4da30574ccd6bbab70b710e3f358f%3A1476594130342; _iuqxldmzr_=25; __utma=94650624.1038096298.1476521011.1476588849.1476592408.6; __utmb=94650624.11.10.1476592408; __utmc=94650624; __utmz=94650624.1476521011.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)',
    'DNT': '1',
    'Host': 'music.163.com',
    'Pragma': 'no-cache',
    'Referer': 'http://music.163.com/',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'
}


# In[18]:


def save_albums(artist_id, albume_dic):
    params = {'id': artist_id, 'limit': '200'}
    # 获取歌手个人主页
    r = requests.get('http://music.163.com/artist/album', headers=headers, params=params)

    # 网页解析
    soup = BeautifulSoup(r.content.decode(), 'html.parser')
    body = soup.body

    albums = body.find_all('a', attrs={'class': 'tit s-fc0'})  # 获取所有专辑

    for album in albums:
        albume_id = album['href'].replace('/album?id=', '')
        albume_dic[albume_id] = artist_id


# In[19]:


albume_dic = {}
save_albums('2116', albume_dic)


# In[20]:


albume_dic


# ## 根据专辑信息爬取所有的歌曲信息（music_by _album.py）

# In[21]:


def save_music(album_id, music_dic):
    params = {'id': album_id}
    # 获取专辑对应的页面
    r = requests.get('http://music.163.com/album', headers=headers, params=params)
    # 网页解析
    soup = BeautifulSoup(r.content.decode(), 'html.parser')
    body = soup.body
    musics = body.find('ul', attrs={'class': 'f-hide'}).find_all('li')  # 获取专辑的所有音乐
    for music in musics:
        music = music.find('a')
        music_id = music['href'].replace('/song?id=', '')
        music_name = music.getText()
        music_dic[music_id] = [music_name, album_id]


# In[73]:


list(albume_dic.keys())[0]


# In[22]:


music_dic = {}
save_music('6423', music_dic)


# In[23]:


music_dic


# ## 根据歌曲信息爬取其评论条数（comments_by _music.py

# http://music.163.com/#/song?id=516997458
# 
# 
# 很遗憾的是评论数虽然也在详情页内，但是网易云音乐做了防爬处理，
# - 采用AJAX调用评论数API的方式填充评论相关数据，
# - 异步的特性导致我们爬到的页面中评论数是空，
# 
# 我们就找一找这个API吧，通关观察XHR请求发现是下面这个家伙..
# 
# 响应结果很丰富呢，所有评论相关的数据都有，不过经过观察发现这个API是经过加密处理的，不过没关系...
# 
# https://blog.csdn.net/python233/article/details/72825003
# 
# https://www.zhihu.com/question/36081767
# 

# In[29]:


params = {
    'csrf_token': ''
}

data = {
    'params': '5L+s/X1qDy33tb2sjT6to2T4oxv89Fjg1aYRkjgzpNPR6hgCpp0YVjNoTLQAwWu9VYvKROPZQj6qTpBK+sUeJovyNHsnU9/StEfZwCOcKfECFFtAvoNIpulj1TDOtBir',
    'encSecKey': '59079f3e07d6e240410018dc871bf9364f122b720c0735837d7916ac78d48a79ec06c6307e6a0e576605d6228bd0b377a96e1a7fc7c7ddc8f6a3dc6cc50746933352d4ec5cbe7bddd6dcb94de085a3b408d895ebfdf2f43a7c72fc783512b3c9efb860679a88ef21ccec5ff13592be450a1edebf981c0bf779b122ddbd825492'
    
}


# In[155]:


print(url)


# In[26]:


offset = 0
music_id = '65337'
url = 'http://music.163.com/api/v1/resource/comments/R_SO_4_'+ music_id + '?limit=20&offset=' + str(offset)
response = requests.post(url, headers=headers, data=data)
cj = response.json()
cj.keys()


# In[27]:


cj['total'],len(cj['comments']), len(cj['hotComments']), len(cj['topComments'])


# In[28]:


cj['comments'][0]


# ## 翻页的实现
# 
# limit是一页的数量，offset往后的偏移。
# - 比如limit是20，offset是40，就展示第三页的
# 
# http://music.163.com/api/v1/resource/comments/R_SO_4_516997458?limit=20&offset=0
# 
# http://music.163.com/api/v1/resource/comments/R_SO_4_516997458?limit=20&offset=20
# 
# http://music.163.com/api/v1/resource/comments/R_SO_4_516997458?limit=20&offset=40

# ## 另外一种方法

# In[129]:


from Crypto.Cipher import AES
import base64
import requests
import json
import time

# headers
headers = {
    'Host': 'music.163.com',
    'Connection': 'keep-alive',
    'Content-Length': '484',
    'Cache-Control': 'max-age=0',
    'Origin': 'http://music.163.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': '*/*',
    'DNT': '1',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6,zh-TW;q=0.4',
    'Cookie': 'JSESSIONID-WYYY=b66d89ed74ae9e94ead89b16e475556e763dd34f95e6ca357d06830a210abc7b685e82318b9d1d5b52ac4f4b9a55024c7a34024fddaee852404ed410933db994dcc0e398f61e670bfeea81105cbe098294e39ac566e1d5aa7232df741870ba1fe96e5cede8372ca587275d35c1a5d1b23a11e274a4c249afba03e20fa2dafb7a16eebdf6%3A1476373826753; _iuqxldmzr_=25; _ntes_nnid=7fa73e96706f26f3ada99abba6c4a6b2,1476372027128; _ntes_nuid=7fa73e96706f26f3ada99abba6c4a6b2; __utma=94650624.748605760.1476372027.1476372027.1476372027.1; __utmb=94650624.4.10.1476372027; __utmc=94650624; __utmz=94650624.1476372027.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)',
}



#获取params
def get_params(first_param, forth_param):
    iv = "0102030405060708"
    first_key = forth_param
    second_key = 16 * 'F'
    h_encText = AES_encrypt(first_param, first_key.encode(), iv.encode())
    h_encText = AES_encrypt(h_encText.decode(), second_key.encode(), iv.encode())
    return h_encText.decode()


# 获取encSecKey
def get_encSecKey():
    encSecKey = "257348aecb5e556c066de214e531faadd1c55d814f9be95fd06d6bff9f4c7a41f831f6394d5a3fd2e3881736d94a02ca919d952872e7d0a50ebfa1769a7a62d512f5f1ca21aec60bc3819a9c3ffca5eca9a0dba6d6f7249b06f5965ecfff3695b54e1c28f3f624750ed39e7de08fc8493242e26dbc4484a01c76f739e135637c"
    return encSecKey


# 解AES秘
def AES_encrypt(text, key, iv):
    pad = 16 - len(text) % 16
    text = text + pad * chr(pad)
    encryptor = AES.new(key, AES.MODE_CBC, iv)
    encrypt_text = encryptor.encrypt(text.encode())
    encrypt_text = base64.b64encode(encrypt_text)
    return encrypt_text


# 获取json数据
def get_json(url, data):
    response = requests.post(url, headers=headers, data=data)
    return response.content


# 传入post数据
def crypt_api(id, offset):
    url = "http://music.163.com/weapi/v1/resource/comments/R_SO_4_%s/?csrf_token=" % id
    first_param = "{rid:\"\", offset:\"%s\", total:\"true\", limit:\"20\", csrf_token:\"\"}" % offset
    forth_param = "0CoJUm6Qyw8W8jud"
    params = get_params(first_param, forth_param)
    encSecKey = get_encSecKey()
    data = {
        "params": params,
        "encSecKey": encSecKey
    }
    return url, data


# In[138]:


offset = 0
id = '516997458'
url, data = crypt_api(id, offset)
json_text = get_json(url, data)
json_dict = json.loads(json_text.decode("utf-8"))
comments_sum = json_dict['total']
comments_sum


# In[139]:


len(json_dict['comments'])


# In[140]:


json_dict['comments'][0]


# In[141]:


json_dict['comments'][4]


# In[135]:


offset = 20
id = '516997458'
url, data = crypt_api(id, offset)
json_text = get_json(url, data)
json_dict = json.loads(json_text.decode("utf-8"))
comments_sum = json_dict['total']
json_dict['comments'][0]


# In[136]:


offset = 40
id = '516997458'
url, data = crypt_api(id, offset)
json_text = get_json(url, data)
json_dict = json.loads(json_text.decode("utf-8"))
comments_sum = json_dict['total']
json_dict['comments'][0]

