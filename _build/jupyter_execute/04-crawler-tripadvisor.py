#!/usr/bin/env python
# coding: utf-8

# # 使用Selenium抓取TripAdvisor用户评论
# 
# 

# tripadvisor中文网站叫做猫途鹰 [www.tripadvisor.cn/](https://www.tripadvisor.cn/) 。
# - 对于酒店和景点的用户评论。
# - 但是默认只显示中文评论！
# - 需要点击所有评论才能显示其它语言的用户评论。
#     

# In[73]:


url = 'https://www.tripadvisor.cn/Hotel_Review-g294213-d1459920-Reviews-Somerset_Jiefangbei_Chongqing-Chongqing.html#REVIEWS'
print(url)


# In[67]:


from selenium import webdriver

browser = webdriver.Chrome()

url = 'https://www.tripadvisor.cn/Hotel_Review-g294213-d1459920-Reviews-Somerset_Jiefangbei_Chongqing-Chongqing.html#REVIEWS'

browser.get(url) 

loc = '#component_10 > div > div:nth-child(3) > div.location-review-filters-hr-ReviewFilters__filters_wrap--3zsVa > div.ui_columns > div.ui_column.is-3-tablet.is-shown-at-tablet > ul > li:nth-child(1) > label > span.location-review-review-list-parts-LanguageFilter__no_wrap--2Dckv'

browser.find_element_by_css_selector(loc).click()


# In[70]:


source = browser.page_source
soup = BeautifulSoup(source, 'html.parser')
comments = soup.find_all('div', {'class':'hotels-community-tab-common-Card__card--ihfZB hotels-community-tab-common-Card__section--4r93H'})


# In[72]:


comments[0]


# In[81]:


# user id
# https://www.tripadvisor.com/Profile/kimlette
comments[0].find('a', {'class', 'ui_header_link social-member-event-MemberEventOnObjectBlock__member--35-jC'}).text


# In[83]:


comments[0].find('span', {'class', 'default social-member-common-MemberHometown__hometown--3kM9S small'}).text


# In[80]:


# comment title
comments[0].find('a', {'class', 'location-review-review-list-parts-ReviewTitle__reviewTitleText--2tFRT'}).text


# In[79]:


# comment body text
comments[0].find('q', {'class', 'location-review-review-list-parts-ExpandableReview__reviewText--gOmRC'}).text

