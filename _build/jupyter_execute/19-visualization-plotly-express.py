#!/usr/bin/env python
# coding: utf-8

# # Plotly Express in Python
# 
# 来源：https://mp.weixin.qq.com/s/dg390bbxA_LEoggWg6dGlQ
# 
# Plotly Express 是一个新的高级 Python 可视化库：它是 Plotly.py 的高级封装，它为复杂的图表提供了一个简单的语法。 
# 
# 受 Seaborn 和 ggplot2 的启发，它专门设计为具有简洁，一致且易于学习的 API ：只需一次导入，您就可以在一个函数调用中创建丰富的交互式绘图，包括分面绘图（faceting）、地图、动画和趋势线。它带有数据集、颜色面板和主题，就像 Plotly.py 一样。Plotly是新一代的可视化神器，由TopQ量化团队开源。虽然Ploltly功能非常之强大，但是一直没有得到重视，主要原因还是其设置过于繁琐。因此，Plotly推出了其简化接口：Plotly_express，下文中统一简称为px。
# 
# Plotly Express 完全免费：凭借其宽松的开源 MIT 许可证，您可以随意使用它（是的，甚至在商业产品中！）。 
# 
# 最重要的是，Plotly Express 与 Plotly 生态系统的其他部分完全兼容：在您的 Dash 应用程序中使用它，使用 Orca 将您的数据导出为几乎任何文件格式，或使用JupyterLab 图表编辑器在 GUI 中编辑它们！
# 
# 
# https://plotly.com/python/plotly-express/

# ## 安装plotly_express

# In[1]:


pip install plotly_express


# ## 导入数据

# In[3]:


import pandas as pd
import numpy as np
import plotly.express as px  

# 数据集
gapminder = px.data.gapminder()
gapminder.head()  # 取出前5条数据


# In[4]:


gapminder.shape


# ## 可视化示例

# In[5]:


# line 图
fig = px.line(
  gapminder,  # 数据集
  x="year",  # 横坐标
  y="lifeExp",  # 纵坐标
  color="continent",  # 颜色的数据
  line_group="continent",  # 线性分组
  hover_name="country",   # 悬停hover的数据
  line_shape="spline",  # 线的形状
  render_mode="svg"  # 生成的图片模式
)
fig.show()


# In[6]:


# area 图
fig = px.area(
  gapminder,  # 数据集
  x="year",  # 横坐标
  y="pop",  # 纵坐标
  color="continent",   # 颜色
  line_group="country"  # 线性组别
)
fig.show()


# In[8]:


px.scatter(
  gapminder   # 绘图DataFrame数据集
  ,x="gdpPercap"  # 横坐标
  ,y="lifeExp"  # 纵坐标
  ,color="continent"  # 区分颜色
  ,size="pop"   # 区分圆的大小
  ,size_max=60  # 散点大小
)


# In[51]:


fig = px.scatter(
  gapminder   # 绘图使用的数据
  ,x="gdpPercap" # 横纵坐标使用的数据
  ,y="lifeExp"  # 纵坐标数据
  ,color="continent"  # 区分颜色的属性
  ,size="pop"   # 区分圆的大小
  ,size_max=60  # 圆的最大值
  ,hover_name="country"  # 图中可视化最上面的名字
  ,animation_frame="year"  # 横轴滚动栏的属性year
  ,animation_group="country"  # 标注的分组
  ,facet_col="continent"   # 按照国家country属性进行分格显示
  ,log_x=True  # 横坐标表取对数
  ,range_x=[100,100000]  # 横轴取值范围
  ,range_y=[25,90]  # 纵轴范围
  ,labels=dict(pop="Populations",  # 属性名字的变化，更直观
               gdpPercap="GDP per Capital",
               lifeExp="Life Expectancy")
)

fig.show()


# ### 导出HTML

# In[52]:


fig.write_html("./data/plotly_express_gapminder.html")


# In[10]:


px.choropleth(
  gapminder,  # 数据集
  locations="iso_alpha",  # 配合颜色color显示
  color="lifeExp", # 颜色的字段选择
  hover_name="country",  # 悬停字段名字
  animation_frame="year",  # 注释
  color_continuous_scale=px.colors.sequential.Plasma,  # 颜色变化
  projection="natural earth"  # 全球地图
             )


# In[11]:


fig = px.scatter_geo(
  gapminder,   # 数据
  locations="iso_alpha",  # 配合颜色color显示
  color="continent", # 颜色
  hover_name="country", # 悬停数据
  size="pop",  # 大小
  animation_frame="year",  # 数据帧的选择
  projection="natural earth"  # 全球地图
                    )

fig.show()


# In[12]:


px.scatter_geo(gapminder, # 数据集
locations="iso_alpha",  # 配和color显示颜色
color="continent",  # 颜色的字段显示
hover_name="country",  # 悬停数据
size="pop",  # 大小
animation_frame="year"  # 数据联动变化的选择
#,projection="natural earth"   # 去掉projection参数
)


# In[14]:


fig = px.line_geo(
  gapminder,  # 数据集
  locations="iso_alpha",  # 配合和color显示数据
  color="continent",  # 颜色
  projection="orthographic")   # 球形的地图
fig.show()


# In[11]:


import plotly.graph_objects as go

fig = go.Figure(data=go.Scattergeo(
    lat = [40.7127, 51.5072],
    lon = [-74.0059, 0.1275],
    mode = 'lines',
    line = dict(width = 2, color = 'blue'),
))

fig.update_layout(
    title_text = 'London to NYC Great Circle',
    showlegend = False,
    geo = dict(
        resolution = 50,
        showland = True,
        showlakes = True,
        landcolor = 'rgb(204, 204, 204)',
        countrycolor = 'rgb(204, 204, 204)',
        lakecolor = 'rgb(255, 255, 255)',
        projection_type = "equirectangular",
        coastlinewidth = 2,
        lataxis = dict(
            range = [20, 60],
            showgrid = True,
            dtick = 10
        ),
        lonaxis = dict(
            range = [-100, 20],
            showgrid = True,
            dtick = 20
        ),
    )
)

fig.show()


# In[3]:


import plotly.graph_objects as go
import pandas as pd

df_airports = pd.read_csv('./data/2011_february_us_airport_traffic.csv')
df_airports.head()


# In[4]:


df_flight_paths = pd.read_csv('./data/2011_february_aa_flight_paths.csv')
df_flight_paths.head()


# In[13]:


fig = go.Figure()

fig.add_trace(go.Scattergeo(
    locationmode = 'USA-states',
    lon = df_airports['long'],
    lat = df_airports['lat'],
    hoverinfo = 'text',
    text = df_airports['airport'],
    mode = 'markers',
    marker = dict(
        size = 2,
        color = 'rgb(255, 0, 0)',
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))

for i in range(len(df_flight_paths)):
    fig.add_trace(
        go.Scattergeo(
            locationmode = 'USA-states',
            lon = [df_flight_paths['start_lon'][i], df_flight_paths['end_lon'][i]],
            lat = [df_flight_paths['start_lat'][i], df_flight_paths['end_lat'][i]],
            mode = 'lines',
            line = dict(width = 1,color = 'red'),
            opacity = float(df_flight_paths['cnt'][i]) / float(df_flight_paths['cnt'].max()),
        )
    )

fig.update_layout(
    title_text = 'Feb. 2011 American Airline flight paths<br>(Hover for airport names)',
    showlegend = False,
    geo = dict(
        scope = 'north america',
        projection_type = 'azimuthal equal area',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
    ),
)

fig.show()


# In[15]:


iris = px.data.iris()


# In[16]:


fig = px.scatter(
  iris,  # 数据集
  x="sepal_width",  # 横坐标
  y="sepal_length"  # 纵坐标
                )
fig.show()


# In[17]:


px.scatter(
  iris,  # 数据集
  x="sepal_width",  # 横坐标
  y="sepal_length",  # 纵坐标
  color="species"
                )


# In[18]:


px.scatter(
  iris,  # 数据集
  x="sepal_width", # 横坐标
  y="sepal_length",  # 纵坐标
  color="species",  # 颜色
  marginal_x="histogram",  # 横坐标直方图
  marginal_y="rug"   # 细条图
)


# In[19]:


px.scatter(
  iris,  # 数据集
  x="sepal_width",  # 横坐标
  y="sepal_length",  # 纵坐标
  color="species",  # 颜色
  marginal_y="violin",  # 纵坐标小提琴图
  marginal_x="box",  # 横坐标箱型图
  trendline="ols"  # 趋势线
)


# In[20]:


px.scatter_matrix(
  iris,  # 数据
  dimensions=["sepal_width","sepal_length","petal_width","petal_length"],  # 维度选择
  color="species")  # 颜色


# In[21]:


px.parallel_coordinates(
  iris,   # 数据集
  color="species_id",  # 颜色
  labels={"species_id":"Species",  # 各种标签值
          "sepal_width":"Sepal Width",
          "sepal_length":"Sepal Length",
          "petal_length":"Petal Length",
          "petal_width":"Petal Width"},
  color_continuous_scale=px.colors.diverging.Tealrose,
  color_continuous_midpoint=2)


# In[22]:


# 对当前值加上下两个误差值
iris["e"] = iris["sepal_width"] / 100
px.scatter(
  iris,  # 绘图数据集
  x="sepal_width",  # 横坐标
  y="sepal_length",  # 纵坐标
  color="species",  # 颜色值
  error_x="e",  # 横轴误差
  error_y="e"  # 纵轴误差
          )


# In[23]:


px.density_contour(
  iris,  # 绘图数据集
  x="sepal_width",  # 横坐标
  y="sepal_length",  # 纵坐标值
  color="species"  # 颜色
)


# In[24]:


px.density_contour(
  iris, # 数据集
  x="sepal_width",  # 横坐标值
  y="sepal_length",  # 纵坐标值
  color="species",  # 颜色
  marginal_x="rug",  # 横轴为线条图
  marginal_y="histogram"   # 纵轴为直方图
                  )


# In[25]:


px.density_heatmap(
  iris,  # 数据集
  x="sepal_width",   # 横坐标值
  y="sepal_length",  # 纵坐标值
  marginal_y="rug",  # 纵坐标值为线型图
  marginal_x="histogram"  # 直方图
                  )


# In[26]:


tips = px.data.tips()


# In[27]:


fig = px.parallel_categories(
  tips,  # 数据集 
  color="size",  # 颜色
  color_continuous_scale=px.colors.sequential.Inferno)  # 颜色变化取值
fig.show()


# In[28]:


px.bar(
  tips,  # 数据集
  x="sex",  # 横轴
  y="total_bill",  # 纵轴
  color="smoker",  # 颜色参数取值
  barmode="group")


# In[29]:


fig = px.bar(
  tips,  # 数据集
  x="sex",  # 横轴
  y="total_bill",  # 纵轴
  color="smoker",  # 颜色参数取值
  barmode="group",  # 柱状图模式取值
  facet_row="time",  # 行取值
  facet_col="day",  # 列元素取值
  category_orders={
    "day": ["Thur","Fri","Sat","Sun"],  # 分类顺序
    "time":["Lunch", "Dinner"]})
fig.show()


# In[30]:


fig = px.histogram(
  tips,  # 绘图数据集
  x="sex",  # 横轴为性别
  y="tip",  # 纵轴为费用
  histfunc="avg",  # 直方图显示的函数
  color="smoker",  # 颜色
  barmode="group",  # 柱状图模式
  facet_row="time",  # 行取值
  facet_col="day",   # 列取值
  category_orders={  # 分类顺序
    "day":["Thur","Fri","Sat","Sun"],
    "time":["Lunch","Dinner"]}
)

fig.show()


# In[31]:


# notched=True显示连接处的锥形部分
px.box(tips,  # 数据集
       x="day",  # 横轴数据
       y="total_bill",  # 纵轴数据
       color="smoker",  # 颜色
       notched=True)  # 连接处的锥形部分显示出来


# In[32]:


px.box(
  tips,  # 数据集
  x="day",  # 横轴
 y="total_bill",  # 纵轴 
 color="smoker",  # 颜色
#         notched=True   # 隐藏参数
      )


# In[33]:


px.violin(
    tips,   # 数据集
    x="smoker",  # 横轴坐标
    y="tip",  # 纵轴坐标  
    color="sex",   # 颜色参数取值
    box=True,   # box是显示内部的箱体
    points="all",  # 同时显示数值点
    hover_data=tips.columns)  # 结果中显示全部数据


# In[34]:


wind = px.data.wind()


# In[36]:


fig = px.scatter_polar(
    wind,  # 数据集
    r="frequency",  # 半径
    theta="direction",  # 角度
    color="strength",  # 颜色
    symbol="strength",  # 线性闭合
    color_discrete_sequence=px.colors.sequential.Plasma_r)  # 颜色变化
fig.show()


# In[37]:


fig = px.line_polar(
    wind,  # 数据集
    r="frequency",  # 半径
    theta="direction",  # 角度
    color="strength",  # 颜色
    line_close=True,  # 线性闭合
    color_discrete_sequence=px.colors.sequential.Plasma_r)  # 颜色变化
fig.show()


# In[38]:


fig = px.bar_polar(
    wind,   # 数据集
    r="frequency",   # 半径
    theta="direction",  # 角度
    color="strength",  # 颜色
    template="plotly_dark",  # 主题
    color_discrete_sequence=px.colors.sequential.Plasma_r)  # 颜色变化
fig.show()


# ## 选择颜色

# In[39]:


px.colors.qualitative.swatches()


# In[40]:


px.colors.sequential.swatches()


# ## 选择template

# In[41]:


px.scatter(
    iris,  # 数据集
    x="sepal_width",  # 横坐标值
    y="sepal_length",  # 纵坐标取值
    color="species",  # 颜色
    marginal_x="box",  # 横坐标为箱型图
    marginal_y="histogram",  # 纵坐标为直方图
    height=600,  # 高度
    trendline="ols",  # 显示趋势线
    template="plotly")  # 主题


# In[42]:


px.scatter(
    iris,  # 数据集
    x="sepal_width",  # 横坐标值
    y="sepal_length",  # 纵坐标取值
    color="species",  # 颜色
    marginal_x="box",  # 横坐标为箱型图
    marginal_y="histogram",  # 纵坐标为直方图
    height=600,  # 高度
    trendline="ols",  # 显示趋势线
    template="plotly_white")  # 主题    


# In[43]:


px.scatter(
    iris,  # 数据集
    x="sepal_width",  # 横坐标值
    y="sepal_length",  # 纵坐标取值
    color="species",  # 颜色
    marginal_x="box",  # 横坐标为箱型图
    marginal_y="histogram",  # 纵坐标为直方图
    height=600,  # 高度
    trendline="ols",  # 显示趋势线
    template="plotly_dark")  # 主题   


# 本文中利用大量的篇幅讲解了如何通过plotly_express来绘制：柱状图、线型图、散点图、小提琴图、极坐标图等各种常见的图形。通过观察上面Plotly_express绘制图形过程，我们不难发现它有三个主要的优点：
# 
# - 快速出图，少量的代码就能满足多数的制图要求。基本上都是几个参数的设置我们就能快速出图
# - 图形漂亮，绘制出来的可视化图形颜色亮丽，也有很多的颜色供选择。
# - 图形是动态可视化的。文章中图形都是截图，如果是在Jupyter notebook中都是动态图形
