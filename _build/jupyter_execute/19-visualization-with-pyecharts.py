#!/usr/bin/env python
# coding: utf-8

# 
# # 使用PyEcharts进行可视化
# 
# Visualization with PyEcharts
# 
# 
# ![image.png](images/author.png)
# 

# ## Echarts
# 
# https://echarts.apache.org/examples/zh/index.html
# 
# - 第一步，选取图类型 
# - 第二步，修改图
# - 第三步，点击download下载html文件
# - 第四步，修改下载的html文件
# 
# 案例1：散点图
# https://echarts.apache.org/examples/zh/editor.html?c=bubble-gradient

# 案例2：Put echarts into a html
# 
# Note: set the **height** of section.
# 
# **Question**: How to add more echarts into a html?

# 案例3：读取json数据
# 
# https://echarts.apache.org/examples/zh/editor.html?c=scatter-life-expectancy-timeline
# 
# 前端的开发的html给我们的时候，由于内部有一些ajax请求的json的数据，需要在一个web server中查看，每次放到http服务器太麻烦。还是直接用python造一个最方便。最简单的，直接用
# 
# > python3 -m http.server
# 
# 同时，读取json数据时，需要调用jquery
# 
# ```
# <script type="text/javascript" src="https://cdn.staticfile.org/jquery/1.10.2/jquery.min.js"></script>
# ```
# 

# ## pyecharts安装
#  https://github.com/pyecharts/pyecharts
# 
# >  pip install pyecharts -U
# 
#     
# - pip install echarts-countries-pypkg
# - pip install echarts-china-provinces-pypkg
# - pip install echarts-china-cities-pypkg

# ## pyecharts使用简介
# Echarts 是一个由百度开源的数据可视化，凭借着良好的交互性，精巧的图表设计，得到了众多开发者的认可。而 Python 是一门富有表达力的语言，很适合用于数据处理。当数据分析遇上数据可视化时，pyecharts 诞生了。https://pyecharts.org/#/
# 
# - 配置项: 全局配置项 | 系列配置项
# - 基本使用: 图表 API | 示例数据 | 全局变量
# - 图表类型: 基本图表 | 直角坐标系图表 | 地理图表 | 3D 图表 | 组合图表 | HTML 组件
# - 进阶话题: 参数传递 | 数据格式 | 定制主题 | 定制地图 | 渲染图片 | Notebook | 原生 Javascript | 资源引用
# 

# In[5]:


from pyecharts.charts import Bar

bar = Bar()
bar.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
bar.add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
# render 会生成本地 HTML 文件，默认会在当前目录生成 render.html 文件
# 也可以传入路径参数，如 bar.render("mycharts.html")
bar.render_notebook()


# In[6]:


from pyecharts.charts import Bar

bar = (
    Bar()
    .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    .add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
)
bar.render_notebook()


# In[7]:


from pyecharts.charts import Bar
from pyecharts import options as opts

bar = (
    Bar()
    .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    .add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
    .set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"))
    # 或者直接使用字典参数
    # .set_global_opts(title_opts={"text": "主标题", "subtext": "副标题"})
)
bar.render_notebook()


# In[8]:


from pyecharts.charts import Bar
from pyecharts import options as opts
# 内置主题类型可查看 pyecharts.globals.ThemeType
from pyecharts.globals import ThemeType

bar = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
    .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    .add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
    .add_yaxis("商家B", [15, 6, 45, 20, 35, 66])
    .set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"))
)

bar.render_notebook()


# ## PyEcharts Gallery
# 
# https://github.com/pyecharts/pyecharts-gallery
# 
# ![image.png](images/pyecharts.png)

# ## Bar
# 
# https://gallery.pyecharts.org/#/Bar/bar_base

# In[146]:


# vis
from pyecharts.charts import Bar
from pyecharts import options as opts

# V1 版本开始支持链式调用
bar = (
    Bar()
    .add_xaxis(["衬衫", "毛衣", "领带", "裤子", "风衣", "高跟鞋", "袜子"])
    .add_yaxis("商家A", [114, 55, 27, 101, 125, 27, 105])
    .add_yaxis("商家B", [57, 134, 137, 129, 145, 60, 49])
    .set_global_opts(title_opts=opts.TitleOpts(title="某商场销售情况"))
)
bar.render_notebook()


# ## Bar3D

# In[147]:


# vis
import random

from pyecharts import options as opts
from pyecharts.charts import Bar3D
from pyecharts.faker import Faker


data = [(i, j, random.randint(0, 12)) for i in range(6) for j in range(24)]
bar3d = (
    Bar3D()
    .add(
        "",
        [[d[1], d[0], d[2]] for d in data],
        xaxis3d_opts=opts.Axis3DOpts(Faker.clock, type_="category"),
        yaxis3d_opts=opts.Axis3DOpts(Faker.week_en, type_="category"),
        zaxis3d_opts=opts.Axis3DOpts(type_="value"),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(max_=20),
        title_opts=opts.TitleOpts(title="Bar3D-基本示例"),
    )
    #.render("bar3d_base.html")
)

bar3d.render_notebook()


# ## EffectScatter

# In[116]:


# vis
from pyecharts import options as opts
from pyecharts.charts import EffectScatter
from pyecharts.faker import Faker

c = (
    EffectScatter()
    .add_xaxis(Faker.choose())
    .add_yaxis("", Faker.values())
    .set_global_opts(title_opts=opts.TitleOpts(title="EffectScatter-基本示例"))
    #.render("effectscatter_base.html")
)

c.render_notebook()


# ## Funnel

# In[148]:


from pyecharts import options as opts
from pyecharts.charts import Funnel
from pyecharts.faker import Faker

c = (
    Funnel()
    .add("商品", [list(z) for z in zip(Faker.choose(), Faker.values())])
    .set_global_opts(title_opts=opts.TitleOpts(title="Funnel-基本示例"))
    #.render("funnel_base.html")
)

c.render_notebook()


# ## Gauge

# In[149]:


from pyecharts import options as opts
from pyecharts.charts import Gauge

c = (
    Gauge()
    .add("", [("完成率", 55.6)])
    .set_global_opts(title_opts=opts.TitleOpts(title="Gauge-基本示例"))
    #.render("gauge_base.html")
)

c.render_notebook()


# ## Geo

# In[150]:


from pyecharts.charts import Geo

data = [
    ("海门", 9),("鄂尔多斯", 12),("招远", 12),("舟山", 12),("齐齐哈尔", 14),("盐城", 15),
    ("赤峰", 16),("青岛", 18),("乳山", 18),("金昌", 19),("泉州", 21),("莱西", 21),
    ("日照", 21),("胶南", 22),("南通", 23),("拉萨", 24),("云浮", 24),("梅州", 25),
    ("文登", 25),("上海", 25),("攀枝花", 25),("威海", 25),("承德", 25),("厦门", 26),
    ("汕尾", 26),("潮州", 26),("丹东", 27),("太仓", 27),("曲靖", 27),("烟台", 28),
    ("福州", 29),("瓦房店", 30),("即墨", 30),("抚顺", 31),("玉溪", 31),("张家口", 31),
    ("阳泉", 31),("莱州", 32),("湖州", 32),("汕头", 32),("昆山", 33),("宁波", 33),
    ("湛江", 33),("揭阳", 34),("荣成", 34),("连云港", 35),("葫芦岛", 35),("常熟", 36),
    ("东莞", 36),("河源", 36),("淮安", 36),("泰州", 36),("南宁", 37),("营口", 37),
    ("惠州", 37),("江阴", 37),("蓬莱", 37),("韶关", 38),("嘉峪关", 38),("广州", 38),
    ("延安", 38),("太原", 39),("清远", 39),("中山", 39),("昆明", 39),("寿光", 40),
    ("盘锦", 40),("长治", 41),("深圳", 41),("珠海", 42),("宿迁", 43),("咸阳", 43),
    ("铜川", 44),("平度", 44),("佛山", 44),("海口", 44),("江门", 45),("章丘", 45),
    ("肇庆", 46),("大连", 47),("临汾", 47),("吴江", 47),("石嘴山", 49),("沈阳", 50),
    ("苏州", 50),("茂名", 50),("嘉兴", 51),("长春", 51),("胶州", 52),("银川", 52),
    ("张家港", 52),("三门峡", 53),("锦州", 54),("南昌", 54),("柳州", 54),("三亚", 54),
    ("自贡", 56),("吉林", 56),("阳江", 57),("泸州", 57),("西宁", 57),("宜宾", 58),
    ("呼和浩特", 58),("成都", 58),("大同", 58),("镇江", 59),("桂林", 59),("张家界", 59),
    ("宜兴", 59),("北海", 60),("西安", 61),("金坛", 62),("东营", 62),("牡丹江", 63),
    ("遵义", 63),("绍兴", 63),("扬州", 64),("常州", 64),("潍坊", 65),("重庆", 66),
    ("台州", 67),("南京", 67),("滨州", 70),("贵阳", 71),("无锡", 71),("本溪", 71),
    ("克拉玛依", 72),("渭南", 72),("马鞍山", 72),("宝鸡", 72),("焦作", 75),("句容", 75),
    ("北京", 79),("徐州", 79),("衡水", 80),("包头", 80),("绵阳", 80),("乌鲁木齐", 84),
    ("枣庄", 84),("杭州", 84),("淄博", 85),("鞍山", 86),("溧阳", 86),("库尔勒", 86),
    ("安阳", 90),("开封", 90),("济南", 92),("德阳", 93),("温州", 95),("九江", 96),
    ("邯郸", 98),("临安", 99),("兰州", 99),("沧州", 100),("临沂", 103),("南充", 104),
    ("天津", 105),("富阳", 106),("泰安", 112),("诸暨", 112),("郑州", 113),("哈尔滨", 114),
    ("聊城", 116),("芜湖", 117),("唐山", 119),("平顶山", 119),("邢台", 119),("德州", 120),
    ("济宁", 120),("荆州", 127),("宜昌", 130),("义乌", 132),("丽水", 133),("洛阳", 134),
    ("秦皇岛", 136),("株洲", 143),("石家庄", 147),("莱芜", 148),("常德", 152),("保定", 153),
    ("湘潭", 154),("金华", 157),("岳阳", 169),("长沙", 175),("衢州", 177),("廊坊", 193),
    ("菏泽", 194),("合肥", 229),("武汉", 273),("大庆", 279)]


# In[151]:


c = (
    Geo()
    .add_schema(maptype="china")
    .add("geo", [list(z) for z in data])
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(is_piecewise=True),
        title_opts=opts.TitleOpts(title="Geo-VisualMap（分段型）"),
    )
    #.render("geo_visualmap_piecewise.html")
)

c.render_notebook()


# In[152]:


from pyecharts.globals import ChartType

c = (
    Geo()
    .add_schema(maptype="china")
    .add(
        "geo",
        [list(z) for z in data],
        type_=ChartType.HEATMAP,
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(),
        title_opts=opts.TitleOpts(title="Geo-HeatMap"),
    )
)

c.render_notebook()


# ## Graph

# In[153]:


import json

from pyecharts import options as opts
from pyecharts.charts import Graph


with open("les-miserables.json", "r", encoding="utf-8") as f:
    j = json.load(f)
    nodes = j["nodes"]
    links = j["links"]
    categories = j["categories"]


# In[154]:


nodes[0]


# In[124]:


links[0] 


# In[155]:


c = (
    Graph(init_opts=opts.InitOpts(width="1000px", height="600px"))
    .add(
        "",
        nodes=nodes,
        links=links,
        categories=categories,
        layout="circular",
        is_rotate_label=True,
        linestyle_opts=opts.LineStyleOpts(color="source", curve=0.5),
        label_opts=opts.LabelOpts(position="right"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Graph-Les Miserables"),
        legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%"),
    )
    #.render("graph_les_miserables.html")
)
c.render_notebook()


# In[126]:


c = (
    Graph(init_opts=opts.InitOpts(width="1000px", height="600px"))
    .add(
        "",
        nodes=nodes,
        links=links,
        categories=categories,
        layout="none",
        is_rotate_label=True,
        linestyle_opts=opts.LineStyleOpts(color="source", curve=0.3),
        label_opts=opts.LabelOpts(position="right"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Graph-Les Miserables"),
        legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%"),
    )
    #.render("graph_les_miserables.html")
)
c.render_notebook()


# In[127]:


# vis
from pyecharts import options as opts
from pyecharts.charts import Graph
import networkx as nx

ba=nx.nx.karate_club_graph()
links = []
nodes = []
for i in ba.edges:
    links.append({"source": i[0], "target": i[1]})
for i in ba.nodes:
    nodes.append({'name': i})
    
c = (
    Graph()
    .add("", nodes, links, repulsion=80)
    .set_global_opts(title_opts=opts.TitleOpts(title="Graph-基本示例"))
    #.render("graph_base.html")
)
c.render_notebook()


# ## HeatMap

# In[139]:


# vis
import random
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.faker import Faker

value = [[i, j, random.randint(0, 50)] for i in range(24) for j in range(7)]
c = (
    HeatMap()
    .add_xaxis(Faker.clock)
    .add_yaxis(
        "series0",
        Faker.week,
        value,
        label_opts=opts.LabelOpts(is_show=True, position="inside"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="HeatMap-Label 显示"),
        visualmap_opts=opts.VisualMapOpts(),
    )
    #.render("heatmap_with_label_show.html")
)

c.render_notebook()


# ## Line3D

# In[130]:


import math

from pyecharts import options as opts
from pyecharts.charts import Line3D
from pyecharts.faker import Faker

data = []
for t in range(0, 25000):
    _t = t / 1000
    x = (1 + 0.25 * math.cos(75 * _t)) * math.cos(_t)
    y = (1 + 0.25 * math.cos(75 * _t)) * math.sin(_t)
    z = _t + 2.0 * math.sin(75 * _t)
    data.append([x, y, z])


# In[131]:


c = (
    Line3D()
    .add(
        "",
        data,
        xaxis3d_opts=opts.Axis3DOpts(Faker.clock, type_="value"),
        yaxis3d_opts=opts.Axis3DOpts(Faker.week_en, type_="value"),
        grid3d_opts=opts.Grid3DOpts(
            width=100, depth=100, rotate_speed=150, is_rotate=True
        ),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            max_=30, min_=0, range_color=Faker.visual_color
        ),
        title_opts=opts.TitleOpts(title="Line3D-旋转的弹簧"),
    )
    #.render("line3d_autorotate.html")
)
c.render_notebook()


# In[137]:


# vis
import pyecharts.options as opts
from pyecharts.charts import ThemeRiver

x_data = ["DQ", "TY", "SS", "QG", "SY", "DD"]
y_data = [
    ["2015/11/08", 10, "DQ"],
    ["2015/11/09", 15, "DQ"],
    ["2015/11/10", 35, "DQ"],
    ["2015/11/11", 38, "DQ"],
    ["2015/11/12", 22, "DQ"],
    ["2015/11/13", 16, "DQ"],
    ["2015/11/14", 7, "DQ"],
    ["2015/11/15", 2, "DQ"],
    ["2015/11/16", 17, "DQ"],
    ["2015/11/17", 33, "DQ"],
    ["2015/11/18", 40, "DQ"],
    ["2015/11/19", 32, "DQ"],
    ["2015/11/20", 26, "DQ"],
    ["2015/11/21", 35, "DQ"],
    ["2015/11/22", 40, "DQ"],
    ["2015/11/23", 32, "DQ"],
    ["2015/11/24", 26, "DQ"],
    ["2015/11/25", 22, "DQ"],
    ["2015/11/26", 16, "DQ"],
    ["2015/11/27", 22, "DQ"],
    ["2015/11/28", 10, "DQ"],
    ["2015/11/08", 35, "TY"],
    ["2015/11/09", 36, "TY"],
    ["2015/11/10", 37, "TY"],
    ["2015/11/11", 22, "TY"],
    ["2015/11/12", 24, "TY"],
    ["2015/11/13", 26, "TY"],
    ["2015/11/14", 34, "TY"],
    ["2015/11/15", 21, "TY"],
    ["2015/11/16", 18, "TY"],
    ["2015/11/17", 45, "TY"],
    ["2015/11/18", 32, "TY"],
    ["2015/11/19", 35, "TY"],
    ["2015/11/20", 30, "TY"],
    ["2015/11/21", 28, "TY"],
    ["2015/11/22", 27, "TY"],
    ["2015/11/23", 26, "TY"],
    ["2015/11/24", 15, "TY"],
    ["2015/11/25", 30, "TY"],
    ["2015/11/26", 35, "TY"],
    ["2015/11/27", 42, "TY"],
    ["2015/11/28", 42, "TY"],
    ["2015/11/08", 21, "SS"],
    ["2015/11/09", 25, "SS"],
    ["2015/11/10", 27, "SS"],
    ["2015/11/11", 23, "SS"],
    ["2015/11/12", 24, "SS"],
    ["2015/11/13", 21, "SS"],
    ["2015/11/14", 35, "SS"],
    ["2015/11/15", 39, "SS"],
    ["2015/11/16", 40, "SS"],
    ["2015/11/17", 36, "SS"],
    ["2015/11/18", 33, "SS"],
    ["2015/11/19", 43, "SS"],
    ["2015/11/20", 40, "SS"],
    ["2015/11/21", 34, "SS"],
    ["2015/11/22", 28, "SS"],
    ["2015/11/23", 26, "SS"],
    ["2015/11/24", 37, "SS"],
    ["2015/11/25", 41, "SS"],
    ["2015/11/26", 46, "SS"],
    ["2015/11/27", 47, "SS"],
    ["2015/11/28", 41, "SS"],
    ["2015/11/08", 10, "QG"],
    ["2015/11/09", 15, "QG"],
    ["2015/11/10", 35, "QG"],
    ["2015/11/11", 38, "QG"],
    ["2015/11/12", 22, "QG"],
    ["2015/11/13", 16, "QG"],
    ["2015/11/14", 7, "QG"],
    ["2015/11/15", 2, "QG"],
    ["2015/11/16", 17, "QG"],
    ["2015/11/17", 33, "QG"],
    ["2015/11/18", 40, "QG"],
    ["2015/11/19", 32, "QG"],
    ["2015/11/20", 26, "QG"],
    ["2015/11/21", 35, "QG"],
    ["2015/11/22", 40, "QG"],
    ["2015/11/23", 32, "QG"],
    ["2015/11/24", 26, "QG"],
    ["2015/11/25", 22, "QG"],
    ["2015/11/26", 16, "QG"],
    ["2015/11/27", 22, "QG"],
    ["2015/11/28", 10, "QG"],
    ["2015/11/08", 10, "SY"],
    ["2015/11/09", 15, "SY"],
    ["2015/11/10", 35, "SY"],
    ["2015/11/11", 38, "SY"],
    ["2015/11/12", 22, "SY"],
    ["2015/11/13", 16, "SY"],
    ["2015/11/14", 7, "SY"],
    ["2015/11/15", 2, "SY"],
    ["2015/11/16", 17, "SY"],
    ["2015/11/17", 33, "SY"],
    ["2015/11/18", 40, "SY"],
    ["2015/11/19", 32, "SY"],
    ["2015/11/20", 26, "SY"],
    ["2015/11/21", 35, "SY"],
    ["2015/11/22", 4, "SY"],
    ["2015/11/23", 32, "SY"],
    ["2015/11/24", 26, "SY"],
    ["2015/11/25", 22, "SY"],
    ["2015/11/26", 16, "SY"],
    ["2015/11/27", 22, "SY"],
    ["2015/11/28", 10, "SY"],
    ["2015/11/08", 10, "DD"],
    ["2015/11/09", 15, "DD"],
    ["2015/11/10", 35, "DD"],
    ["2015/11/11", 38, "DD"],
    ["2015/11/12", 22, "DD"],
    ["2015/11/13", 16, "DD"],
    ["2015/11/14", 7, "DD"],
    ["2015/11/15", 2, "DD"],
    ["2015/11/16", 17, "DD"],
    ["2015/11/17", 33, "DD"],
    ["2015/11/18", 4, "DD"],
    ["2015/11/19", 32, "DD"],
    ["2015/11/20", 26, "DD"],
    ["2015/11/21", 35, "DD"],
    ["2015/11/22", 40, "DD"],
    ["2015/11/23", 32, "DD"],
    ["2015/11/24", 26, "DD"],
    ["2015/11/25", 22, "DD"],
    ["2015/11/26", 16, "DD"],
    ["2015/11/27", 22, "DD"],
    ["2015/11/28", 10, "DD"],
]

c = (
    ThemeRiver(init_opts=opts.InitOpts(width="1600px", height="800px"))
    .add(
        series_name=x_data,
        data=y_data,
        singleaxis_opts=opts.SingleAxisOpts(
            pos_top="50", pos_bottom="50", type_="time"
        ),
    )
    .set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line")
    )
    #.render("theme_river.html")
)

c.render_notebook()


# In[138]:


# vis
from pyecharts import options as opts
from pyecharts.charts import Boxplot

v1 = [
    [850, 740, 900, 1070, 930, 850, 950, 980, 980, 880, 1000, 980],
    [960, 940, 960, 940, 880, 800, 850, 880, 900, 840, 830, 790],
]
v2 = [
    [890, 810, 810, 820, 800, 770, 760, 740, 750, 760, 910, 920],
    [890, 840, 780, 810, 760, 810, 790, 810, 820, 850, 870, 870],
]
c = Boxplot()
c.add_xaxis(["expr1", "expr2"])
c.add_yaxis("A", c.prepare_data(v1))
c.add_yaxis("B", c.prepare_data(v2))
c.set_global_opts(title_opts=opts.TitleOpts(title="BoxPlot-基本示例"))
c.render_notebook()


# ## WordCloud

# In[156]:


# vis
import pyecharts.options as opts
from pyecharts.charts import WordCloud

import jieba.analyse
import numpy as np

with open('../data/gov_reports1954-2017.txt', 'r') as f:
    reports = f.readlines()

txt = reports[-1]
tf = jieba.analyse.extract_tags(txt, topK=100, withWeight=True)

c = (
    WordCloud()
    .add(series_name="热点分析", data_pair=tf, word_size_range=[6, 100])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="热点分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
    #.render("basic_wordcloud.html")
)

c.render_notebook()


# ## Timeline

# In[136]:


# vis
from pyecharts import options as opts
from pyecharts.charts import Bar, Timeline
from pyecharts.faker import Faker

x = Faker.choose()
tl = Timeline()
for i in range(2015, 2020):
    bar = (
        Bar()
        .add_xaxis(x)
        .add_yaxis("商家A", Faker.values())
        .add_yaxis("商家B", Faker.values())
        .set_global_opts(title_opts=opts.TitleOpts("某商店{}年营业额".format(i)))
    )
    tl.add(bar, "{}年".format(i))
tl.render_notebook()


# ## Grid

# In[134]:


# vis
from pyecharts import options as opts
from pyecharts.charts import Grid, Line, Scatter
from pyecharts.faker import Faker

scatter = (
    Scatter()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(
        #title_opts=opts.TitleOpts(title="Grid-Scatter"),
        #legend_opts=opts.LegendOpts(pos_left="20%"),
    )
)
line = (
    Line()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(
        #title_opts=opts.TitleOpts(title="Grid-Line", pos_right="5%"),
        #legend_opts=opts.LegendOpts(pos_right="20%"),
    )
)

scatter2 = (
    Scatter()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(
        #title_opts=opts.TitleOpts(title="Grid-Scatter"),
        #legend_opts=opts.LegendOpts(pos_left="20%"),
    )
)
line2 = (
    Line()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(
        #title_opts=opts.TitleOpts(title="Grid-Line", pos_right="5%"),
        #legend_opts=opts.LegendOpts(pos_right="20%"),
    )
)
grid = (
    Grid()
    .add(scatter, grid_opts=opts.GridOpts(pos_bottom="60%",pos_left="60%"))
    .add(line, grid_opts=opts.GridOpts(pos_bottom="60%",pos_right="60%"))
    .add(line2, grid_opts=opts.GridOpts(pos_top="60%",pos_left="60%"))
    .add(scatter2, grid_opts=opts.GridOpts(pos_top="60%",pos_right="60%"))
    #.render("grid_horizontal.html")
)
grid.render_notebook() 


# ## Overlap

# In[135]:


# vis
from pyecharts import options as opts
from pyecharts.charts import Bar, Line
from pyecharts.faker import Faker

v1 = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3]
v2 = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
v3 = [2.0, 2.2, 3.3, 4.5, 6.3, 10.2, 20.3, 23.4, 23.0, 16.5, 12.0, 6.2]


bar = (
    Bar()
    .add_xaxis(Faker.months)
    .add_yaxis("蒸发量", v1)
    .add_yaxis("降水量", v2)
    .extend_axis(
        yaxis=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(formatter="{value} °C"), interval=5
        )
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Overlap-bar+line"),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value} ml")),
    )
)

line = Line().add_xaxis(Faker.months).add_yaxis("平均温度", v3, yaxis_index=1)
bar.overlap(line)
bar.render_notebook()


# ## Save html files

# In[96]:


grid.render(path = 'grid.html')
#help(grid.render)


# ![image.png](images/end.png)
