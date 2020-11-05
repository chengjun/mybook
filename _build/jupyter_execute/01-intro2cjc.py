#!/usr/bin/env python
# coding: utf-8

# 
# 
# # 第一章 计算传播学简介
# 
# 
# 
# 王成军 
# 
# wangchengjun@nju.edu.cn
# 
# 计算传播网 http://computational-communication.com
# 

# 
# 内容简介
# 
# - 一、引言：大数据时代
# - 二、如何认识世界
# - 三、科学的四重境界
# - 四、可计算性
# - 五、定义计算传播
# - 六、通往计算传播学之路
#     - 方法\工具\案例
# 
# 

# ## 一、引言：社会科学
# Feynman on social sciences
# 
# - Pseudo-science
# - Forms
# - Laws
# 

# ### 计算社会科学
# - Lazer et al (2009) Compuational social science. Science. V323. 6 Feb 2009
#     - 计算社会科学正在涌现
#     - 大规模的数据收集和数据分析
#     - 网络科学视角
#     - 揭示个体和群体行为的模式
# - D. Watts, A twenty-first century science. Nature 445, 489 (2007).
#     - 互联网大数据
#     - 网络科学视角
# 

# 产生背景
# - 网络科学（network science）
# - 计算语言学 （computational linguistics）
# - 数据科学（data science）
# 
# - 社会计算 （social computing）
# - 普适计算（ubiquitous computing）
# 
# - 可视化 （visualization）
# 
# - 数据新闻学 (data journalism)
# - 计算广告学 (computational advising)
# 

# ### 如何认识计算社会科学？
# 

# 计算社会科学社区的发展
# 
# <img src = './img/cc_rise.png' width = 500px>
# 
# 王成军（2015）计算传播学:作为计算社会科学的传播学.《中国网络传播研究》8.193-208

# ### 跨学科视野
# 
# 被引用数量前十名的期刊
# 
# 
# <img src = './img/cc_journal.png' width = 500px>
# 
# 王成军（2015）计算传播学:作为计算社会科学的传播学.《中国网络传播研究》8.193-208

# ### 研究脉络
# 被引用数量前十名的文献
# 
# <img src = './img/cc_pers.png' width = 500px>
# 
# 王成军（2015）计算传播学:作为计算社会科学的传播学.《中国网络传播研究》8.193-208

# ### 计算社会科学是科学吗？
# 
# 
# Geology is not a real science
# > Sheldon from the big bang theory
# 
# http://www.youtube.com/watch?v=sYMFHON8LFw
# 

# ## 二、如何认识世界？
# - 洞穴之喻
# - 开放思维
# - 康德：“我们所有的知识起源于感知，然后发展为理解，终结为理性。没有比理性更高的东西。”
#     - Immanuel Kant: All our knowledge begins with the senses, proceeds then to the understanding, and ends with reason. There is nothing higher than reason. 
# - Paul Erdos: My brain is open. 
# - Follow your logic
# 

# ### 常识? 
# - Everything is obvious: Once your know the answer
#     > Duncan J. Watts: 
# 
# - 当我们谈到社会科学时候，我们总觉得任何事情都是明显的，似乎采用常识（common sense）就可以回答绝大多数问题。
# - 硬科学与软科学
# 
# 

# Lazarsfeld, Paul F. 1949. “The American Soldier—An Expository Review.”Public Opinion Quarterly 13 (3):377—404.
# 
# 美国士兵研究起源于二战，美国陆军部对参战的60万士兵进行了调查，毫无疑问这一定花费了很多纳税人的钱。拉杂斯菲尔德“声称”他们发现来自农村的士兵更快乐。这是为什么呢？
# 

# 事实上，拉杂斯菲尔德告诉读者，他们的发现恰好与之相反：来自城市的士兵更快乐！
# 这又是为什么呢？

# 所以这个发现也是显而易见的！于是问题就来了：当每个答案以及与它们的对立的答案从常识的角度都看上去如此明显，那么我们对于“显而易见”（obviousness)的理解肯定出了问题。
# 

# ### 理论的三个比喻
# - 理论即逻辑的组合。
#     - 网络、望远镜、地图

# ### 理论的沙漏模型
# 
# 学术论文的结构: 以小见大\问题驱动\理论驱动\兴趣驱动\研究设计
# 
# <img src = './img/sandglass.png' width = 500px>

# ### 理论是一棵树
# 
# <img src='./img/theorytree.jpg' width = 500px>

# ## 三、科学的四重境界
# 科学的金字塔
# 
# - 数据
# - 模式、定律
# - 机制
# - 原则
# 

# ### 引力研究为例
# 托勒密：地球处于宇宙中心
# 
# - 引力第一重境界
#     - 哥白尼、弟谷
# - 引力第二重境界
#     - 开普勒
# - 引力第三重境界
#     - 牛顿
# - 引力的第四重境界
#     - 爱因斯坦
# 
# Richard Feynman, which is titled The Character of Physical Law – Part 1 The Law of Gravitation http://v.youku.com/v_show/id_XNzc4Mjk1NjA=.html
# 
# 

# 开普勒定律
# 
# - ①椭圆定律所有行星绕太阳的轨道都是椭圆，太阳在椭圆的一个焦点上。
# - ②面积定律行星和太阳的连线在相等的时间间隔内扫过相等的面积。
# - ③调和定律所有行星绕太阳一周的恒星时间T的平方与它们轨道长半轴A的立方成比例
# 
# 

# 牛顿运动定律
# 
# 由艾萨克·牛顿在1687年于《自然哲学的数学原理》一书中总结提出。
# - 第一定律说明了力的含义：力是改变物体运动状态的原因；
# - 第二定律指出了力的作用效果：力使物体获得加速度；
# - 第三定律揭示出力的本质：力是物体间的相互作用。[2] 
# 
# 

# 伽利略的相对性原理
# 
# - 一切彼此做匀速直线运动的惯性系，对于描写机械运动的力学规律来说是完全等价的。并不存在一个比其它惯性系更为优越的惯性系。
# - 在一个惯性系内部所作的任何力学实验都不能够确定这一惯性系本身是在静止状态，还是在作匀速直线运动。
# 

# 等待牛顿
# 
# - 太阳系行星的椭圆轨道
# - 彗星的抛物线轨道
# - 地球上的抛物线运动

# 传播学在哪里？
# 
# - Claude Shannon 
# 
# - Paul Felix Lazarsfeld 
# - Kurt Zadek Lewin 
# - Harold Dwight Lasswell 
# - Carl Iver Hovland 
# 
#   
# - Everett Rogers  
# 
# - Maxwell McCombs
# - Elihu Katz
# 
# - Elisabeth Noelle-Neumann 
# - Jürgen Habermas 
# 
# - George Gerbner
# - Wilbur Lang Schramm 
# - Walter Lippmann
# - Herbert Marshall McLuhan
# - Theodor W. Adorno
# 

# ## 四、可计算性
# 
# - 关注事物本身可以被计算的程度
# > Computability is the ability to solve a problem in an effective manner。
# The computability of a problem is closely linked to the existence of an algorithm to solve the problem.
# - 算法的可计算函数
#     - 图灵停机：你能用编程语言写出来并运行的都是可计算函数
# 

# ### 可计算化(Computational)
# - 关注事物本身可以被计算的方式
# - 计算思维（computational thinking）
#     - 问题抽象、任务的分解、自动化实现。
#         - Analyzing and logically organizing data
#         - Data modeling, data abstractions, and simulations
#         - Formulating problems such that computers may assist
#         - Identifying, testing, and implementing possible solutions
#         - Automating solutions via algorithmic thinking
#         - Generalizing and applying this process to other problems
# 
# 

# 传播学可计算化的基础存在吗？是什么？
# 

# 可计算性与科学研究
# 
# <img src = './img/comcom.png' width = 500px >

# 他山之石：网络科学
# 
# - We live life in the network
# > Lazer et al (2009) Compuational social science. Science. V323. 6 Feb 2009
# 
# - Complex networks have been studied extensively owing to their relevance to many real systems such as the world-wide web, the Internet, energy landscapes and biological and social networks. Song et al (2005)
# 
# - Network science supplies a mathematical structure of the social phenomenon. 

# ## 五、定义计算传播学
# 
# 计算传播学（computational communication research）是可计算社会科学（computational social science）的重要分支。
# - 主要关注人类传播行为的可计算性基础。
# - 以传播网络分析、传播文本挖掘、数学建模等为主要分析工具
# - （以非介入地方式）大规模地收集并分析人类传播行为数据
# - 挖掘人类传播行为背后的模式和法则
# - 分析模式背后的生成机制与基本原理
# - 可以被广泛地应用于数据新闻和计算广告等场景
# - 注重编程训练、数学建模、可计算思维
# 

# 计算传播社区
# 
# - 计算传播网
# http://computational-communication.com/
# - 计算传播学豆瓣小站
# http://site.douban.com/146782/
# - 计算传播学邮件组
# https://groups.google.com/group/computational-communication
# - 计算传播学实验中心 http://cc.nju.eud.cn
# 
# 

# Meme背后的社区
# 
# - Meme为什么能持续流行？
#     - 社区驱动 + 解决问题
# - Big data和machine learning：互联网公司
#     - 特征工程
# - Open science： 学术期刊、学会和大学
#     - 出版流程
# - Data journalism：媒体、新闻从业者、程序员
#     - 可视化需求
# - Network science：网络研究者、社交网站
#     - 复杂网络研究
# 

# 我们的愿景
# 
# - 寻找人类传播行为可计算化的基因。
# - 基因是生物学飞跃的原因，货币是经济学发展的关键。人类传播行为所隐藏的计算化“基因”是什么？
# - 计算传播学致力于寻找传播学可计算化的基因、学习和传播可计算化思维/方法（电子化数据收集能力、编程能力、数学建模能力、网络分析、文本挖掘）、了解和训练计算传播学的社会化应用方法（数据新闻、计算广告、可视化等）。
# 

# ## 六、通往计算传播学之路
# - 方法：从数据到模型
#     - 开放数据 (open data)
#         - 实证数据
#         - 工具
#             - 开源（open source）
#             - R和Python
#     - 开放科学（open science）
#         - 数值模拟
#         - 多主体建模
#         - 分析模型
#         - 计算模型
# 

# 书籍
# - Big Data
# - <del>Doing data science<del>
# - Beginning Python
# - Networks, crowds, and Markets

# 大数据：数字化“指纹”
# - Behavioral Data
# - Relational Data
# - Longitudinal 
# - Big Data
# - Digital Data

# D. Watts, A twenty-first century science. Nature 445, 489 (2007).
# > If handled appropriately, data about Internet-based communication and interactivity could revolutionize our understanding of collective human behaviour.
# 
# 很少有人会认为社会科学会成为21世纪科学的中心
# - 因为社会现象是最难解决的科学问题之一
# - 社会现象当中卷入了海量的异质性的个体之间的互动
# 
# 网站记录（Website Logs）与基于互联网的实验（Web-based experiments）
#     - 互联网公司与研究者的合作
# 

# 学科基础
# - 物理学
# - 数学
# - 计算机科学
# - 数据科学
# - 计算语言学
# - 网络科学

# ### 模式或法则：异速增长定律
# 
# <img src = './img/allowmetric.png' width = 500px>
# 
# http://www.nature.com/scitable/knowledge/library/allometry-the-study-of-biological-scaling-13228439
# 

# <img src = './img/allowmetric2.png' width = 500px >

# <img src = './img/allowmetric3.png' widht = 500px>
# 
# Wu & Zhang (2011) Accelerating growth and size-dependent distribution of human online activities. PhysRevE.84.026113
# 

# ### 在线社交网络
# 
# 选举行为可以通过社交网络传染
# 
# <img src = './img/sns.png' width = 500px>
# 
# Robert M. Bond et al. A 61-million-person experiment in social influence and political mobilization. Nature. 2012
# 

# ### 手机通话网络
# 传播的多样性制约社会经济的发展?
# 
# <img src = './img/macy.png' width = 500px>
# 
# Nathan Eagle, Michael Macy and Rob Claxton: Network Diversity and Economic Development, Science 328, 1029–1031, 2010.
# 

# ### 情感分析
# 
# <img src = './img/miller.png' width = 500px>
# 
# Miller (2011) Social scientists wade into the tweet stream. Science
# 

# ### 预测股票市场？
# Emotion: Calm\Alert\Sure\Vital\Kind\Happy
# 
# <img src = './img/bollen.png' width = 300px>
# 
# Bollen (2011) Twitter mood predicts the stock market. JOCS
# 

# ### The Twitter Political Index
# 
# <img src = './img/twitter.png' width  = 500px>
# 
# Figures source: election.twitter.com

# ### Google Flu Trends
# 使用搜索引擎预测流感
# 
# <img src = './img/googleflu.png' width = 500px>
# 
# Ginsberg et al. Detecting influenza epidemics using search engine query data. Nature 457, 1012-1014 (19 February 2009)
# 
# 
# http://www.google.com/trends/correlate/comic
# 

# <img src = './img/cdc.png' width = 500px>
# 
# “Nature reported that Google flu trends (GFT) was predicting more than double the proportion of doctor visits for influenza-like illness (ILI) than the Centers for Disease Control and Prevention (CDC), which bases its estimates on surveillance reports from laboratories across the United States (1, 2).”
# 
# Lazer et al. (2014) The parable of Google Flu Traps in big data analysis. Science
# 
# 

# ### 理论的最高标准
# 
# > Per Bak:“It puzzles me that geophysicists show little interest in underlying principles of their science. Perhaps they take it for granted that the earth is so complicated and messy that no general principles apply”. 
# 
# - How Nature Works? 
# 

# ### Hack定律
# 
# Hack定律指出，在河流网络中，支流的长度（stream length）L和相对应的蓄水盆地面积（basin area）A之间存在如下标度关系：$L = A^h$
# 
# 其中h的数值在大多数水系的实证数据中都被测为0.6左右。
# 
# 

# Before God we are all equally wise and equally foolish. Do not worry about your difficulties in Mathematics. I can assure you mine are still greater.
# ——Albert Einstein 
# 

# 不管时代的潮流和社会的风尚怎样，人总可以凭着自己高尚的品质，超脱时代和社会，走自己正确的道路。现在，大家都为了电冰箱、汽车、房子而奔波、追逐、竞争。这就是我们这个时代的特征了。但是也还有不少人，他们不追求这些物质的东西，他们追求理想和真理，得到了内心的自由和安宁。 —- 爱因斯坦

# ### 阅读文献
# - Adam Mann (2016) Core Concepts: Computational social science. 113: 468–470, doi: 10.1073/pnas.1524881113 http://www.pnas.org/content/113/3/468.full
# - Watts, D. J. (2007). A twenty-first century science. Nature, 445(7127), 489-489. 
# - Lazer, D., Pentland, A. S., Adamic, L., Aral, S., Barabasi, A. L., Brewer, D., ... & Van Alstyne, M. (2009). Computational social science. Science (New York, NY), 323(5915), 721. ↩
# - Cioffi‐Revilla, C. (2010). Computational social science. Wiley Interdisciplinary Reviews: Computational Statistics, 2(3), 259-271. 
# - Strohmaier, M., & Wagner, C. (2014). Computational Social Science for the World Wide Web. IEEE Intelligent Systems, (5), 84-88. ↩
# - Conte, Rosaria, Nigel Gilbert, Guilia Bonelli, Claudio Cioffi-Revilla, Guillaume Deffuant, Janos Kertesz, Vittorio Loreto et al. "Manifesto of computational social science." The European Physical Journal Special Topics 214, no. 1 (2012): 325-346.
# - Watts, D. J. (2007). A twenty-first century science. Nature, 445(7127), 489-489.
# - Duncan J. Watts 2011 Everything Is Obvious：Once You Know the Answer. Crown Business. 2011-3-29
# - 祝建华， 彭泰权， 梁海， 王成军， 秦洁， 陈鹤鑫 (2014) 计算社会科学在新闻传播研究中的应用 科研信息化技术与应用 5 (2), 3-13

# This is the End.
# > Thank you for your attention!

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# 02-bigdata
# ```
# 
