#!/usr/bin/env python
# coding: utf-8

# # The Book of Why
# 
# 
# ![image.png](img/causal_model_book.png)
# 

# <img src=img/causal_model_mount.png align = left width= 900px>
# 
# 
# Scaling “Mount Intervention.” 
# 
# The most familiar methods to estimate the effect of an intervention, in the presence of confounders, are the back-door adjustment and instrumental variables. 
# 
# The method of front-door adjustment was unknown before the introduction of causal diagrams. 
# 
# The do-calculus, which my students have fully automated, makes it possible to tailor the adjustment method to any particular causal diagram. (Source: Drawing by Dakota Harr.)
# 
# 

# ## Mediator, Confounder, Collider
# 
# 中介、混杂、对撞
# 
# - (a) In a chain junction, $A \rightarrow B \rightarrow C$, controlling for B prevents information about A from getting to C or vice versa.
# - (b) In a fork or confounding junction, $A \leftarrow B \rightarrow C$, controlling for B prevents information about A from getting to C or vice versa.
# - (c) In a collider, $A \rightarrow B \leftarrow C$, exactly the opposite rules hold. The variables A and C start out independent, so that information about A tells you nothing about C. 
#     - But if you control for B, then information starts flowing through the “pipe,”  due to the explain-away effect.
# 

# (d) Controlling for descendants (or proxies) of a variable is like “partially” controlling for the variable itself. 
# - Controlling for a descendant of a mediator partly closes the pipe; 
# - Controlling for a descendant of a collider partly opens the pipe.
# 
# 
# $$
# A \leftarrow B \leftarrow C \rightarrow D \leftarrow E \rightarrow F \rightarrow G \leftarrow H \rightarrow I \rightarrow J
# $$
# 
# If a single junction is blocked, then J cannot “find out” about A through this path.
# 
# 
# “So we have many options to block communication between A and J: 
# 
# - control for B, control for C, don’t control for D (because it’s a collider), control for E, and so forth. 
# - Any one of these is sufficient. 
# 
# **This is why the usual statistical procedure of controlling for everything that we can measure is so misguided.**
# - In fact, this particular path is blocked if we don’t control for anything! 
# - The colliders at D and G block the path without any outside help. 
# - Controlling for D and G would open this path and enable J to listen to A.
# 
# 

# Finally, to deconfound two variables X and Y, we need only block every noncausal path between them without blocking any causal paths. 
# 
# A **back-door path** is any path from X to Y that starts with an arrow pointing into X. 
# -  it allows spurious correlation between X and Y
# - X and Y will be deconfounded if we block every back-door path.
# 
# If we do this by controlling for some set of variables Z, we also need to make sure that 
# - no member of Z is a descendant of X on a causal path; 
#     - otherwise we might partly or completely close off that path.

# Confounding was the primary obstacle that caused us to confuse seeing with doing. Having removed this obstacle with the tools of “path blocking” and the back-door criterion, we can now map the routes up **Mount Intervention** with systematic precision. 
# 
# For the novice climber, the safest routes up the mountain are 
# - **the back-door adjustment**
# - and its various cousins, 
#     - some going under the rubric of **front-door adjustment** 
#     - some under **instrumental variables**.
# 
# 

# ## THE SIMPLEST ROUTE: THE BACK-DOOR ADJUSTMENT FORMULA
# 
# 
# ### The back-door adjustment formula
# 
# $$
# P(Y|do(X)) = \sum_{u} P(Y|X, U = u)P(U = u) 
# $$
# 
# 
# If you are confident that you have data on a sufficient set of variables (called **deconfounders**) to block all the back-door paths between the intervention and the outcome.
# 
# - To do this, we measure the average causal effect of an intervention by first estimating its effect at each “level,” or stratum, of the deconfounder. 
# - We then compute a weighted average of those strata, where each stratum is weighted according to its prevalence in the population. 
# 
# 
# 
# 

# <img src=img/causal_model_simpson2.png align = right width = 500px>
# 
# For example, the deconfounder is gender, 
# we first estimate the causal effect for males and females. Then we average the two, if the population is (as usual) half male and half female. If the proportions are different—say, two-thirds male and one-third female—then to estimate the average causal effect we would take a correspondingly weighted average.
# 
# $$
# P(Y|do(X)) =  P(Y|X, U = female)P(U = female) + P(Y|X, U = male)P(U = male) 
# $$

# ## Front-door Adjustment
# 
# <img src=img/causal_model_frontdoor.png align = right width = 500px>
# 
# Suppose we are doing an observational study and have collected data on Smoking, Tar, and Cancer for each of the participants. Unfortunately, we cannot collect data on the Smoking Gene because we do not know whether such a gene exists. 
# - Lacking data on the confounding variable, we cannot block the back-door path $Smoking \leftarrow Smoking Gene \rightarrow Cancer$. 
# - Thus we cannot use **back-door adjustment** to control for the effect of the confounder.
# 
# 
# 
# http://bayes.cs.ucla.edu/WHY/errata-pages-PearlMackenzie_BookofWhy_Final.pdf

# Instead of going in the back door, we can go in the front door! 
# 
# - In this case, the front door is the direct causal path $Smoking \rightarrow Tar \rightarrow Cancer$, for which we do have
# data on all three variables.
# 
# First, we can estimate the average causal effect of Smoking on Tar
# - because there is no unblocked back-door path from Smoking to Tar, as the $Smoking \leftarrow Smoking \space Gene \rightarrow Cancer \leftarrow Tar$ path is already blocked by the *collider* at Cancer. 
#     - Because it is blocked already, we don’t even need back-door adjustment. 
# 
# We can simply observe $P(tar | smoking)$ and $P(tar | no \space smoking)$, and the difference between them will be the average causal effect of Smoking on Tar.

# Likewise, the diagram allows us to estimate the average causal effect of **Tar on Cancer**. 
# - To do this we can block the back-door path from Tar to Cancer, $Tar \leftarrow Smoking \leftarrow Smoking Gene \rightarrow  Cancer$, by adjusting for Smoking. 
# 
# We only need data on a sufficient set of deconfounders (i.e., Smoking). 
# 
# Then the back-door adjustment formula will give us $P(cancer | do(tar))$ and $P(cancer | do(no \space tar))$. 
# 
# $$
# P(cancer | do(tar)) = P(cancer | tar, smoking) P(smoking) +
# $$
# $$
# P(cancer | tar, no \space smoking)P(no \space smoking) 
# $$
# 
# 
# $$
# P(cancer | do(no \space tar)) = P(cancer | no \space tar, smoking) P(smoking) +
# $$
# $$
# P(cancer | no \space tar, no \space smoking)P(no \space smoking) 
# $$
# 
# 
# The difference between these is the average causal effect of Tar on Cancer.
# 
# 

# We can combine $P(cancer | do (tar))$ and $P(tar|do(smoking))$ to obtain the average increase in cancer due to smoking. Cancer can come about in two ways: in the presence of Tar or in the absence of Tar. 
# 
# - If we force a person to smoke, 
#     - then the probabilities of these two states are **P(tar | do(smoking))** and **P(no tar | do(no smoking))**, respectively. 
#     - If a Tar state evolves, the likelihood of causing Cancer is **P(cancer | do(tar))**. 
#     - If, on the other hand, a No-Tar state evolves, then it would result in a Cancer likelihood of **P(cancer | do(no tar))**. 
#     - We can weight the two scenarios by their respective probabilities under do(smoking) and in this way compute the total probability of cancer due to smoking. 
# - The same argument holds if we prevent a person from smoking, do(no smoking).
# - The difference between the two gives us the average causal effect on cancer of smoking versus not smoking.

# ### The  front-door adjustment formula
# 
# $$
# P(Y|do(X)) = \sum_{z} P(Z = z|X) \sum_{x} P(Y|X = x, Z = z)P(X = x)
# $$
# 
# Here X stands for Smoking, Y stands for Cancer, Z stands for Tar, and U (which is conspicuously absent from the formula) stands for the unobservable variable, the Smoking Gene.
# 
# $$
# P(cancer|do(smoking)) = P(tar|smoking) P(cancer|smoking, tar)P(smoking) + 
# $$
# 
# $$
#   P(tar|no \space smoking) P(cancer|no \space smoking, tar)P(no \space smoking) + 
# $$
# 
# $$
#   P(no \space tar|smoking) P(cancer|smoking, no \space tar)P(smoking) + 
# $$
# 
# 
# $$
#   P(no \space tar|no \space smoking) P(cancer|no \space smoking,no \space tar)P(no \space smoking)  
# $$
# 
# 
# 
# 
# 

# ![image.png](img/causal_model_error.png)

# ![image.png](img/causal_model_do.png)

# ## Instrumental Variable
# 
# ![image.png](img/causal_model_iv.png)

# # DoWhy: Different estimation methods for causal inference
# 
# https://microsoft.github.io/dowhy/example_notebooks/dowhy_estimation_methods.html#Method-1:-Regression
# 
# This is a quick introduction to the DoWhy causal inference library.
# We will load in a sample dataset and use different methods for estimating the causal effect of a (pre-specified)treatment variable on a (pre-specified) outcome variable.
# 
# We will see that not all estimators return the correct effect for this dataset.
# 
# First, let us add the required path for Python to find the DoWhy code and load all required packages

# In[3]:


pip install dowhy


# In[4]:


import numpy as np
import pandas as pd
import logging

import dowhy
from dowhy import CausalModel
import dowhy.datasets 


# Now, let us load a dataset. For simplicity, we simulate a dataset with linear relationships between common causes and treatment, and common causes and outcome. 
# 
# Beta is the true causal effect. 

# In[5]:


data = dowhy.datasets.linear_dataset(beta=10,
        num_common_causes=5, 
        num_instruments = 2,
        num_treatments=1,
        num_samples=10000,
        treatment_is_binary=True,
        outcome_is_binary=False)
df = data["df"]
df


# In[17]:


df.columns


# Note that we are using a pandas dataframe to load the data.

# ## Simple Multiple Regression

# In[18]:


import statsmodels.api as sm
import statsmodels.formula.api as smf 


# In[21]:


results = smf.ols('y ~ v0+ Z0 + Z1 + W0+W1 + W2 + W3+W4', data=df).fit()

results.summary()


# ## Identifying the causal estimand

# We now input a causal graph in the DOT graph format.

# In[6]:


# With graph
model=CausalModel(
        data = df,
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"],
        instruments=data["instrument_names"]
        )


# In[7]:


model.view_model()


# In[9]:


from IPython.display import Image, display
display(Image(filename="img/causal_model.png"))


# We get a causal graph. Now identification and estimation is done. 

# In[10]:


identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)


# ## Method 1: Regression
# 
# Use linear regression.

# In[11]:


causal_estimate_reg = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True)
print(causal_estimate_reg)
print("Causal Estimate is " + str(causal_estimate_reg.value))


# ## Method 2: Stratification
# 
# We will be using propensity scores to stratify units in the data.

# In[12]:


causal_estimate_strat = model.estimate_effect(identified_estimand,
                                              method_name="backdoor.propensity_score_stratification",
                                              target_units="att")
print(causal_estimate_strat)
print("Causal Estimate is " + str(causal_estimate_strat.value))


# ## Method 3: Matching
# 
# We will be using propensity scores to match units in the data.

# In[13]:


causal_estimate_match = model.estimate_effect(identified_estimand,
                                              method_name="backdoor.propensity_score_matching",
                                              target_units="atc")
print(causal_estimate_match)
print("Causal Estimate is " + str(causal_estimate_match.value))


# ## Method 4: Weighting
# 
# We will be using (inverse) propensity scores to assign weights to units in the data. DoWhy supports a few different weighting schemes:
# 1. Vanilla Inverse Propensity Score weighting (IPS) (weighting_scheme="ips_weight")
# 2. Self-normalized IPS weighting (also known as the Hajek estimator) (weighting_scheme="ips_normalized_weight")
# 3. Stabilized IPS weighting (weighting_scheme = "ips_stabilized_weight")

# In[14]:


causal_estimate_ipw = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.propensity_score_weighting",
                                            target_units = "ate",
                                            method_params={"weighting_scheme":"ips_weight"})
print(causal_estimate_ipw)
print("Causal Estimate is " + str(causal_estimate_ipw.value))


# ## Method 5: Instrumental Variable
# 
# We will be using the Wald estimator for the provided instrumental variable.

# In[15]:


causal_estimate_iv = model.estimate_effect(identified_estimand,
        method_name="iv.instrumental_variable", method_params = {'iv_instrument_name': 'Z0'})
print(causal_estimate_iv)
print("Causal Estimate is " + str(causal_estimate_iv.value))


# ## Method 6: Regression Discontinuity
# 
# We will be internally converting this to an equivalent instrumental variables problem.

# In[16]:


causal_estimate_regdist = model.estimate_effect(identified_estimand,
        method_name="iv.regression_discontinuity", 
        method_params={'rd_variable_name':'Z1',
                       'rd_threshold_value':0.5,
                       'rd_bandwidth': 0.1})
print(causal_estimate_regdist)
print("Causal Estimate is " + str(causal_estimate_regdist.value))


# In[ ]:




