# %%
%load_ext autoreload
%autoreload 2

import iml  # helper functions and classes (see iml/)
import numpy as np  # array processing
import pandas as pd  # dataframes, reading files
import matplotlib.pyplot as plt  # making plots

# %% [markdown]
# **Group members:** [your names]

# %% [markdown]
# ## Part 1: Partial Derivatives

# %% [markdown]
# ### (a)
# > Compute the derivative $\partial L / \partial x(i).$ Leaving all other
# > variables fixed, what value of $x(i)$ minimizes $L$?

# %% [markdown]
# ### (b)
# > Compute the derivative $\partial L / \partial \mu.$ If $\lambda = 0,$ what
# > values of $\mu$ minimize $L$?

# %% [markdown]
# ### (c)
# > Compute the derivative $\partial L / \partial \theta.$

# %% [markdown]
# ### (d)
# > Compute $$
# > \frac{\partial}{\partial z_i} \ln \operatorname{softmax}(z)_j.
# > $$
# > (Hint: first define $Z = \ln \sum_{j = 1}^k e^{z_j}$ and compute
# > $\partial Z / \partial z_j.$)


# %% [markdown]
# ## Part 2: Linear Regression

# %% [markdown]
# ### (a)
# > Consider the linear model, defined by parameters $(\theta_0, \theta_1)$,
# > which predicts $$
# > \hat y(i) = \theta_0 + \theta_1 x(i)
# > $$ for each index
# > $i.$ Express the gradient $\nabla L$ of mean squared error as a function of
# > the parameter vector $\theta$ and the datapoints $(x(i), y(i)).$

# %% [markdown]
# ### (b)
# > Extend the `SimpleRegressor` class from `iml/simple_regressor.py`,
# > implementing the method `fit(self, x, y, lr, steps)` to implement gradient
# > descent on mean squared error. Run `fit()` on the _unnormalized_ data
# > vectors `x` and `y`, and show the training process over 20 iterations with
# > `SimpleRegressor.show_training()`. Find a learning rate that makes the loss
# > decrease.

# %% [markdown]
# ### (c)
# > Now, normalize your data in some way and run `fit()` and `show_training()`
# > again, ensuring that you converge to an acceptable model in 20 iterations.


# %% [markdown]
# ## Part 3: k-Means and PCA

# %% [markdown]
# ### (a)
# > Suppose we didn't have access to the species attribute of the iris dataset.
# > Implement the Lloyd-Forgy algorithm for $k$-means and use it to infer a
# > partition of the iris dataset into three species. Compute the confusion
# > matrix and the accuracy of the partition relative to the true species.

# %% [markdown]
# ### (b)
# > Investigate `data/ulu.csv` using PCA and explain how this dataset was
# > produced. Using scikit-learn, run both $k$-means clustering and DBSCAN on
# > this dataset. Which method produces more reasonable "clusters"?


