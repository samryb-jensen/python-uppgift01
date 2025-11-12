# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: python-uppgift01
#     language: python
#     name: python-uppgift01
# ---

# %%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
my_series = [5, 9, 12, 27]
table = pd.Series(my_series)
table

# %%
table2 = pd.Series(my_series, ["A", "B", "C", "D"])
table2

# %%
cars = {"Tesla": 12, "Porche": 4, "Ferrari": 1}
table3 = pd.Series(cars)
table3

# %%
