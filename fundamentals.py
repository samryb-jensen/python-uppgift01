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

print(torch.__version__)

# %%
scalar = torch.tensor(7)
scalar

# %%
scalar.ndim

# %%
scalar.item()

# %%
vector = torch.tensor([7, 7])
vector

# %%
vector.ndim

# %%
vector.shape

# %%
MATRIX = torch.tensor([[7, 8], [9, 10]])
MATRIX

# %%
MATRIX.ndim

# %%
MATRIX[0]

# %%
MATRIX.shape

# %%
TENSOR = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])
TENSOR

# %%
TENSOR.ndim

# %%
TENSOR.shape

# %%
TENSOR[0]

# %%
random_tensor = torch.rand(3, 4)
random_tensor

# %%
random_tensor.ndim

# %%
random_image_size_tensor = torch.rand(size=(3, 224, 224))
random_image_size_tensor.shape, random_image_size_tensor.ndim

# %%
