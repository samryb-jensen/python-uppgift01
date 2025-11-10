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

# %% [markdown]
# ## Import libraries for data loading, modeling, and visualization
# We bring in PyTorch for building neural networks, torchvision for the MNIST dataset and image transforms, and matplotlib so we can visualize digits later on.

# %%
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

# %% [markdown]
# ## Download the MNIST datasets and convert them to tensors
# MNIST is a collection of 28x28 grayscale images of handwritten digits (0–9). We download both the training and test splits and immediately convert each image into a PyTorch tensor so the model can work with them.

# %%
train_data = datasets.MNIST(
    root="data", train=True, transform=ToTensor(), download=True
)

test_data = datasets.MNIST(
    root="data", train=False, transform=ToTensor(), download=True
)

# %% [markdown]
# ## Double-check the tensor shape of the training set
# Inspecting the shape confirms how many training samples we have and ensures each image is 28x28 pixels with a single channel.

# %%
train_data.data.shape

# %% [markdown]
# ## Verify the tensor shape of the test set
# The test set should mirror the same structure so evaluation uses data identical to the training format.

# %%
test_data.data.shape

# %% [markdown]
# ## Inspect the label tensor to confirm class coverage
# Viewing the labels helps verify that all digits are present and that labels align with the image tensors.

# %%
train_data.targets

# %% [markdown]
# ## Build PyTorch data loaders for training and evaluation
# DataLoaders handle shuffling, batching, and parallel loading, which keeps the training loop clean and efficient.

# %%
loaders = {
    "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
}


# %% [markdown]
# ## Define the convolutional neural network used for MNIST classification
# This class builds a small CNN: two convolutional layers capture image patterns, followed by fully connected layers that map the extracted features to digit predictions.


# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


# %% [markdown]
# ## Configure device, instantiate the model, and define optimizer, loss, and loops
# We move everything to GPU if available, create the CNN, choose the Adam optimizer to adjust weights, and set cross-entropy as the loss to measure prediction quality against labels. The `train` helper runs one pass through the shuffled training batches, while `test` switches the model to evaluation mode and measures accuracy on unseen data without updating weights.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100.0 * batch_idx / len(loaders['train']):.0f}%)]\t{loss.item():.6}"
            )


def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders["test"].dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders['test'].dataset)} ({100.0 * correct / len(loaders['test'].dataset):.0f}%\n)"
    )


# %% [markdown]
# ## Train for multiple epochs and report accuracy after each epoch
# We loop through the dataset ten times, calling the training and testing helpers so we can watch accuracy improve (or diagnose issues) after every pass.

# %%
for epoch in range(1, 11):
    train(epoch)
    test()

# %% [markdown]
# ## Quickly confirm whether training ran on CPU or GPU
# Printing the `device` lets us confirm whether we benefitted from GPU acceleration or fell back to CPU.

# %%
print(device)

# %% [markdown]
# ## Run a sample inference and visualize the corresponding digit
# After training, we grab one test image, run it through the model to get a prediction, and display the digit so we can visually verify the result makes sense.

# %%
model.eval()

data, target = test_data[0]

data = data.unsqueeze(0).to(device)

output = model(data)

prediction = output.argmax(dim=1, keepdim=True).item()

print(f"Prediction: {prediction}")

image = data.squeeze(0).squeeze(0).cpu().numpy()

plt.imshow(image, cmap="gray")
plt.show()

# %%
