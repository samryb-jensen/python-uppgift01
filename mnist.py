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
# ## MNIST utilities
# Torch- and torchvision-based helpers for training, evaluating, and using the
# convolutional neural network that powers the MNIST CLI app.

# %% [markdown]
# ### Library imports
# Core PyTorch building blocks plus torchvision datasets/transforms used below.

# %%
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# %% [markdown]
# ### CNN backbone
# Feature extractor invoked by the higher-level `Model` wrapper.


# %%
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


# %%
MODEL_STATE_PATH = Path("mnist_cnn.pt")


# %% [markdown]
# ### High-level model API
# Handles dataset setup, training loops, evaluation, checkpointing, and inference.


# %%
class Model:
    def __init__(
        self,
        state_path: Path | str = MODEL_STATE_PATH,
        batch_size: int = 100,
        data_root: str = "data",
        download: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_path = Path(state_path)
        self.batch_size = batch_size
        self.data_root = data_root

        transform = ToTensor()
        self.train_data = datasets.MNIST(
            root=data_root, train=True, transform=transform, download=download
        )
        self.test_data = datasets.MNIST(
            root=data_root, train=False, transform=transform, download=download
        )

        self.loaders = {
            "train": DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True, num_workers=1
            ),
            "test": DataLoader(
                self.test_data, batch_size=batch_size, shuffle=True, num_workers=1
            ),
        }

        self.network = CNN().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

    @classmethod
    def load(cls, state_path: Path | str = MODEL_STATE_PATH, **kwargs) -> "Model":
        model = cls(state_path=state_path, download=False, **kwargs)
        model.load_weights(state_path)
        return model

    def load_weights(self, state_path: Path | str | None = None) -> None:
        path = Path(state_path) if state_path else self.state_path
        if not path.exists():
            raise FileNotFoundError(
                f"No trained weights found at {path}. Run training first."
            )
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.to(self.device)
        print(f"Loaded trained weights from {path}")

    def save(self, state_path: Path | str | None = None) -> None:
        path = Path(state_path) if state_path else self.state_path
        torch.save(self.network.state_dict(), path)
        print(f"Saved trained weights to {path}")

    def train(self, epochs: int = 10, save: bool = True) -> None:
        print(f"Model is currently being trained using {str(self.device).upper()}")
        for epoch in range(1, epochs + 1):
            self._train_single_epoch(epoch)
            self.test()

        if save:
            self.save()

    def _train_single_epoch(self, epoch: int) -> None:
        self.network.train()
        for batch_idx, (data, target) in enumerate(self.loaders["train"]):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                seen = batch_idx * len(data)
                total = len(self.loaders["train"].dataset)
                pct = 100.0 * batch_idx / len(self.loaders["train"])
                print(
                    f"Train Epoch: {epoch} [{seen}/{total} ({pct:.0f}%)]\t{loss.item():.6}"
                )

    def test(self) -> dict[str, float | int]:
        self.network.eval()
        test_loss = 0.0
        correct = 0
        total = len(self.loaders["test"].dataset)

        with torch.no_grad():
            for data, target in self.loaders["test"]:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                test_loss += self.loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = test_loss / total
        accuracy = 100.0 * correct / total
        print(
            f"\nTest set: Average loss: {avg_loss:.4f}, Accuracy {correct}/{total} ({accuracy:.0f}%)\n"
        )
        return {
            "loss": avg_loss,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }

    def get_max_test_index(self) -> int:
        return len(self.test_data) - 1

    def _get_test_sample(self, sample_index: int):
        clamped_index = max(0, min(sample_index, self.get_max_test_index()))
        data, target = self.test_data[clamped_index]
        return clamped_index, data, target

    def classify(self, sample_index: int) -> dict[str, object]:
        self.network.eval()
        clamped_index, data, target = self._get_test_sample(sample_index)
        batch = data.unsqueeze(0).to(self.device)
        output = self.network(batch)
        prediction = output.argmax(dim=1, keepdim=True).item()
        return {
            "index": clamped_index,
            "prediction": prediction,
            "label": int(target),
            "image": data.squeeze(0).cpu().numpy(),
        }


# %%
