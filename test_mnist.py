import pytest
import torch

import mnist


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        values = torch.arange(4 * 28 * 28, dtype=torch.float32)
        self.images = values.view(4, 1, 28, 28) / values.max()
        self.labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def use_tiny_dataset(monkeypatch):
    def fake_mnist(
        root: str,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = True,
    ):
        return TinyDataset()

    monkeypatch.setattr(mnist.datasets, "MNIST", fake_mnist)
    torch.manual_seed(0)


@pytest.fixture
def weights_path(tmp_path):
    return tmp_path / "weights.pt"


def test_train_creates_checkpoint(monkeypatch, weights_path):
    use_tiny_dataset(monkeypatch)
    model = mnist.Model(state_path=weights_path, batch_size=2, download=False)

    model.train(epochs=1)

    assert weights_path.exists()


def test_classify_clamps_index(monkeypatch):
    use_tiny_dataset(monkeypatch)
    model = mnist.Model(batch_size=2, download=False)

    result = model.classify(10_000)

    assert result["index"] == model.get_max_test_index()
    assert result["image"].shape == (28, 28)


def test_load_returns_same_weights(monkeypatch, weights_path):
    use_tiny_dataset(monkeypatch)
    original = mnist.Model(state_path=weights_path, batch_size=2, download=False)
    original.train(epochs=1)
    original.save()

    loaded = mnist.Model.load(state_path=weights_path, batch_size=2)

    for name, tensor in original.network.state_dict().items():
        assert torch.equal(tensor, loaded.network.state_dict()[name])
