from __future__ import annotations

from typing import *

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from torch import Tensor
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sig_min = 0.001
eps = 1e-5


class CLFM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.n_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        return self.layers[-1](x)


def get_mnist_data():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    data = mnist["data"]
    target = mnist["target"]
    return data, target


def get_flow_matching_path_mu(targets: Tensor, noise: Tensor, time_steps: Tensor):
    mu = targets * time_steps + (1 - time_steps) * noise
    return mu


n_points = 10_000
test_size = 0.2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
input_dim = 784
output_dim = 784
hidden_dim = 1024
n_layers = 3
n_epochs = 100

clfm_model = CLFM(
    input_dim=input_dim + 1,
    output_dim=output_dim,
    hidden_dim=hidden_dim,
    n_layers=n_layers,
).to(device)


features, target = get_mnist_data()
# test train split sklean

x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=test_size,
)

x_train = x_train.astype("float32")
y_train = y_train.astype("int32")
x_test = x_test.astype("float32")
y_test = y_test.astype("int32")

train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


optimizer = torch.optim.Adam(clfm_model.parameters(), lr=1e-3)

sigma = 0.01
for _epoch in tqdm(range(n_epochs)):
    for batch_i, batch in enumerate(train_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        noise = torch.randn_like(x)
        t = torch.rand((x.shape[0], 1), device=device)

        mu = get_flow_matching_path_mu(targets=x, noise=noise, time_steps=t)

        # sample a random value of t from a uniform distribution for each batch element
        reparam_noise = torch.randn_like(mu)

        sample_x = mu + sigma * reparam_noise

        sample_x = torch.cat([sample_x, t], dim=1)
        # forward pass
        predictions = clfm_model(sample_x)
        loss = torch.mean((predictions - (x - noise)) ** 2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_i % 100 == 0:
            print(f"Epoch: {_epoch}, Loss: {loss.item()}")


# Sampling
n_samples = 10_000
with torch.no_grad():
    x_0 = torch.randn(n_samples, 2, device=device)
    x_1_hat = v_t.decode(x_0)


x_1_hat = x_1_hat.cpu().numpy()
plt.hist2d(x_1_hat[:, 0], x_1_hat[:, 1], bins=164)
plt.show()
