from torch import nn, Tensor


class DataProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.data_projection_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.data_projection_y = nn.Linear(1, self.hidden_dim)
        self.data_projection_t = nn.Linear(1, self.hidden_dim)

    def forward(self, xs: Tensor, ys: Tensor, ts: Tensor) -> Tensor:
        # Change the data type to float
        xs = xs.float()  # batch_size, sequence_length, features
        ys = ys.float()  # batch_size, sequence_length
        ts = ts.float()  # batch_size

        # ts should be repeated to match the sequence length of xs
        ts = ts.unsqueeze(-1).unsqueeze(-1)
        ts = ts.repeat(1, xs.shape[1], 1)

        # Ensure ys has the shape (batch_size, sequence_length, 1)
        ys = ys.unsqueeze(-1)

        # Project the input tensors
        xs = self.data_projection_x(xs)
        ts = self.data_projection_t(ts)
        ys = self.data_projection_y(ys)

        # Return the combined tensor
        return xs + ys + ts
