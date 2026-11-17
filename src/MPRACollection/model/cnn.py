"""Residual CNN for predicting MPRA readouts from a one-hot encoded sequence."""

from torch import nn


class ResBlock(nn.Module):
    """x + ReLU(Conv(BatchNorm(x))), with length-preserving padding."""

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.block(x)


class ResCNN(nn.Module):
    """Convolution, residual blocks, max pooling, then two fully connected layers.

    Input (batch, 4, seq_len); output (batch, n_out), one column per readout.
    """

    def __init__(
        self,
        seq_len: int,
        n_out: int = 1,
        channels: int = 256,
        n_blocks: int = 4,
        kernel_size_head: int = 13,
        padding_head: int = 6,
        kernel_size: int = 3,
        padding: int = 1,
        pool: int = 4,
        fc_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size_head, padding=padding_head),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels, kernel_size, padding) for _ in range(n_blocks)])
        self.pool = nn.MaxPool1d(pool)
        len_head = seq_len + 2 * padding_head - kernel_size_head + 1
        n_flat = channels * (len_head // pool)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, n_out),
        )

    def forward(self, x):
        """x: (batch, 4, seq_len) -> (batch, n_out)"""
        x = self.head(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.fc(x)
