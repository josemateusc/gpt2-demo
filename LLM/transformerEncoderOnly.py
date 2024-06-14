import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
    """one head of self-attention mechanism"""

    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(
            n_embd, head_size, bias=False
        )  # Linear layer for computing keys
        self.query = nn.Linear(
            n_embd, head_size, bias=False
        )  # Linear layer for computing queries
        self.value = nn.Linear(
            n_embd, head_size, bias=False
        )  # Linear layer for computing values
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # Registering a lower triangular matrix for masking future positions (triangular mask)

        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = (
            x.shape
        )  # B = batch size, T = number of time steps, C = number of channels
        k = self.key(x)  # Compute keys, result is of shape (B, T, head_size)
        q = self.query(x)  # Compute queries, result is of shape (B, T, head_size)

        # Compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # Matrix multiplication between queries and transposed keys, scaled by sqrt of head_size, result is of shape (B, T, T)
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # Apply mask to prevent attending to future positions, result is of shape (B, T, T)
        wei = F.softmax(
            wei, dim=-1
        )  # Apply softmax to get attention weights, result is of shape (B, T, T)
        wei = self.dropout(wei)  # Apply dropout for regularization

        # perform the weighted aggregation of the values
        v = self.value(x)  # Compute values, result is of shape (B, T, head_size)
        out = (
            wei @ v
        )  # Matrix multiplication between attention weights and values, result is of shape (B, T, head_size)
        # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        # Create a list of 'num_heads' Head instances
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        # Linear layer to project the concatenated output of all heads back to the embedding size
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the output from each head along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project the concatenated output and apply dropout
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        # Define a sequential network with:
        # 1. A linear layer that expands the embedding size by a factor of 4
        # 2. A ReLU activation function for non-linearity
        # 3. A linear layer that reduces the dimension back to the original embedding size
        # 4. A dropout layer for regularization
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand dimensions
            nn.ReLU(),  # Apply ReLU activation
            nn.Linear(4 * n_embd, n_embd),  # Reduce dimensions back to original
            nn.Dropout(dropout),  # Apply dropout for regularization
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head  # Define the size of each head
        # Multi-head self-attention module
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        # Feed-forward neural network
        self.ffwd = FeedFoward(
            n_embd
        )  # Layer normalization applied before self-attention and feed-forward networks
        self.ln1 = nn.LayerNorm(
            n_embd
        )  # Layer normalization for self-attention # a little different from the original paper
        self.ln2 = nn.LayerNorm(
            n_embd
        )  # Layer normalization for feed-foward # a little different from the original paper

    def forward(self, x):
        # Apply layer normalization followed by self-attention, and add the input (residual connection)
        x = x + self.sa(self.ln1(x))
        # Apply layer normalization followed by feed-forward network, and add the input (residual connection)
        x = x + self.ffwd(self.ln2(x))
        return x
