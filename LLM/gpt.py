import torch
from torch import nn
import torch.nn.functional as F
from LLM import Block


@torch.no_grad()
def estimate_loss(model, eval_iters, batch_size):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = model.dataset.get_batch(
                split, model.block_size, batch_size, model.device
            )
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        n_embd,
        dataset,
        n_head,
        n_layer,
        block_size,
        dropout,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.dataset = dataset
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(dataset.vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of Transformer blocks
        self.blocks = nn.Sequential(
            *[
                Block(
                    self.n_embd,
                    n_head=self.n_head,
                    block_size=self.block_size,
                    dropout=self.dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(
            n_embd, dataset.vocab_size
        )  # Linear layer to generate logits

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B = batch size, T = sequence length

        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # Token embeddings of shape (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # Position embeddings of shape (T, C)
        x = tok_emb + pos_emb  # Add token and position embeddings (B, T, C)
        x = self.blocks(x)  # Pass through Transformer blocks (B, T, C)
        x = self.ln_f(x)  # Final layer normalization (B, T, C)
        logits = self.lm_head(x)  # Linear layer to get logits (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Flatten logits (B*T, vocab_size)
            targets = targets.view(B * T)  # Flatten targets (B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
