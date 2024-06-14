import torch
import torch.nn as nn
from torch.nn import functional as F
from LLM import Dataset, GPTLanguageModel, estimate_loss

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

print("Loading dataset")
dataset = Dataset("machado")
dataset.train_val_split(dataset.data)
print("Dataset loaded")

print("Creating model")
model = GPTLanguageModel(
    n_embd,
    dataset,
    n_head,
    n_layer,
    block_size,
    dropout,
)
m = model.to(device)
# print the number of parameters in the model
print(
    "Model created with", sum(p.numel() for p in m.parameters()) / 1e6, "M parameters"
)


# ---------------------------------------------------------------
# Optimazing the model
print("Optimizing model")
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, eval_iters, batch_size)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = dataset.get_batch("train", block_size, batch_size, device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print("Optimization finished")
# --------------------------------------------------------------
# Generating text
print("Generating text:\n\n")
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(dataset.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open("output.txt", "w").write(
    dataset.decode(m.generate(context, max_new_tokens=10000)[0].tolist())
)
