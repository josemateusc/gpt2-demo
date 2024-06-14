import nltk
import re
import torch

# Import the corpus
import nltk.corpus


class Dataset:

    # Getting the dataset
    @staticmethod
    def get_dataset(dataset="machado"):
        nltk.download(dataset, quiet=True)

        def limpa(raw):
            return [
                re.sub(r"\W+\s*", " ", i).lstrip()
                for i in raw.lower().split(".")
                if i != ""
            ]

        s = nltk.corpus.machado.readme()
        livros = re.findall(r"(\w+\/\w+\.txt):", s)
        corpus = []
        for t in livros:
            raw = nltk.corpus.machado.raw(t)
            corpus += limpa(raw)

        corpus = " ".join(corpus)
        return corpus

    def __init__(self, data="machado"):
        self.corpus = self.get_dataset(data)
        self.chars = sorted(list(set(self.corpus)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [
            self.stoi[c] for c in s
        ]  # encoder: take a string, output a list of integers
        self.decode = lambda l: "".join(
            [self.itos[i] for i in l]
        )  # decoder: take a list of integers, output a string
        self.data = torch.tensor(self.encode(self.corpus), dtype=torch.long)
        self.train_data = None
        self.val_data = None
        self.train_val_split(self.data)

    def train_val_split(self, data, split=0.9):
        n = int(split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    # data loading
    def get_batch(self, split, block_size, batch_size, device):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
