import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(
            self,
            n_dim: int,
            vocab_size: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_dim = n_dim

        self.word_embeddings = nn.Embedding(self.vocab_size, self.n_dim)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index: index + 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(x)


class SinePositionalEmbedding(nn.Module):
    def __init__(
            self,
            n_dim: int,
            dropout: float = 0.0,
            alpha: bool = True,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.x_scale = 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encodings = None
        self.extend_positional_encodings(torch.tensor(0.0).expand(1, 4000))

    def extend_positional_encodings(self, x):
        """Reset the positional encodings."""
        if self.positional_encodings is not None:
            if self.positional_encodings.size(1) >= x.size(1):
                if self.positional_encodings.dtype != x.dtype or self.positional_encodings.device != x.device:
                    self.positional_encodings = self.positional_encodings.to(dtype=x.dtype, device=x.device)
                return
        positional_encodings = torch.zeros(x.size(1), self.n_dim)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.n_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.n_dim)
        )
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        positional_encodings = positional_encodings.unsqueeze(0)
        self.positional_encodings = positional_encodings.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.extend_positional_encodings(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.positional_encodings[:, : x.size(1)]
        return self.dropout(output)

