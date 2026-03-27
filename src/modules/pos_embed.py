from src.base.model_base import PositionalEmbedderBase
import torch
from ..utils.module_utils import isolate_torch_rng
import random
from typing import Literal
from torch import nn



class PositionalEmbedder(PositionalEmbedderBase):
    def __init__(
            self, 
            seed: int=None,
            feature_positional_embedding: (
            Literal[
                "normal_rand_vec",
                "uni_rand_vec",
                "learned",
                "subspace",
            ]
        ) = None,
        ninp: int = None,
    ):
        super().__init__()
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.feature_positional_embedding = feature_positional_embedding
        if feature_positional_embedding == "learned":
            assert ninp is not None, "ninp must be provided for learned positional embedding"
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            assert ninp is not None, "ninp must be provided for subspace positional embedding"
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)



    def embed(self, x: torch.Tensor):
        with isolate_torch_rng(self.seed, device=x.device):
            if self.feature_positional_embedding == "normal_rand_vec":
                embs = torch.randn(
                    (x.shape[-2], x.shape[-1]),
                    device=x.device,
                    dtype=x.dtype,
                )
                x += embs[None, None, None]
            elif self.feature_positional_embedding == "uni_rand_vec":
                embs = (
                    torch.rand(
                        (x.shape[-2], x.shape[-1]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    * 2
                    - 1
                )
                x += embs[None, None, None]
            elif self.feature_positional_embedding == "learned":
                w = self.feature_positional_embedding_embeddings.weight
                embs = w[
                    torch.randint(
                        0,
                        w.shape[0],
                        (x.shape[-2],),
                    )
                ]
                x += embs[None, None, None]
            elif self.feature_positional_embedding == "subspace":
                embs = torch.randn(
                    (x.shape[-2], x.shape[-1] // 4),
                    device=x.device,
                    dtype=x.dtype,
                )
                embs = self.feature_positional_embedding_embeddings(embs)
                x += embs[None, None, None]
            elif self.feature_positional_embedding is None:
                embs = None
            else:
                raise ValueError(f"Unknown {self.feature_positional_embedding=}")
        return x