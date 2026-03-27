import torch
from torch import nn
from torch.nn.modules.transformer import Module
from typing import Union, List
from enum import Enum
from packaging import version
import torch.nn.functional as F
from typing import Union
from src.base.model_base import AttentionBase, TransformerBase
import einops
from .attention import MultiHeadAttention, NSAttention, MultiHeadAttentionInTabPFN, MultiHeadAttentionCrossGroup


class Activation(Enum):
    """Enum for activation functions."""
    GELU = 1
    RELU = 2

class MLP(nn.Module):
    linear1: torch.nn.Linear
    linear2: torch.nn.Linear
    activation: Activation

    def __init__(
        self,
        size: int,
        hidden_size: int,
        activation: Union[Activation, str],
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(
            size,
            hidden_size,
            bias=False,
        )
        self.linear2 = torch.nn.Linear(
            hidden_size,
            size,
            bias=False,
        )
        if isinstance(activation, str):
            activation = Activation[activation.upper()]
        self.activation = activation

        

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward pass of the MLP.

        Args:
            x: The input tensor.
        """
        input_shape = x.shape
        x = x.reshape(-1, x.size(-1))

        res = x
        x = self.linear1(x)
        if self.activation is Activation.GELU:
            x = torch.nn.functional.gelu(x)
        elif self.activation is Activation.RELU:
            x = torch.nn.functional.relu(x)
        else:
            raise NotImplementedError(
                f"Activation Function {self.activation} is not implemented.",
            )
        x = res + self.linear2(x)
    
        return x.reshape(input_shape)


class TransformerLayer(TransformerBase):
    def __init__(
        self,
        attention_between_features: AttentionBase,
        attention_between_ts: AttentionBase,
        attention_between_samples: AttentionBase,
        d_model: int,
        dim_feedforward: Union[int, None] = None,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        second_mlp: bool = False,
        layer_norm_with_elementwise_affine: bool = False,
        structure_type: str = '3D', 
    ):
        super().__init__(attention_between_features=attention_between_features, 
                         attention_between_ts=attention_between_ts,
                         attention_between_samples=attention_between_samples)

        self.structure_type = structure_type
        if dim_feedforward is None:
            dim_feedforward = 2 * d_model

        self.mlp = MLP(
            size=d_model,
            hidden_size=dim_feedforward,
            activation=activation,
        )
        total_layers = 5 if second_mlp else 4
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    d_model,
                    eps=layer_norm_eps,
                    elementwise_affine=layer_norm_with_elementwise_affine,
                )
                for _ in range(total_layers)
            ]
        )

        self.second_mlp = None
        if second_mlp:
            self.second_mlp = MLP(
                size=d_model,
                hidden_size=dim_feedforward,
                activation=activation,
            )
        


    def forward(
        self,
        X: torch.Tensor,
        num_test: Union[int, None] = None,
        lookback_window: Union[int, None] = None,
        ) -> torch.Tensor:
        '''
        Args:
            X:
                The transformer state passed as input to the layer of shape
                (batch_size, num_context, seq_len, num_feature_blocks, d_model).
            num_test:
                number of test samples.
                same in the batch
        Returns:
            The transformer state passed through the encoder layer.
        '''
        assert (
            len(X.shape) == 5
        ), "src must be of shape (batch_size, num_sample, seq_len, num_feature_blocks, d_model)"

        batch = X.shape[0]
        num_sample = X.shape[1]
        seq_len = X.shape[2]
        
        # feature
        X = einops.rearrange(X, 'b n s c e -> (b n s) c e') # b n s f d
        X = self.attention_between_features(X)
        X = einops.rearrange(X, '(b n s) c e -> b n s c e', b=batch, n=num_sample)
        i = 0
        X = self.layer_norms[i](X)
        
        # ts if 3D else pass
        i += 1
        X = einops.rearrange(X, 'b n s c e -> (b n c) s e')
        if X.shape[-2] > 1:
            X = self.attention_between_ts(X)
            X = self.layer_norms[i](X)
        X = einops.rearrange(X, '(b n c) s e -> b n s c e', b=batch, n=num_sample)
        

        if self.second_mlp is not None:
            X = self.second_mlp(X)
            i += 1
            X = self.layer_norms[i](X)


        # context
        kwargs = {}
        if NSAttention is not None and isinstance(self.attention_between_samples, NSAttention):
            kwargs["fine_selection_flex_mask"] = NSAttention.create_fine_mask(X.shape[1], self.attention_between_samples.fine_bock_size, causal=False)
        elif isinstance(self.attention_between_samples, MultiHeadAttention):
            kwargs["mask"] = num_sample - num_test
        if X.shape[-4] > num_test:
            ## 3D 
            if isinstance(self.attention_between_samples, MultiHeadAttentionInTabPFN):
                X = einops.rearrange(X, 'b n s c e -> (b s c) n e')
                new_x_test = self.attention_between_samples(
                    X[:, -num_test:],
                    X[:, :-num_test],
                    reuse_first_head_kv=True,
                )
                new_x_train = self.attention_between_samples(X[:, :-num_test])
                X = torch.cat((new_x_train, new_x_test), dim=1)
                X = einops.rearrange(X, '(b s c) n e -> b n s c e', b=batch, s=seq_len)
            elif isinstance(self.attention_between_samples, MultiHeadAttentionCrossGroup):
                # MultiHeadAttentionCrossGroup input: b n s e
                X = einops.rearrange(X, 'b n s c e -> (b c) n s e')
                X = self.attention_between_samples(X, **kwargs)
                X = einops.rearrange(X, '(b c) n s e -> b n s c e', b=batch)
            else:
                X = einops.rearrange(X, 'b n s c e -> (b s c) n e')
                X = self.attention_between_samples(X, **kwargs)
                X = einops.rearrange(X, '(b s c) n e -> b n s c e', b=batch, s=seq_len)
            
        i += 1
        X = self.layer_norms[i](X)
        X = self.mlp(X)
        i += 1
        X = self.layer_norms[i](X)
        return X