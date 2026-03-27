from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class PositionalEmbedderBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            inputs: 5D tensor with shape (batch_size, num_samples, seq_len_new, num_channels, embed_dim)
        Returns:
            outputs: 5D tensor with shape (batch_size, num_samples, seq_len_new, num_channels, embed_dim)
        '''
        pass

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.dim() == 5, "Input should be 5D tensor"
        outputs = self.embed(inputs)
        assert outputs.dim() == 5, "Output should be 5D tensor"
        return outputs


class EncoderBase(nn.Module, ABC):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    @abstractmethod
    def encode(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        Args:
            inputs: 4D tensor with shape (batch_size, dataset_size, seq_len, num_channels)
        Returns:
            outputs: 5D tensor with shape (batch_size, dataset_size, seq_len, num_channels, embed_dim)
        '''
        pass

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        assert inputs.dim() == 4, "Input should be 3D tensor"
        outputs = self.encode(inputs, **kwargs)
        assert outputs.dim() == 5, "Output should be 5D tensor"
        return outputs

class JointEncoderBase(nn.Module, ABC):
    def __init__(self, embeding_dim: int):
        super().__init__()
        self.embeding_dim = embeding_dim
        
    @abstractmethod
    def encode(self, feature_ts: torch.Tensor, target_ts: torch.Tensor, **kwargs) -> tuple:
        '''
        Args:
            feature_ts: 4D tensor with shape (batch_size, num_samples, seq_lens, num_channels_f)
            target_ts: 4D tensor with shape (batch_size, num_samples, seq_lens, num_channels_t)
        Returns:
            feature_ts_embedding: 5D tensor with shape (batch_size, num_samples, seq_len_new, num_channels_f_new, embed_dim)
            target_ts_embedding: 5D tensor with shape (batch_size, num_samples, seq_len_new, num_channels_t, embed_dim)
        '''
        pass

    def forward(self, feature_ts: torch.Tensor, target_ts: torch.Tensor, **kwargs) -> torch.Tensor:
        assert feature_ts.dim() == 4, "feature_ts should be 4D tensor"
        assert target_ts.dim() == 4, "target_ts should be 4D tensor"
        outputs = self.encode(feature_ts, target_ts, **kwargs)
        return outputs

class AttentionBase(nn.Module, ABC):
    def __init__(self, d_model: int, nheads: int, d_in: int = None):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        assert d_model % nheads == 0, "d_model must be divisible by nheads"
        self.d_head = d_model // nheads
        
    @abstractmethod
    def compute_attention(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        Args:
            inputs: 3D tensor with shape 
                (batch_size, num_context, embed_dim)
                or (batch_size, num_channels, embed_dim)
                or (batch_size, ts, embed_dim)

            kwargs: additional arguments, such as mask
        Returns:
            outputs: 3D tensor with the same shape as inputs
        '''
        pass

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:  
        assert inputs.dim() == 3, "Input should be 3D tensor"
        attention_outputs = self.compute_attention(inputs, *args, **kwargs)
        assert attention_outputs.dim() == 3, "Output should be 3D tensor"
        return attention_outputs


class DecodeBase(nn.Module, ABC):
    def __init__(self, input_dim: int, output_dim: int, patch_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        
    @abstractmethod
    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        Note: if one token is a patch of samples or features, each token should be decoded to multiple values.

        Args:
            inputs: 4D tensor with shape (batch_size, forecast_horizon, num_targets, embed_dim)
        Returns:
            outputs: 4D tensor with shape (batch_size, forecast_horizon, num_targets, unit_per_token)
        '''
        pass


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.dim() == 5, "Input should be 5D tensor"
        outputs = self.decode(inputs)
        assert outputs.dim() == 5, "Output should be 5D tensor"
        return outputs



class TransformerBase(nn.Module, ABC):
    def __init__(
        self,
        attention_between_features: AttentionBase,
        attention_between_ts: AttentionBase,
        attention_between_samples: AttentionBase,
    ):
        super().__init__()
        self.attention_between_features = attention_between_features
        self.attention_between_ts = attention_between_ts
        self.attention_between_samples = attention_between_samples
        

    @abstractmethod
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        Args:
            input: 5D tensor with shape (batch_size, num_context, seq_len, num_channels + num_targets, embed_dim)
            kwargs: additional arguments, such as mask
        Returns:
            outputs: 5D tensor with shape (batch_size, num_context, seq_len, num_channels + num_targets, embed_dim)
        '''
        pass
