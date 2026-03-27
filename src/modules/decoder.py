from src.base.model_base import DecodeBase
from torch import nn
import torch
import numpy as np
import einops
from ..utils.module_utils import halfnormal_with_p_weight_before, translate_probs_across_borders
        

class ClassifierDecoder(DecodeBase):
    def __init__(self, input_dim: int, hidden_size: int, patch_size: int, output_dim: int = 5000, temperature=0.9):
        super().__init__(input_dim, output_dim, patch_size)
        self.linear1 = nn.Linear(self.input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.output_dim * self.patch_size)
        self.activation = nn.GELU()
        self.borders = nn.Parameter(torch.linspace(start=-10, end=10, steps=output_dim+1), requires_grad=False)
        bucket_width = self.borders[1:] - self.borders[:-1]
        self.bucket_means = self.borders[:-1] + bucket_width / 2 # 5000
        self.bucket_means = self.bucket_means.reshape(1, 1, 1, 1, -1) # 1, 1, 1, 1, 5000
        self.temperature = temperature


    def decode(self, inputs, **kwargs):
        self.bucket_means = self.bucket_means.to(inputs.device)
        x = self.linear1(inputs)
        x = self.activation(x)
        output = self.linear2(x)   # batch num_test_y forecast_period num_channle patch_size*output_dim
        output = einops.rearrange(output, "b y f n (p o) -> b y (f p) n o", p = self.patch_size)
        # output = torch.softmax(output, dim=-1)
        if self.training:
            # training -> prob
            return output
        
        # mean under temperature T
        output = torch.softmax(output/self.temperature, dim=-1)
        bucket_means = self.bucket_means.expand(*output.shape[:-1], -1) # b c s f 5000
        return (output * bucket_means).sum(dim=-1, keepdim=True) # b, s_test, n_y, 1
    
    def decode_to_quantile(self, inputs, quantiles):
        if isinstance(quantiles, list):
            quantiles = torch.tensor(quantiles)
        quantiles = quantiles.to(inputs.device)
        self.bucket_means = self.bucket_means.to(inputs.device)
        x = self.linear1(inputs)
        x = self.activation(x)
        output = self.linear2(x)   # batch num_test_y forecast_period num_channle patch_size*output_dim
        output = einops.rearrange(output, "b y f n (p o) -> b y (f p) n o", p = self.patch_size)
        output = torch.softmax(output/self.temperature, dim=-1)
        cdf = torch.cumsum(output, dim=-1)
        indices = torch.searchsorted(cdf, quantiles.expand(*cdf.shape[:-1],-1).contiguous(), right=False) 
        output = torch.gather(self.bucket_means.expand(*indices.shape[:-1], -1), -1, indices)
        return output
        

        
class MlpDecoder(DecodeBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int):
        super().__init__(input_dim, output_dim)
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_dim)
        self.activation = nn.GELU()

    def decode(self, inputs):
        x = self.linear1(inputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x