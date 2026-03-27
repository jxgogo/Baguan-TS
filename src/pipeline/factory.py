import importlib
import yaml
from torch import nn
import lightning as L
import torch
from typing import List
import torch.distributed as dist
import random
import numpy as np
import einops
from src.modules.transformer import TransformerLayer
from src.utils.module_utils import get_cosine_schedule_with_warmup, get_openai_lr, z_score_normalizer, CRLoss
from src.base.model_base import EncoderBase, AttentionBase, DecodeBase


class TimeDICL(nn.Module):
    def __init__(self, 
                 input_encoder: EncoderBase, 
                 positional_embedder: EncoderBase,
                 transformers: List[TransformerLayer],
                 prediction_head: DecodeBase):
        super().__init__()
        self.input_encoder = input_encoder
        self.pos_embedder = positional_embedder
        self.transformer_encoder = nn.ModuleList(transformers)
        self.head = prediction_head
        self.merge_time_feature = False
    
    def forward(self, feature_ts: torch.Tensor, target_ts: torch.Tensor, 
                num_test: int, predict_len: int, predict_type='mean+quantile', quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) -> torch.Tensor:
        '''
        Args:
            feature_ts: (batch_size, num_samples, seq_len, num_channels_f)
            target_ts: (batch_size, num_samples, seq_len, num_channels_t)
            num_test: int | None
            predict_len: int | None
        Returns:
            out: (batch_size, num_test, predict_len, num_channels_t)
        '''
        assert target_ts is not None, "y must be provided"
        assert feature_ts.shape[0] == target_ts.shape[0], "batch size mismatch"
        assert feature_ts.shape[1] == target_ts.shape[1], "data size mismatch"
        assert feature_ts.shape[2] == target_ts.shape[2], "seqence len mismatch"
        if num_test > feature_ts.shape[1]:
            raise ValueError(f'number of test samples should less than number of dataset')
        # print(f'feature_ts.shape={feature_ts.shape}')
        if predict_len > feature_ts.shape[2]:
            raise ValueError(f'predict length should less than the squence length')
        num_channel_t = target_ts.shape[-1]
        seq_len = target_ts.shape[-2]
        context_ts_len = seq_len - predict_len

        ################# Encoding #################
        if self.training:
            ## add mask to simulate missing values
            mask_ratio = 0.3
            mask_per_size = (np.random.randint(5)+1)*self.input_encoder.patch_size
            b, n, s, f = target_ts.shape
            mask = torch.rand((b,n,s//mask_per_size,f),device=target_ts.device) <= mask_ratio
            mask = mask.unsqueeze(-2).repeat(1, 1, 1, mask_per_size, 1)
            mask = mask.reshape(b,n,-1,f)
            if mask.shape[-2] < s:
                 mask = torch.cat((mask, torch.zeros((b,n,s-mask.shape[-2],f),device=mask.device, dtype=mask.dtype)),dim=-2)
            target_ts.masked_fill_(mask>0.5, float('nan'))
        num_test_s = num_test
        feature_ts_e, target_ts_e = self.input_encoder(feature_ts, target_ts, num_test=num_test_s, predict_len=predict_len)
        #embedding: (batch_size, num_samples, num_patch, num_chennel_all, embedding_dim)
       
        out = torch.cat((self.pos_embedder(feature_ts_e), target_ts_e), dim=-2)
        if self.merge_time_feature:
            num_feature = out.shape[-2]
            out = einops.rearrange(out, 'b n s f e -> b n (s f) e').unsqueeze(-3)

        ######### transformer #########
        if context_ts_len % self.input_encoder.patch_size > 0:
            lookback_patch = context_ts_len // self.input_encoder.patch_size + 1
        else:
            lookback_patch = context_ts_len // self.input_encoder.patch_size
        if random.random() > -0.1:
            idx_th = 100
        else:
            idx_th = np.random.permutation(12)[0] + 1

        for idx, layer in enumerate(self.transformer_encoder):
            out = layer(out, num_test_s, lookback_patch) # b n s c_f+c_t e
            if idx >= idx_th:
                break
        if self.merge_time_feature:
            out = out.squeeze(-3)
            out = einops.rearrange(out, 'b n (s f) e -> b n s f e', f=num_feature)
        out = out[:, :, :, -num_channel_t:, :]
        decode_begin = int(np.floor(context_ts_len//self.input_encoder.patch_size)) 
        left_pad_len = context_ts_len - decode_begin*self.input_encoder.patch_size
        test_encoder_out = out[:, -num_test:, decode_begin:,-num_channel_t:,:] # b, s_test, n_t, c_t, e
        
        # ######### decoder #########
        out = self.head(test_encoder_out) # b, s_test, n_t, c_t, 1
        out = out[:,:,left_pad_len:left_pad_len+predict_len,:,:] # true prediction
        if predict_type == 'mean':
            return out
        elif predict_type == 'mean+quantile':
            if quantiles is None:
                raise ValueError(f'require quantiles values for quantile type prediction')
            quantile_out = self.head.decode_to_quantile(test_encoder_out, quantiles) 
            quantile_out = quantile_out[:,:,left_pad_len:left_pad_len+predict_len,...] # true needed prediction
            return out, quantile_out
        else:
            raise ValueError(f'unsupport predict type {predict_type}')

class ModelFactory:
    @staticmethod
    def from_config(config_path: str, return_config: bool = False):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        components = {}
        for module_type in ['input_encoder', 
                          'positional_embedder',
                          'prediction_head']:
            cls_config = config[module_type]
            module = importlib.import_module(f"src.modules.{cls_config['module']}")
            cls = getattr(module, cls_config['class'])
            components[module_type] = cls(**cls_config.get('params', {}))

        transformers = []
        cls_config = config['transformers']
        module = importlib.import_module(f"src.modules.{cls_config['module']}")
        cls = getattr(module, cls_config['class'])
        params = cls_config.get('params', {})
        attentions = {}
        for module_type in ['attention_between_features','attention_between_ts', 'attention_between_samples']:
            param_cls_config = params[module_type]
            param_module = importlib.import_module(f"src.modules.{param_cls_config['module']}")
            param_cls = getattr(param_module, param_cls_config['class'])
            if param_cls is None:
                raise ValueError(f"module {param_cls_config['module']} has no class {param_cls_config['class']}")
            attentions[module_type] = {
                'class': param_cls,
                'params': param_cls_config.get('params', {})
            }
        for _ in range(config["nlayers"]):
            for module_type in ['attention_between_features', 'attention_between_ts', 'attention_between_samples']:
                params[module_type] = attentions[module_type]["class"](**attentions[module_type]["params"])

        
            transformers.append(cls(**params))
        components['transformers'] = transformers
            
        return TimeDICL(**components)
    


