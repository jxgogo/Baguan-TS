import torch.nn as nn
from src.base.model_base import JointEncoderBase
import torch
import einops
import inspect


class ConvXYEncoder_Fourier(JointEncoderBase):
    def __init__(self, ninp: int, patch_size: int=8, features_per_group: int=1):
        super().__init__(ninp // 2)
        self.features_per_group = features_per_group
        self.patch_size = patch_size
        
        self.conv_x = nn.Conv2d(1, self.embeding_dim, (patch_size, features_per_group), stride=(patch_size, features_per_group))
        self.conv_y = nn.Conv2d(2, self.embeding_dim, (patch_size, 1), stride=(patch_size, 1))

        
    def encode(self, feature_ts, target_ts, **kwargs):
        num_test = kwargs["num_test"] 
        predict_len = kwargs["predict_len"] 
        batch_size, num_samples, seq_len, num_channels_f = feature_ts.shape

        # prepare for predicting part
        target_ts[:, -num_test:, -predict_len:, :] = float('nan') 
        indicator = torch.isnan(target_ts).float()
        target_ts = target_ts.nan_to_num()

        # pad to multiple of features_per_group
        missing_to_next_feature = (
            self.features_per_group - (num_channels_f % self.features_per_group)
        ) % self.features_per_group
        missing_to_next_ts = (
            self.patch_size - (seq_len % self.patch_size)
        ) % self.patch_size

        # when feature need pad
        if missing_to_next_feature > 0:
            feature_ts = torch.cat(
                [feature_ts, 
                 torch.zeros(batch_size, num_samples, seq_len, missing_to_next_feature, 
                             device=feature_ts.device, dtype=feature_ts.dtype)
                ], dim=-1
            )
            
        # when time need pad
        if missing_to_next_ts > 0:
            feature_ts = torch.cat(
                [feature_ts, feature_ts[:,:,-1:,:].repeat(1,1,missing_to_next_ts,1)
                 ], dim=-2
            )
            target_ts = torch.cat(
                [target_ts, target_ts[:,:,-1:,:].repeat(1,1,missing_to_next_ts,1),
                 ], dim=-2
            )
            indicator = torch.cat(
                [indicator, indicator[:, :, -1:, :].repeat(1, 1, missing_to_next_ts, 1),
                 ], dim=-2
            )

        ## encoding feature_ts
        # Splits up the input into subgroups
        feature_ts = einops.rearrange(feature_ts, "b n s c -> (b n) s c" ).unsqueeze(-3)
        # (b n) 1 s c -> (b n) e s' c' -> (b n) s' c' e -> b n s' c' e
        feature_ts = self.conv_x(feature_ts).permute(0, 2, 3, 1)
        feature_ts = einops.rearrange(feature_ts, "(b n) s c e -> b n s c e", b = batch_size)
        ## encoding target_ts
        target_ts = einops.rearrange(target_ts, "b n s c -> (b n) s c" ).unsqueeze(-3)
        indicator = einops.rearrange(indicator, "b n s c -> (b n) s c" ).unsqueeze(-3)

        target_ts = torch.cat((target_ts*(1-indicator),indicator), dim=-3)
        # (b n) 2 s c -> (b n) e s' c -> (b n) s' c e -> b n s' c e
        target_ts = self.conv_y(target_ts).permute(0, 2, 3, 1)
        target_ts = einops.rearrange(target_ts, "(b n) s c e -> b n s c e", b = batch_size)


        feature_ts = torch.cat((torch.cos(feature_ts), torch.sin(feature_ts)),dim=-1)
        target_ts = torch.cat((torch.cos(target_ts), torch.sin(target_ts)),dim=-1)
        return feature_ts.contiguous(), target_ts.contiguous()
