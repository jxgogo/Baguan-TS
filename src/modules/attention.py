from torch import nn
import torch
from typing import Union, List
import torch.nn.functional as F
from src.base.model_base import AttentionBase
from torch.backends.cuda import SDPBackend

class MultiHeadAttentionInTabPFN(AttentionBase):
    '''
    '''
    def __init__(
        self, 
        d_model:int = 128,
        nheads: int = 4,
        d_in: int = None,
        dropout_p: Union[float, None] = None,
        softmax_scale: Union[float, None] = None,
        flash: bool = True,
        ):
        super().__init__(d_model, nheads, d_in)
        self.flash = flash

        d_in = d_model if d_in is None else d_in
        d_head = d_model // nheads
        
        self.dropout_p = dropout_p if dropout_p is not None else 0.0
        self.scale = self.d_head ** -0.5 if softmax_scale is None else softmax_scale
        self.to_qkv = nn.Parameter(torch.randn(3, nheads, d_head, d_in))
        self.to_out = nn.Parameter(torch.randn(nheads, d_head, d_model))
        nn.init.normal_(self.to_qkv, std=0.01)
        nn.init.normal_(self.to_out, std=0.01)

        # for attention weights
        self.save_attention = False  # false for minimize influence on training
        self.latest_attention_map = None # to save attention weights
        

    def compute_attention(self, x: torch.Tensor, x_kv: torch.Tensor = None,
                           reuse_first_head_kv: bool = False) -> torch.Tensor:
        '''
        Args:
            x:
                The input tensor of shape (batch_size, sequence_length, d_in).
            x_kv:
                The key and value tensor of shape (batch_size, sequence_length, d_in).
            reuse_first_head_kv:
                Whether to reuse the key and value of the first head for all heads.
        Returns:
            The output tensor of shape (batch_size, sequence_length, d_model).
        '''
        assert len(x.shape) == 3, "x must be of shape (batch_size, sequence_length, d_in)"
        b, l, _ = x.shape 

        if reuse_first_head_kv:
            assert x_kv is not None, (
                "x and x_kv must be different tensors. That means reuse_first_head_kv"
                "is not compatible with self attention only cross attention."
            )


        if x_kv is None:
            # qkv = self.to_qkv(x).view(b, l, self.nheads, -1, 3).transpose(-4, -3)
            # q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]

            qkv = torch.einsum("... s, j h d s -> ... j h d", x, self.to_qkv) # b, l, 3, n_heads, d_head
            q, k, v = qkv.unbind(dim=-3) # b, l, n_heads, d_head
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        else:
            w_q, w_kv = self.to_qkv[0], self.to_qkv[1:]
            q = torch.einsum("... s, h d s -> ... h d", x, w_q) # b, l, n_heads, d_head
            if reuse_first_head_kv:
                orig_num_heads = w_kv.shape[1]
                w_kv = w_kv[:, :1]
            kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, w_kv)
            if reuse_first_head_kv:
                expand_shape = [-1 for _ in kv.shape]
                expand_shape[-2] = orig_num_heads
                kv = kv.expand(*expand_shape)
                k, v = kv.unbind(dim=-3)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


        cuda_config = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

        use_flash = self.flash and (not self.save_attention)

        if use_flash:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            # Check if there is a compatible device for flash attention
            # config = self.cuda_config if q.is_cuda else self.cpu_config

            #TODO: this is deprecated. if torch.__version__ >= version.parse("2.3.0"), can use :
            # with torch.nn.attention.sdpa_kernel(cuda_config):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, 
                enable_math=True, 
                enable_mem_efficient=True
            ):
                out = F.scaled_dot_product_attention(q, k, v,
                    dropout_p=self.dropout_p, scale=self.scale)
            attn_weights_to_return = False

        else:
            
            attn = torch.einsum("...ld, ...md -> ...lm", q, k) * self.scale # b, n_heads, l, l
            attn = torch.nn.functional.softmax(attn, dim=-1)
            attn_weights_to_return = attn
            if self.dropout_p is not None:
                attn = torch.nn.functional.dropout(attn, p=self.dropout_p, training=self.training)

            out = torch.einsum("...lm, ...md -> ...ld", attn, v) # b, n_heads, l, d_head
        
        if self.save_attention and (attn_weights_to_return is not None) and (x_kv is not None):
            self.latest_attention_map = attn_weights_to_return.detach().cpu()

        # out = out.transpose(-3, -2).contiguous().view(b, l, -1)
        out = torch.einsum(
            "b h l d, h d s -> b l s",
            out,
            self.to_out,
        )


        return out + x


class MultiHeadAttentionCrossGroup(nn.Module):
    def __init__(
        self, 
        d_model:int = 128,
        nheads: int = 4,
        d_in: int = None,
        dropout_p: Union[float, None] = None,
        softmax_scale: Union[float, None] = None,
        flash: bool = True,
        is_causal: bool = False,
        enable_rope: bool = False,
        initial_type: str = None,
        ):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        assert d_model % nheads == 0, "d_model must be divisible by nheads"
        self.d_head = d_model // nheads
        self.flash = flash
        self.is_causal = is_causal
        self.enable_rope = enable_rope
        # self.cpu_config, self.cuda_config = set_device_config(flash)

        d_in = d_model if d_in is None else d_in
        
        self.dropout_p = dropout_p if dropout_p is not None else 0.0
        self.scale = self.d_head ** -0.5 if softmax_scale is None else softmax_scale

        self.to_qkv = nn.Linear(d_in, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

        if initial_type == 'zero':
            nn.init.zeros_(self.to_qkv.weight)
            nn.init.zeros_(self.to_out.weight)
        elif initial_type == 'small_normal':
            nn.init.normal_(self.to_qkv.weight, std=0.01)
            nn.init.normal_(self.to_out.weight, std=0.01)

    def forward(self, x: torch.Tensor, mask:Union[torch.Tensor, List[int], int] = None) -> torch.Tensor:
        '''
        Args:
            x:
                The input tensor of shape (batch_size, sequence_length, group, d_in).
            mask:
                if mask is a list of integers, it represents the last mask samples is the test set, which needs to be masked.
                size is batch_size.
        Returns:
            The output tensor of shape (batch_size, sequence_length, group, d_model).
        '''
        assert len(x.shape) == 4, "x must be of shape (batch_size, sequence_length, group, d_in)"
        b, l, g, _ = x.shape 

        qkv = self.to_qkv(x).view(b, l, g, self.nheads, -1, 3).transpose(-5, -3) # b h g l d' 3
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2] # b h g l d'
        q_group, k_group = q.mean(dim=2), k.mean(dim=2) # b h l d'
        attn_score = torch.einsum('bhld,bhld->bhll', q_group, k_group) * self.scale # b h l l
        assert isinstance(mask, int)
        attn_score[...,mask:] = -torch.inf # b h l l
        attn_score = torch.softmax(attn_score, dim=-1) # b h l l
        out = torch.einsum('bhlm, bhgmd->bhgld', attn_score, v).transpose(1, -2).contiguous().view(b, l, g, -1) # bhgld -> blghd -> blgd
        return self.to_out(out) + x # blgd


class MultiHeadAttention(AttentionBase):
    def __init__(
        self, 
        d_model:int = 128,
        nheads: int = 4,
        d_in: int = None,
        dropout_p: Union[float, None] = None,
        softmax_scale: Union[float, None] = None,
        flash: bool = True,
        is_causal: bool = False,
        enable_rope: bool = False,
        initial_type: str = None,
        ):
        super().__init__(d_model, nheads, d_in)
        self.flash = flash
        self.is_causal = is_causal
        self.enable_rope = enable_rope
        # self.cpu_config, self.cuda_config = set_device_config(flash)

        d_in = d_model if d_in is None else d_in
        
        self.dropout_p = dropout_p if dropout_p is not None else 0.0
        self.scale = self.d_head ** -0.5 if softmax_scale is None else softmax_scale

        self.to_qkv = nn.Linear(d_in, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

        if initial_type == 'zero':
            nn.init.zeros_(self.to_qkv.weight)
            nn.init.zeros_(self.to_out.weight)
        elif initial_type == 'small_normal':
            nn.init.normal_(self.to_qkv.weight, std=0.01)
            nn.init.normal_(self.to_out.weight, std=0.01)
    

    def add_rope(self, embedded_input, dim):
        """
        dim of rope
        """
        d_model = embedded_input.shape[-1]
        M = embedded_input.shape[dim]
        m_list = torch.arange(M, device=embedded_input.device).reshape(-1,1)
        base_freq = 10000 # 200 
        theta_list = (base_freq**(-2*torch.arange(0, d_model, 2, device=embedded_input.device)/d_model)).reshape(1,-1)
        pos_embed_theta = m_list*theta_list
        freqs_shape = [1] * len(embedded_input.shape)
        freqs_shape[dim] = M
        pos_embed_theta = pos_embed_theta.view(*freqs_shape[:-1], d_model//2)
        # pos_embed_theta = pos_embed_theta.unsqueeze(0).unsqueeze(0).unsqueeze(-2)
        embedded_input_u = embedded_input[...,:d_model//2] * torch.cos(pos_embed_theta) - embedded_input[...,d_model//2:] * torch.sin(pos_embed_theta)
        embedded_input_l = embedded_input[...,d_model//2:] * torch.cos(pos_embed_theta) + embedded_input[...,:d_model//2] * torch.sin(pos_embed_theta)
        out =  torch.cat((embedded_input_u, embedded_input_l), dim=-1)
        return out

    def compute_attention(self, x: torch.Tensor, mask:Union[torch.Tensor, List[int], int] = None) -> torch.Tensor:
        '''
        Args:
            x:
                The input tensor of shape (batch_size, sequence_length, d_in).
            mask:
                if mask is a list of integers, it represents the last mask samples is the test set, which needs to be masked.
                size is batch_size.
        Returns:
            The output tensor of shape (batch_size, sequence_length, d_model).
        '''
        assert len(x.shape) == 3, "x must be of shape (batch_size, sequence_length, d_in)"
        b, l, _ = x.shape 

        qkv = self.to_qkv(x).view(b, l, self.nheads, -1, 3).transpose(-4, -3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if not self.is_causal:
            if mask is not None:
                if isinstance(mask, list):
                    assert b[0] % len(mask) == 0, "mask size must match batch_size"
                    feature_size = b[0] // len(mask)
                    temp = torch.ones(b, self.nheads, l, l, device=q.device)
                    for i, m in enumerate(mask):
                        temp[i*feature_size: (i+1)*feature_size, ..., m:] = 0
                    mask = (1 - temp) * (-torch.inf)
                elif isinstance(mask, int):
                    temp = torch.zeros(l, l, device=q.device)
                    temp[..., mask:] = -torch.inf
                    mask = temp
                else:
                    mask = None
        else:
            mask = None
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if self.enable_rope:
            q = self.add_rope(q, dim=-2)
            k = self.add_rope(k, dim=-2)

        if self.flash:
            cuda_config = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
            
            # Check if there is a compatible device for flash attention
            # config = self.cuda_config if q.is_cuda else self.cpu_config

            #TODO: this is deprecated. if torch.__version__ >= version.parse("2.3.0"), can use :
            # with torch.nn.attention.sdpa_kernel(cuda_config):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, 
                enable_math=True, 
                enable_mem_efficient=True
            ):
                out = F.scaled_dot_product_attention(q, k, v,
                    attn_mask=mask, dropout_p=self.dropout_p, scale=self.scale,is_causal=self.is_causal)

        else:
            out = F.scaled_dot_product_attention(q, k, v,
                    attn_mask=mask, dropout_p=self.dropout_p, scale=self.scale,is_causal=self.is_causal)
        out = out.transpose(-3, -2).contiguous().view(b, l, -1)
    
        return self.to_out(out) + x


NSAttention = None  
try:
    from native_sparse_attention_pytorch import SparseAttention
    from native_sparse_attention_pytorch.native_sparse_attention import create_fine_mask
    class NSAttention(AttentionBase, SparseAttention):
        
        def __init__(self, d_model: int, nheads: int, d_in: int = None, fine_block_size: int = 128, num_selected_blocks: int = 2):
            AttentionBase.__init__(self, d_model, nheads)
            SparseAttention.__init__(
                self,
                dim=d_model,
                heads=nheads,
                dim_head=d_model // nheads,
                sliding_window_size=fine_block_size,
                compress_block_size=fine_block_size,
                selection_block_size=fine_block_size,
                num_selected_blocks=num_selected_blocks,
            )
            self.fine_block_size = fine_block_size
            
        def compute_attention(self, x: torch.Tensor, fine_selection_flex_mask) -> torch.Tensor:
            assert len(x.shape) == 3, "x must be of shape (batch_size, sequence_length, d_in)"
            return SparseAttention.forward(self, x, fine_selection_flex_mask=fine_selection_flex_mask)
        
        @classmethod
        def create_fine_mask(cls, sequence_length: int, fine_block_size: int, causal: bool = False):
            return create_fine_mask(sequence_length, fine_block_size, causal)
except ImportError as e:
    import logging
    logging.debug(f"Sparse attention not available: {e}")
    
    NSAttention = None  

