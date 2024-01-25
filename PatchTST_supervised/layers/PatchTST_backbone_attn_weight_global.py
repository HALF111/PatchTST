__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# Cell
class PatchTST_backbone_attn_weight_global(nn.Module):
    def __init__(self, 
                 c_in:int, 
                 context_window:int, 
                 target_window:int, 
                 patch_len:int, 
                 stride:int, 
                 max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, 
                 d_model=128, 
                 n_heads=16, 
                 d_k:Optional[int]=None, 
                 d_v:Optional[int]=None,
                 d_ff:int=256, 
                 norm:str='BatchNorm', 
                 attn_dropout:float=0., 
                 dropout:float=0., 
                 act:str="gelu", 
                 key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, 
                 attn_mask:Optional[Tensor]=None, 
                 res_attention:bool=True, 
                 pre_norm:bool=False, 
                 store_attn:bool=False,
                 pe:str='zeros', 
                 learn_pe:bool=True, 
                 fc_dropout:float=0., 
                 head_dropout = 0, 
                 padding_patch = None,
                 pretrain_head:bool=False, 
                 head_type = 'flatten', 
                 individual = False, 
                 revin = True, 
                 affine = True, 
                 subtract_last = False,
                 verbose:bool=False, 
                 acf_lst = None, batch_size=None,
                 **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        # PS：这里context_window就等于seq_len，target_window就等于pred_len
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        self.d_model = d_model
        
        # 获得的总Patch数
        patch_num = int((context_window - patch_len) / stride + 1)
        
        # 默认是需要做padding的？
        # padding大小事实上只是一个stride的大小！
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
            
        # 注意：patch_num要+1后再存储下来
        self.patch_num = patch_num
        
        
        # 在传入head之前，需要先将acf_lst变换一下
        # 原来：[channel, seq_len+pred_len]
        # 现在如果加在W_pred上为：[channel, patch_num*d_model, pred_len]
        # 现在如果加在mask上为：[channel, patch_num]
        new_acf_lst = None
        if acf_lst is not None:
            new_acf_lst = self.calc_acf(acf_lst)
            # 并且将其变成tensor形式
            new_acf_lst = torch.tensor(new_acf_lst)  # [channel, patch_num]
            # 为了和后面的大小持平，我们这里需要造出两个不同的acf_lst
            acf_lst_attn = new_acf_lst.unsqueeze(0).repeat(self.n_vars,1,1)
            acf_lst_pred = new_acf_lst
        
        
        # Backbone
        # * 注意：由于注意力中的mask需要用到data_len，也即token的个数，对应于patch_num，所以这里需要将其传入进去！！！
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, 
                                data_len=patch_num, acf_lst=new_acf_lst, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        
        # 首先从子模块中取出掩码矩阵对应的可学习参数w_s
        # 其维度为[1, patch_num]
        v_l = self.backbone.encoder.v_l
        w_s = self.backbone.encoder.w_s
        L_n = self.backbone.encoder.L_n
        # ! 别忘记这里需要对self.w_s做softmax！！！
        mask_global = torch.mm(v_l, F.softmax(w_s.weight))
        mask_global = torch.mm(mask_global, L_n)  # mask: [patch_num, patch_num] or [max_q_len, q_len]
        
        # print("mask_global[0]:", mask_global[0])
        # print("z.shape:", z.shape)
        # 然后就可以添加mask权重数据，这里只需要乘上mask_global[0]就可以了
        z = torch.mul(z, mask_global[0])                                           # z: [bs x nvars x d_model x patch_num]
        
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                )

    # # 这个计算出的结果是加在预测层的W_pred参数上的？？
    # def calc_acf(self, acf_lst):  # acf_lst: [channel, seq_len+pred_len]
    #     '''
    #     input: [channel, seq_len+pred_len]
    #     output: [channel, patch_num*d_model, pred_len]
    #     '''
    #     channels, _ = acf_lst.shape
    #     seq_len = self.context_window
    #     pred_len = self.target_window
        
    #     # 先计算[channels, seq_len, pred_len]的ACF_lst
    #     # [channel, seq_len, pred_len]
    #     new_acf_1 = np.zeros((channels, seq_len, pred_len))
    #     for i in range(seq_len):
    #         for j in range(pred_len):
    #             new_acf_1[:,i,j] = acf_lst[:, j-i+seq_len]
        
    #     # 然后通过avg操作合并同一个patch内的不同点的ACF
    #     # [channel, patch_num, d_model, pred_len]
    #     new_acf_2 = np.zeros((channels, self.patch_num, self.d_model, pred_len))
    #     for cur_patch in range(self.patch_num):
    #         start_idx = cur_patch * self.stride
    #         end_idx = start_idx + self.patch_len
    #         # 由于存在padding，而padding部分数据为0，那么最后一个patch只对无patch部分取平均值即可
    #         if end_idx > seq_len: end_idx = seq_len
            
    #         # 先计算abs，再计算mean
    #         select_acf_part = new_acf_1[:, start_idx:end_idx, :]  # [channel, patch_len, pred_len]
    #         cur_avg_abs_acf = np.mean(np.abs(select_acf_part), axis=1) # [channel, pred_len]
    #         # [channel, 1, pred_len]
    #         cur_avg_abs_acf = cur_avg_abs_acf.reshape(channels, 1, pred_len)
            
    #         # 广播到[channel, 1, d_model, pred_len]
    #         new_acf_2[:, cur_patch, :, :] = cur_avg_abs_acf
        
    #     # 最后将其reshape成[channel, patch_num*d_model, pred_len]
    #     new_acf_2 = new_acf_2.reshape(channels, self.patch_num*self.d_model, pred_len)
        
    #     return new_acf_2  # [channel, patch_num*d_model, pred_len]

    # 这个计算出的结果才应该是加在mask的可训练向量上的
    def calc_acf(self, acf_lst):  # acf_lst: [channel, seq_len+pred_len]
        '''
        input: [channel, seq_len+pred_len]
        output: [channel, patch_num]
        '''
        channels, _ = acf_lst.shape
        seq_len = self.context_window
        pred_len = self.target_window
        
        # 1、先计算[channels, seq_len, pred_len]的ACF_lst
        # [channel, seq_len, pred_len]
        new_acf_1 = np.zeros((channels, seq_len, pred_len))
        for i in range(seq_len):
            for j in range(pred_len):
                new_acf_1[:,i,j] = acf_lst[:, j-i+seq_len]
        
        # 2、然后通过avg操作合并同一个patch内的不同点的ACF
        new_acf_2 = np.zeros((channels, self.patch_num, pred_len))
        for cur_patch in range(self.patch_num):
            start_idx = cur_patch * self.stride  # 以stride步幅前进
            end_idx = start_idx + self.patch_len  # 但每次取出的patch长度均为patch_len
            # 由于存在padding，而padding部分数据为0，那么最后一个patch只对无patch部分取平均值即可
            if end_idx > seq_len: end_idx = seq_len
            
            # 先挑出这一个patch，再计算abs，再计算mean
            select_acf_part = new_acf_1[:, start_idx:end_idx, :]  # [channel, patch_len, pred_len]
            cur_avg_abs_acf = np.mean(np.abs(select_acf_part), axis=1) # [channel, pred_len]
            
            # 填入[channel, 1, pred_len]中
            new_acf_2[:, cur_patch, :] = cur_avg_abs_acf
        
        # 此时new_acf_2为[channel, patch_num, pred_len]
        # 再沿着pred_len做平均，即可得到[channel, patch_num]
        new_acf_2 = np.mean(new_acf_2, axis=-1)
        
        return new_acf_2  # [channel, patch_num]
    

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0, acf_lst=None):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        # 转换后的acf_lst的shape为：[channel, patch_num*d_model, pred_len]
        self.acf_lst = acf_lst
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            # if self.acf_lst is not None:
            #     # 注意：先转float再移动device
            #     self.acf_lst = self.acf_lst.float().to(x.device)
                
            #     x = self.flatten(x)  # x: [bs x nvars x (d_model * patch_num)]
            #     # x = self.linear(x)  # x: [bs x nvars x target_window]
            #     # 需要在这里额外加上一个ACF权重矩阵，并乘在输出上？
            #     # 转换后的acf_lst的shape为：[channel, (patch_num*d_model), pred_len]
            #     weight = self.linear.weight  # [pred_len, (patch_num*d_model)]
            #     weight = weight.unsqueeze(0).repeat(self.n_vars,1,1)  # [channel, pred_len, (patch_num*d_model)]
            #     weight = torch.mul(weight, self.acf_lst.permute(0,2,1))  # [channel, pred_len, (patch_num*d_model)]
            #     weight = weight.unsqueeze(0).repeat(x.shape[0],1,1,1) # [bs, n_vars, pred_len, (patch_num*d_model)]
            #     bias = self.linear.bias  # [pred_len]
                
            #     # print(x.shape, weight.shape)
            #     x = torch.matmul(weight, x.unsqueeze(-1)).squeeze(-1)
            #     x = x + bias
                
            #     x = self.dropout(x)
            # else:
                
            # 先展平、再线性映射，还外加一个dropout（不过默认为0）
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
    
    
class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, 
                 data_len=None, acf_lst=None, **kwargs):
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn,
                                   data_len=data_len, acf_lst=acf_lst)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))    # u: [(bs * nvars) x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [(bs * nvars) x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [(bs * nvars) x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))             # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
                        data_len=None, acf_lst=None):
        super().__init__()
        
        # ! 为了保证各层encoder共享相同的mask，应该将mask提前加到这里来？！
        # 维度为[1, N]的可学习参数
        # 另全为1的向量为v_l = [1,...,1]，维度为[1, M]
        # 那么掩码矩阵M_l = v_l^T * softmax(w_s) * L_n，维度为[M, N]：
        # 其中w_s维度为[1, N]为可学习参数，且可以初始化为全为1/N的向量。
        # L_n仍为上三角矩阵。
        
        # 由于这里的mask需要和query以及k-v的个数相关，所以需要乘上一个常数
        self.v_l = torch.ones(data_len, 1)
        # 可训练mask参数，并且bias置为False不存在
        # 或者也可以用nn.Parameter来实现
        self.w_s = nn.Linear(data_len, 1, bias=False)
        # 一个普通的上三角矩阵
        self.L_n = torch.triu(torch.ones(data_len, data_len), diagonal=0)
        # 同时我们对w_s采用constant=1/N初始化
        nn.init.constant_(self.w_s.weight, 1/data_len)
        
        # 存储acf_lst
        self.acf_lst = acf_lst

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn, 
                                                      data_len=data_len) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        src: [(bs * nvars) x patch_num x d_model]
        '''
        
        # 先计算mask
        # 先把几个常量挪到cuda上来
        self.v_l = self.v_l.to(src.device)
        self.L_n = self.L_n.to(src.device)
        # ! 别忘记这里需要对self.w_s做softmax！！！
        mask_global = torch.mm(self.v_l, F.softmax(self.w_s.weight))
        mask_global = torch.mm(mask_global, self.L_n)  # mask: [patch_num, patch_num] or [max_q_len, q_len]
        
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask, mask_global=mask_global)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask, mask_global=mask_global)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False,
                 data_len=None):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention,
                                             data_len=data_len)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, mask_global=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask, mask_global=mask_global)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask, mask_global=mask_global)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, data_len=None):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        
        # PS：d_q和d_k是相同的，所以这里只保留了d_k
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa, 
                                                   data_len=data_len)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None,
                mask_global=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                                              mask_global=mask_global)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                                 mask_global=mask_global)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False, data_len=None):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, mask_global=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        # * 事实上，我们应该在计算q和k之间的矩阵乘法时额外引入attention的global的mask：
        # * 目前由于我们希望各个attention层共享相同的mask_global，所以是从上面传下来的。
        # * 并且mask_global的shape为[max_q_len x q_len]
        attn_scores = torch.matmul(q, k)  # attn_scores : [bs x n_heads x max_q_len x q_len]
        # print("mask_global[0]:", mask_global[0])
        attn_scores = torch.mul(attn_scores, mask_global)  # attn_scores : [bs x n_heads x max_q_len x q_len]
        attn_scores = attn_scores * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        # 这里还需要多传一个self.w_s参数的值
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights