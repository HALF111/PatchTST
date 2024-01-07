__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# https://github.com/luo3300612/Visualizer
# from visualizer import get_local

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# Cell
class PatchTST_backbone_attn_weighted(nn.Module):
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
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        # 这里默认是做RevIN的，且维度c_in为channel维
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        # PS：这里context_window就等于seq_len，target_window就等于pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        # 获得总的patch数
        patch_num = int((context_window - patch_len) / stride + 1)
        
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone
        # 核心的Encoder框架
        # （PS：embedding和位置编码也都会在TSTiEncoder内完成）
        # * 注意：由于注意力中的mask需要用到data_len，也即token的个数，对应于patch_num，所以这里需要将其传入进去！！！
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, 
                                data_len=patch_num, **kwargs)

        # Head
        # 也即最后一个"展平 & 线性层"
        # 其将(d_model, patch_num)的输入战平后并映射至(1, pred_len)的输出
        self.d_model = d_model
        self.head_nf = d_model * patch_num  # d_model和patch总数的乘积，为encoder的输出
        self.n_vars = c_in  # n_vars就设置为channel数
        self.pretrain_head = pretrain_head  # 默认为False
        self.head_type = head_type  # 默认为"flatten"
        self.individual = individual  # 默认为False
        
        # [bs x nvars x (patch_num*d_model)]
        self.weight_lst = torch.zeros((self.n_vars, patch_num))
        self.avg_weight_lst = torch.zeros((self.n_vars, patch_num))
        
        # # 生成权重矩阵/掩码矩阵
        # self.weight_mask = WeightMask(patch_num*d_model, d_model, weight_lst=self.weight_lst)

        if self.pretrain_head:
            # 如果要预训练的话，那么改用一个一维卷积的预测头
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            # 如果不预训练线性层的话，那么调用Flatten_Head
            # 它会完成：展平 + 线性映射 + dropout （不过这里的dropout默认设置为0）
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
            
    
    def forward(self, z):                                                   # z: [bs x nvars x seq_len]
        # norm
        # 1、先做RevIN归一化
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        # 2、做patching分割
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        # unfold函数啊按照选定的尺寸与步长来切分矩阵，相当于滑动窗口操作，也即只有卷、没有积
        # 参数为（dim,size,step）：dim表明想要切分的维度，size表明切分块（滑动窗口）的大小，step表明切分的步长
        # 这里用于从一个分批输入的张量中提取滑动的局部块
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        # 3、经过encoder主干模型（PS：embedding和位置编码都会在backbone内完成）
        z = self.backbone(z)                                                # z: [bs x nvars x d_model x patch_num]
        
        # # 添加权重数据
        # z, mask_for_print = self.weight_mask(z)                                             # z: [bs x nvars x d_model x patch_num]
        
        # print("weight_lst:")
        # self.weight_lst = self.weight_lst.to(z.device)
        # self.weight_lst += mask_for_print
        # print(self.weight_lst)
        # self.avg_weight_lst = self.avg_weight_lst.to(z.device)
        # self.avg_weight_lst = self.weight_lst / self.weight_lst[-1][-1]
        # print("avg_weight_lst:")
        # print(self.avg_weight_lst)
        
        # 4、展平 + 线性映射（+dropout）
        z = self.head(z)                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        # 5、再用RevIN做denorm回来
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        # 这里的dropout会使用fc_dropout
        return nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Conv1d(in_channels=head_nf, out_channels=vars, kernel_size=1)
                )

# 计算权重掩码矩阵
class WeightMask(nn.Module):
    def __init__(self, data_len, segment_len=None, acf_lst=None) -> None:
        super().__init__()
        
        self.segment_len = segment_len
        
        # # 因为acf_lst是univariate的，所以需要修改其和channel数一样多
        # if acf_lst is None:
        #     self.acf_lst = acf_lst
        # else:
        #     if self.segment_len is not None:
        #         self.acf_lst = []
        #         for item in acf_lst:
        #             tmp_lst = [item]*self.segment_len
        #             self.acf_lst.extend(tmp_lst)
        #         # 最后别忘记将list转成tensor
        #         self.acf_lst = torch.tensor(self.acf_lst)
        #         self.acf_lst = self.acf_lst.float()  # 转成float的dtype
        #     else:
        #         raise Exception("the combiation of segment_len and acf_lst is illegal!")
        
        if segment_len is None:
            self.triu = torch.triu(torch.ones(data_len, data_len), diagonal=0)
        else:
            self.triu = torch.zeros(data_len, data_len)
            for i in range(data_len):
                for j in range(data_len):
                    import math
                    # 在判断条件时应该按照[1, data_len]的范围来判断，而非[0, data_len-1]来
                    if (i+1) <= segment_len * math.ceil((j+1) / segment_len):
                        self.triu[i][j] = 1
        
        self.mask_weight = nn.Linear(data_len, data_len)
        
        # * 由于这里的输入[bs x nvars x (patch_num*d_model)]，
        # * 所以我们应当是沿着dim=-1的patch_num的维度做softmax的！！！
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x: [bs x nvars x d_model x patch_num]
        # 由于我们需要对各个patch_num做掩码矩阵，也即给不同的patch_num以不同的权重值
        # 但又由于我们不希望不同d_model会有不相同的权重，所以这里要将后两维展平，并做分段的mask
        
        x = x.permute(0,1,3,2)  # 先permute一下，变成[bs x nvars x patch_num x d_model]
        
        bs, nvars, patch_num, d_model = x.shape
        
        x = x.reshape(bs, nvars, -1)  # x: [bs x nvars x (patch_num*d_model)]
        
        # 将上三角函数移到device上
        self.triu = self.triu.to(x.device)  # triu: [bs x nvars x (patch_num*d_model)]
        
        # 1.计算x和Linear的乘积，得到门控值
        gate = self.mask_weight(x)  # gate: [bs x nvars x (patch_num*d_model)]
        
        # # 1.5 如果还有acf_lst，那么这个时候需要提前先乘上acf_lst
        # # 相当于对每个seq_len的位置引入了先验知识
        # if self.acf_lst is not None:
        #     self.acf_lst = self.acf_lst.to(x.device)  # 先切换到同一个device
        #     cur_seq_len = x.shape[1]  # 取出需要的哪一个部分的seq_len
        #     gate = gate * self.acf_lst[-cur_seq_len:]  # 这里是逐元素乘法
        
        # 2.门控值需要先过softmax做归一化
        # !!! 一定不要忘记softmax！！！
        # ! 之前就是这里没有归一到[0, 1]之间，导致后面做exp运算时超出数字边界，得到inf值了！！！
        gate = self.softmax(gate)  # gate: [bs x nvars x (patch_num*d_model)]
        
        # print(x.dtype)
        # print(gate.dtype)
        # print(self.acf_lst.dtype)
        # print(self.triu.dtype)
        
        # 3.和上三角矩阵L_n做矩阵乘法
        mask = torch.matmul(gate, self.triu)  # mask: [bs x nvars x (patch_num*d_model)]
        
        # 打印gate和mask等的相关的信息
        torch.set_printoptions(profile="full")
        # print(f"self.triu: {self.triu}")
        # torch.set_printoptions(profile="default") # reset
        # print(f"gate in WeightMask: {gate}")
        mask_for_print = mask[:, :, ::self.segment_len]  # 由于多个channel共享相同的权重值，所以间隔着只取出第一个就ok了
        # mask_for_print = mask[:, :, :]  # 由于多个channel共享相同的权重值，所以间隔着只取出第一个就ok了
        mask_for_print = mask_for_print.mean(dim=0)  # 对当前batch的所有样本取平均值？
        print(f"mask_for_print in WeightMask: {mask_for_print}")
        # print(mask.shape)
        torch.set_printoptions(profile="default") # reset
        
        # 4.最后将得到的mask和原来的x做逐元素乘法，得到最终结果
        # * 简单来说就是给输入中的每个元素以不同的权重
        x = torch.mul(mask, x)  # x: [bs x nvars x (patch_num*d_model)]
        
        # 5.1 重新reshape回来
        x = x.reshape(bs, nvars, patch_num, d_model)  # x: [bs x nvars x patch_num x d_model]
        # 5.2 然后再permute回去
        x = x.permute(0,1,3,2)
        
        return x, mask_for_print


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        # 最后一个"展平 & 线性层"
        # 其将(d_model, patch_num)的输入战平后并映射至(1, pred_len)的输出
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            # 将d_model和patch_num维展平，并做线性映射到1*pred_len维
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            # 遍历各个channel
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            # 先展平、再线性映射，还外加一个dropout（不过默认为0）
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  # "i" means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, 
                 data_len=None, **kwargs):
        
        # 主干网络
        # 包括完成embedding和encoder部分
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        # 这里用一个线性层做value-embedding，将输入从patch_len维映射到d_model维
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        # 有很多位置编码备选项，这里默认使用"zeros"，也即初始化为[-0.02,0.02]区间内的均匀分布
        # * 同时，learn_pe为True，说明这里位置编码默认是可学习的
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        # 最关键的encoder部分
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn,
                                   data_len=data_len)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        # 1、value-embedding，将输入从patch_len映射到d_model
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        # channel independence
        u = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))      # u: [(bs * nvars) x patch_num x d_model]
        
        # 2、将embedding和位置编码相加，再经过dropout
        u = self.dropout(u + self.W_pos)                                         # u: [(bs * nvars) x patch_num x d_model]

        # Encoder
        # 3、进入encoder做处理
        z = self.encoder(u)                                                      # z: [(bs * nvars) x patch_num x d_model]
        
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
                        data_len=None):
        super().__init__()

        # 一共有n_layers层encoder
        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn, data_len=data_len) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                # 上面的这个是原来的code
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                # # ! 注意这里有一个小的改动，目的是为了可视化注意力权重！！！
                # output = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: 
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False,
                 data_len=None):
        super().__init__()
        
        # 如果未设置，那么d_k和d_v都默认为(d_model // n_heads)
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention_Weighted(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, data_len=data_len)

        # Add & Norm
        # 第一个是Attention中的norm
        # 注意，PatchTST参考TST的设计，其中默认使用的是BatchNorm
        # 而不像In/Auto/FED-former等默认使用LayerNorm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        # 这里默认使用GELU作为MLP的中间层，dropout也夹在中间而非末尾
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        # 第二个是FFN中的Norm，同样也是默认为BatchNorm，也可能是LayerNorm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        # 此二者默认为False？？
        self.pre_norm = pre_norm
        self.store_attn = store_attn


    # @get_local('attn')
    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        # 1、在注意力前先做一个BatchNorm（也可能是LayerNorm？）
        # 不过self.pre_norm默认为False，也就是默认不做normalization？？
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        # 2、多头自注意力
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        # 3.1 Add：先做个dropout、然后和残差连接的src相加
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        # 3.2 Norm：然后也做一个BatchNorm（也可能是LayerNorm）
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        # 4、FFN
        # 这里FFN默认为使用GELU激活的MLP
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        # 5、也是Add + Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src



# ! 这里对Multihead的Attention做了修改！！！
class _MultiheadAttention_Weighted(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, data_len=None, segment_len=None, acf_lst=None):
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
        
        # ? 这里应当在多个头之间共享，还是对每个头有一个独立的权重？
        # 目前先做成共享的~
        self.W_Q_L = nn.Linear(d_k, d_k)
        self.W_K_L = nn.Linear(d_k, d_k)
        self.W_Q_R = nn.Linear(d_k, d_k)
        self.W_K_R = nn.Linear(d_k, d_k)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention_Weighted(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        # 用一个线性映射 + dropout映射回原来的维度
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
        
        
        # * 额外加入的上三角矩阵
        # * 用于完成mask的动态注意力
        self.segment_len = segment_len
        
        # # 因为acf_lst是univariate的，所以需要修改其和channel数一样多
        # if acf_lst is None:
        #     self.acf_lst = acf_lst
        # else:
        #     if self.segment_len is not None:
        #         self.acf_lst = []
        #         for item in acf_lst:
        #             tmp_lst = [item]*self.segment_len
        #             self.acf_lst.extend(tmp_lst)
        #         # 最后别忘记将list转成tensor
        #         self.acf_lst = torch.tensor(self.acf_lst)
        #         self.acf_lst = self.acf_lst.float()  # 转成float的dtype
        #     else:
        #         raise Exception("the combiation of segment_len and acf_lst is illegal!")
        
        
        # * 事实上，这里的data_len实际上应该是patch_num
        # * 因为这里子注意力中data_len等于token数量，而token数量恰为patch_num
        if segment_len is None:
            # 一个普通的上三角矩阵
            self.triu = torch.triu(torch.ones(data_len, data_len), diagonal=0)
        else:
            # 一个分段的上三角矩阵
            self.triu = torch.zeros(data_len, data_len)
            for i in range(data_len):
                for j in range(data_len):
                    import math
                    # 在判断条件时应该按照[1, data_len]的范围来判断，而非[0, data_len-1]来
                    if (i+1) <= segment_len * math.ceil((j+1) / segment_len):
                        self.triu[i][j] = 1

    
    def _calculate_weighted_mask_two_side(self, q_s, k_s):
        '''
        q_s: [bs x n_heads x max_q_len x d_k]
        k_s: [bs x n_heads x d_k x q_len]
        '''
        # 先获得scale信息
        scale = nn.Parameter(torch.tensor(self.d_k ** -0.5))
        
        # print(scale)
        # print(q_s.shape)
        # print(k_s.shape)
        
        # 然后对q和k和映射矩阵相乘
        q_L = self.W_Q_L(q_s)  # q_L: [bs x n_heads x max_q_len x d_k]，下同
        q_R = self.W_Q_R(q_s)
        # 注意，由于这里k的d_k在-2维上，所以需要先转置再矩阵乘法再转置回来
        k_L = k_s.permute(0,1,3,2)  # k_L: [bs x n_heads x q_len x d_k]
        k_R = k_s.permute(0,1,3,2)
        k_L = self.W_K_L(k_L)       # k_L: [bs x n_heads x q_len x d_k]
        k_R = self.W_K_R(k_R)
        k_L = k_L.permute(0,1,3,2)  # k_L: [bs x n_heads x d_k x q_len]
        k_R = k_R.permute(0,1,3,2)
        
        # 然后先计算分数
        score_L = torch.matmul(q_L, k_L) * scale  # score_L: [bs x n_heads x max_q_len x q_len]，下同
        score_R = torch.matmul(q_R, k_R) * scale
        
        # 再做softmax
        # 由于我们要对kv对中的n个token做softmax，那么这里需要对最后一维做softmax
        phi_L = F.softmax(score_L, dim=-1)  # phi_L: [bs x n_heads x max_q_len x q_len]
        phi_R = F.softmax(score_R, dim=-1)
        
        # 然后将phi矩阵和上三角矩阵L计算得到掩码矩阵
        self.triu = self.triu.to(q_s.device)  # triu: [q_len x q_len]
        m_L = torch.mul(torch.matmul(phi_L, self.triu), torch.matmul(phi_R, self.triu.T))  # m_L: [bs x n_heads x max_q_len x q_len]
        m_R = torch.mul(torch.matmul(phi_R, self.triu), torch.matmul(phi_L, self.triu.T))
        
        # 合并两个矩阵，得到最终的掩码矩阵
        m_s = m_L + m_R  # m: [bs x n_heads x max_q_len x q_len]
        
        return m_s

    def _calculate_weighted_mask_left(self, q_s, k_s):
        '''
        q_s: [bs x n_heads x max_q_len x d_k]
        k_s: [bs x n_heads x d_k x q_len]
        '''
        # 先获得scale信息
        scale = nn.Parameter(torch.tensor(self.d_k ** -0.5))
        
        # print(scale)
        # print(q_s.shape)
        # print(k_s.shape)
        
        # 然后对q和k和映射矩阵相乘
        q_L = self.W_Q_L(q_s)  # q_L: [bs x n_heads x max_q_len x d_k]
        # 注意，由于这里k的d_k在-2维上，所以需要先转置再矩阵乘法再转置回来
        k_L = k_s.permute(0,1,3,2)  # k_L: [bs x n_heads x q_len x d_k]
        k_L = self.W_K_L(k_L)       # k_L: [bs x n_heads x q_len x d_k]
        k_L = k_L.permute(0,1,3,2)  # k_L: [bs x n_heads x d_k x q_len]
        
        # 然后先计算分数
        score_L = torch.matmul(q_L, k_L) * scale  # score_L: [bs x n_heads x max_q_len x q_len]
        
        # 再做softmax
        # 由于我们要对kv对中的n个token做softmax，那么这里需要对最后一维做softmax
        phi_L = F.softmax(score_L, dim=-1)  # phi_L: [bs x n_heads x max_q_len x q_len]
        
        # 然后将phi矩阵和上三角矩阵L计算得到掩码矩阵
        self.triu = self.triu.to(q_s.device)  # triu: [q_len x q_len]
        m_L = torch.matmul(phi_L, self.triu)  # m_L: [bs x n_heads x max_q_len x q_len]
        
        # 由于我们只对一边做掩码，所以最终的掩码矩阵就等于m_L
        m_s = m_L  # m: [bs x n_heads x max_q_len x q_len]
        
        # # 打印gate和mask等的相关的信息
        # torch.set_printoptions(profile="full")
        # # print(f"self.triu: {self.triu}")
        # mask_for_print = m_s  # 由于多个channel共享相同的权重值，所以间隔着只取出第一个就ok了
        # mask_for_print = mask_for_print.mean(dim=0)  # 对当前batch的所有样本取平均值？
        # mask_for_print = mask_for_print.mean(dim=0)  # 再取一次平均，这次是对head做平均
        # # print(f"mask_for_print.shape in Attention: {mask_for_print.shape}")
        # # print(f"mask_for_print in Attention: {mask_for_print}")  # 取出所有token对应的mask
        # print(f"mask_for_print[0] in Attention: {mask_for_print[0]}")  # 取出第一个token的mask
        # print(f"mask_for_print[-1] in Attention: {mask_for_print[-1]}")  # 取出最后一个token的mask
        # # print(mask.shape)
        # torch.set_printoptions(profile="default") # reset
        
        return m_s


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        # 先将线性变换对应应用在QKV矩阵上
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s: [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s: [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s: [bs x n_heads x q_len x d_v]

        # 计算注意力掩码矩阵
        m_s = self._calculate_weighted_mask_left(q_s, k_s)

        # Apply Scaled Dot-Product Attention (multiple heads)
        # 多头点积注意力输出结果
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask, attn_weighted_mask=m_s)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask, attn_weighted_mask=m_s)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        # transpose + to_out 变换回原来的维度
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention_Weighted(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        # head_dim默认为d_model // n_heads
        head_dim = d_model // n_heads
        
        # scale默认为：sqrt(head_dim)
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, attn_weighted_mask=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
            attn_weighted_mask       : [1 x n_heads x max_seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        # 先计算Q和K的点积
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        # （可选）在softmax前，从之前层加入注意力分数
        if prev is not None: attn_scores = attn_scores + prev

        # 这里我们改变了
        # # Attention mask (optional)
        # if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
        #     if attn_mask.dtype == torch.bool:
        #         attn_scores.masked_fill_(attn_mask, -np.inf)
        #     else:
        #         attn_scores += attn_mask

        # # Key padding mask (optional)
        # if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
        #     attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        

        # normalize the attention weights
        # 将注意力分数再经过softmax
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)
        
        # 这里注意，需要逐元素乘上attn_weighted_mask
        # 并且应当是在昨晚softmax之后再做mask
        # ! 那么dropout应当先做还是后做呢？目前是先做的。
        attn_weights = torch.mul(attn_weights, attn_weighted_mask)    # attn_weights: [bs x n_heads x max_q_len x q_len]

        # compute the new values given the attention weights
        # 将注意力分数和v计算后得到最终输出
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

