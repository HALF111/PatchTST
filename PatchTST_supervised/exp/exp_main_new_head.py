# https://github.com/luo3300612/Visualizer
# ! 注意：模型训练时不要调用get_local！！！
# ! 因为他会存储中间的注意力权重信息，导致这些内存一直没能释放，从而内存泄漏了！！！
# from visualizer import get_local
# get_local.activate()  # 激活装饰器

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from models import PatchTST_multi_MoE, PatchTST_MoE, PatchTST_head_MoE, PatchTST_weighted_avg
from models import PatchTST_weighted_concat, PatchTST_weighted_pred_layer_avg
from models import PatchTST_weighted_concat_no_constrain
from models import PatchTST_pred_layer_avg
from models import PatchTST_attn_weighted, PatchTST_attn_weight_global, PatchTST_attn_weight_global_indiv
from models import PatchTST_attn_weight_corr_dmodel_indiv
from models import PatchTST_random_mask
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main_new_head(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_new_head, self).__init__(args)
        
        # 记录对于每个数据集，其最优的patch大概是多少？
        data_path = self.args.data_path
        if "illness" in data_path:
            best_seq_len = [60, 80, 104]
        elif "ETTh1" in data_path:
            best_seq_len = [504, 1080, 1360]
        elif "ETTm2" in data_path:
            best_seq_len = [336, 720, 1200]
        elif "weather" in data_path:
            best_seq_len = [720, 900]
        else:
            best_seq_len = []
        self.best_patch_num = [(seq_len-self.args.patch_len)//self.args.stride+2 for seq_len in best_seq_len]
            

    def _build_model(self):
        # 需要在初始化的时候就调用函数生成ACF矩阵
        if self.args.get_forecastability:
            self.get_forecastability()
        else:
            self.acf_lst = self.args.acf_lst = None
            
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchTST_multi_MoE': PatchTST_multi_MoE,
            'PatchTST_head_MoE': PatchTST_head_MoE,
            'PatchTST_weighted_avg': PatchTST_weighted_avg,
            'PatchTST_weighted_concat': PatchTST_weighted_concat,
            'PatchTST_weighted_pred_layer_avg': PatchTST_weighted_pred_layer_avg,
            'PatchTST_weighted_concat_no_constrain': PatchTST_weighted_concat_no_constrain,
            'PatchTST_pred_layer_avg': PatchTST_pred_layer_avg,
            'PatchTST_attn_weighted': PatchTST_attn_weighted,
            'PatchTST_attn_weight_global': PatchTST_attn_weight_global,
            'PatchTST_attn_weight_global_indiv': PatchTST_attn_weight_global_indiv,
            'PatchTST_attn_weight_corr_dmodel_indiv': PatchTST_attn_weight_corr_dmodel_indiv,
            'PatchTST_random_mask': PatchTST_random_mask
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        # 这里只对head部分做微调，所以optim也是一样的
        params_head = []
        for n_m, m in self.model.named_modules():
            # print(n_m)
            linear_layer_name = "head.linear"
            embedding_name = "backbone.W_pos"
            if linear_layer_name in n_m or embedding_name in n_m:
                for n_p, p in m.named_parameters():
                    params_head.append(p)
        model_optim = optim.Adam(params_head, lr=self.args.learning_rate)
        
        # lr_multiple = 100
        # # 将模型的参数分组
        # params_main = []
        # params_special = []
        # for name, param in self.model.named_parameters():
        #     if "w_s" in name:  # 这里假设特殊参数有特定的名字标识
        #         params_special.append(param)
        #         print("special:", name)
        #     else:
        #         params_main.append(param)
        #         print("main:", name)
        # model_optim = optim.Adam([{"params": params_main, 'lr': self.args.learning_rate},
        #                           {"params": params_special, 'lr': lr_multiple * self.args.learning_rate}], 
        #                         )
        # print(model_optim)
        # print("-----")
        
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion        
    
    # 这里forecastability相当于是提前就对全部输入数据计算好了的？
    def get_forecastability(self):
        import pandas as pd
        import statsmodels as sm
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        from sklearn.preprocessing import StandardScaler
        
        dataset_file = os.path.join(self.args.root_path, self.args.data_path)

        df_raw = pd.read_csv(dataset_file)
        
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        if "ETTh" in self.args.data:
            border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif "ETTm" in self.args.data:
            border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]


        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        scaler = StandardScaler()
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)
        data = scaler.transform(df_data.values)
        
        print("data.shape:", data.shape)  # data: [sample_num, channel]

        acf_lst = []
        for channel in range(data.shape[1]):
            cur_data = data[:, channel]
            # * 由于我们需要的是seq_len和pred_len之间的ACF，所以最大的lag不会超过seq_len+pred_len
            acf = sm.tsa.stattools.acf(cur_data, nlags=seq_len+pred_len)
            # print(f"acf of channel {channel}: {acf}")
            acf_lst.append(acf)
        
        acf_lst = np.array(acf_lst)
        print("acf_lst.shape:", acf_lst.shape)  # acf_lst: [channel, seq_len+pred_len]
        
        # * 记录为self的成员变量中
        self.acf_lst = acf_lst
        self.args.acf_lst = acf_lst
        
        # return acf_abs[1:nlags+1]
        return acf_lst  # acf_lst: [channel, seq_len+pred_len]
    
    def calc_corr_dmodel(self):
        train_data, train_loader = self._get_data(flag='train')

        mid_embedding_lst = []
        batch_y_lst = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'MoE' in self.args.model:
                    outputs, aux_loss = self.model(batch_x)
                elif 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs, mid_embedding = self.model(batch_x, return_mid_embedding=True)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # pred = outputs.detach().cpu()
                
                # batch_y原来的shape为[bs, pred_len, channel]
                # * 为了和mid_embedding对齐，将其变成[bs, channel, pred_len]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y.detach().cpu()
                batch_y = batch_y.permute(0, 2, 1)
                
                # * mid_emebedding的shape为[bs, channel, patch_num, d_model]
                mid_embedding = mid_embedding.detach().cpu()

                # append到数组上
                batch_y_lst.append(batch_y)
                mid_embedding_lst.append(mid_embedding)
        
        # 先stack、再reshape、最后变成numpy
        # * 此时bs已经变成了sample_num了！！！
        mid_embeddings = torch.stack(mid_embedding_lst)
        batch_ys = torch.stack(batch_y_lst)
        mid_embeddings = mid_embeddings.reshape(-1, mid_embeddings.shape[-3], mid_embeddings.shape[-2], mid_embeddings.shape[-1]).numpy()
        batch_ys = batch_ys.reshape(-1, batch_ys.shape[-2], batch_ys.shape[-1]).numpy()
        
        # 2、计算correlation？
        sample_num, channels, patch_num, d_model = mid_embeddings.shape
        _, _, pred_len = batch_ys.shape
        # 其shape为[channel, patch_num]
        corr_dmodel = np.zeros((channels, patch_num))
        
        # 按照channel顺序和patch舒徐逐一计算correlation
        for cur_channel in range(channels):
            for cur_patch in range(patch_num):
                # 优化原来的两重循环？
                # * solution 2
                x1 = mid_embeddings[:, cur_channel, cur_patch, :]  # [samples, d_model]
                x2 = batch_ys[:, cur_channel, :]  # [samples, pred_len]
                # 由于np.corrcoef算出来是2*2的矩阵，所以只需要从其中取出一个值就可以了
                corr_matrix = np.corrcoef(x1, x2, rowvar=False)
                # print(corr_matrix.shape)  # 得到的结果为[d_model+pred_len, d_model+pred_len]
                # print(corr_matrix)
                # 由于前半段为d_model，后半段为pred_len
                # 那我们分别将其取出就ok了
                cur_corr = np.mean(np.abs(corr_matrix[:d_model, d_model:]))
                corr_dmodel[cur_channel, cur_patch] = cur_corr
                
                # * solution 1
                # cur_corr_lst = []
                # for cur_pred in range(pred_len):
                #     for cur_d in range(d_model):
                #         x1 = mid_embeddings[:, cur_channel, cur_patch, cur_d]
                #         x2 = batch_ys[:, cur_channel, cur_pred]
                #         # 由于np.corrcoef算出来是2*2的矩阵，所以只需要从其中取出一个值就可以了
                #         corr = np.corrcoef(x1, x2)[0, 1]
                #         cur_corr_lst.append(abs(corr))
                # avg_corr = sum(cur_corr_lst) / len(cur_corr_lst)
                # corr_dmodel[cur_channel, cur_patch] = avg_corr
                # # 事实证明，二者确实是相等的
                # print(cur_corr, avg_corr)
        
        # print(corr_dmodel.shape)
        # print(corr_dmodel)
        
        
        self.model.train()
        
        # [channel, patch_num]
        return corr_dmodel
        

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        
        # 获得的总patch数
        patch_num = int((self.args.seq_len - self.args.patch_len) / self.args.stride + 1)
        # 默认是需要做padding的？
        # padding大小事实上只是一个stride的大小！
        if self.args.padding_patch == 'end': # can be modified to general case
            patch_num += 1
            
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'MoE' in self.args.model:
                            outputs, aux_loss = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'random' in self.args.model:
                        import random
                        # validate的时候直接保留全部patches
                        # random_len = 0
                        # idx = random.randint(0, len(self.best_patch_num)-1)
                        unmasked_patch_num = self.best_patch_num[-1]
                        random_len = patch_num - unmasked_patch_num
                        
                        outputs = self.model(batch_x, random_len)
                    elif 'MoE' in self.args.model:
                        outputs, aux_loss = self.model(batch_x)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                # ! 注意：这里应当是loss.item()，而不是直接append上loss自身！
                # ! 应为这样会直接将pytorch的计算图都给存储下来了！！！
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # @profile
    def train(self, setting, acf_lst=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        longest_model_id = self.args.model_id.replace(f"{self.args.seq_len}", f"{self.args.longest_seq_len}", 1)
        longest_setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_fore{}_{}_{}'.format(
                longest_model_id,
                self.args.model,
                self.args.data,
                self.args.features,
                self.args.longest_seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.factor,
                self.args.embed,
                self.args.distil,
                self.args.get_forecastability,
                self.args.des,
                0)
        
        # 先加载一下longest的模型
        # 后面微调的时候只对head部分做微调！
        longest_model_path = os.path.join(self.args.checkpoints, longest_setting)
        longest_model_file = longest_model_path + '/' + 'checkpoint.pth'
        
        longest_model = torch.load(longest_model_file, map_location='cuda:0')
        # print(self.model)
        # 这里要删掉一些一些不匹配的参数，
        # 在这里一个是线性预测层的参数，另一个是embedding层的参数
        # del longest_model['model.backbone.W_P.weight']
        # del longest_model['model.backbone.W_P.bias']
        del longest_model['model.backbone.W_pos']
        del longest_model['model.head.linear.weight']
        del longest_model['model.head.linear.bias']
        self.model.load_state_dict(longest_model, strict=False)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        # 如果需要增加L2正则，那么这里需要修改优化器
        if self.args.add_l2:
            # plan 1: 对所有参数均做正则项
            # 注意：误区! Adam+L2并不能发挥效果! https://zhuanlan.zhihu.com/p/429022216
            # 因此需要改成AdamW优化器，并设置weight_decay
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,
                                      weight_decay=self.args.l2_alpha)
            print("change to AdamW optimizer")

            # # plan 2: 对部分参数施加L2正则：
            # params_all, params_need_L2 = [], []
            # for n_m, m in self.model.named_modules():
            #     # print(n_m)
            #     linear_layer_name = "head.linear"
            #     if linear_layer_name in n_m:
            #         for n_p, p in m.named_parameters():
            #             params_need_L2.append(p)
            #     else:
            #         for n_p, p in m.named_parameters():
            #             params_all.append(p)
            # model_optim = optim.AdamW([
            #         {'params': params_need_L2, 'weight_decay': self.args.l2_alpha},
            #         {'params': params_all, 'weight_decay': 0.0}  # 对部分参数不应用权重衰减
            #     ], lr=self.args.learning_rate)
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        
        # 获得的总Patch数
        patch_num = int((self.args.seq_len - self.args.patch_len) / self.args.stride + 1)
        # 默认是需要做padding的？
        # padding大小事实上只是一个stride的大小！
        if self.args.padding_patch == 'end': # can be modified to general case
            patch_num += 1

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            time_data = 0
            time_forward = 0
            time_backward = 0
            
            # # 测试发现dataloader加载数据的时间会随着epoch增加变得越来越长
            # # ! 后来发现是用visualizer做注意的可视化时存储下来的内容导致的内存泄露
            # time_0 = time.time()
            # for _, (_, _, _, _) in enumerate(train_loader):
            #     a = 0
            #     a += 1
            # print("enumerating time:", time.time() - time_0)

            # 这里微调的时候应该是train还是eval？
            # self.model.train()
            self.model.eval()
            
            # 这里只对head部分做微调，所以optim也是一样的
            params_head = []
            names_head = []
            for n_m, m in self.model.named_modules():
                # print(n_m)
                linear_layer_name = "head.linear"
                embedding_name = "backbone.W_pos"
                if linear_layer_name in n_m or embedding_name in n_m:
                    m.requires_grad_(True)
                    for n_p, p in m.named_parameters():
                        params_head.append(p)
                        names_head.append(f"{n_m}.{n_p}")
            model_optim = optim.Adam(params_head, lr=self.args.learning_rate)
            
            
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                time_before_get_data = time.time()
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                time_before_model = time.time()
                time_data += (time_before_model - time_before_get_data)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'MoE' in self.args.model:
                            outputs, aux_loss = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'random' in self.args.model:
                        import random
                        # 至少保留一个patch？
                        # 又由于最后一个patch一般是padding出来的，所以还是希望至少保留2个patch会更好。
                        # random_len = random.randint(0, patch_num-2)
                        
                        # random_len = patch_num - 42  # 改为固定只使用最后的42个patch的数据，看看会变好吗
                        
                        idx = random.randint(0, len(self.best_patch_num)-1)
                        unmasked_patch_num = self.best_patch_num[idx]
                        random_len = patch_num - unmasked_patch_num
                        
                        outputs = self.model(batch_x, random_len)
                    elif 'MoE' in self.args.model:
                        outputs, aux_loss = self.model(batch_x)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    
                    time_after_model = time.time()
                    time_forward += (time_after_model - time_before_model)
                    
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    # # 定义一下L2正则的表达式
                    # def l2_regularization(model, l2_alpha):
                    #     l2_loss = []
                    #     for n_m, m in model.named_modules():
                    #         # print(n_m)
                    #         linear_layer_name = "head.linear"
                    #         if linear_layer_name in n_m:
                    #             l2_loss.append((m.weight ** 2).sum() / 2.0)
                    #     return l2_alpha * sum(l2_loss)
                    
                    # 计算loss
                    loss = criterion(outputs, batch_y)
                    
                    # # 增加L2正则
                    # if self.args.add_l2:
                    #     l2_loss = l2_regularization(self.model, self.args.l2_alpha)
                    #     # print(f"original_loss={loss}, l2_loss={l2_loss}")
                    #     loss = loss + l2_loss
                    
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                    # 如果是PatchTST_attn_weight_global
                    # 首先从子模块中取出掩码矩阵对应的可学习参数w_s
                    # 其维度为[1, patch_num]
                    if self.args.model == "PatchTST_attn_weight_global":
                        device = self._acquire_device()
                        w_s = self.model.model.backbone.encoder.w_s.to(device)
                        print(w_s.grad.shape)
                        print(w_s.grad)
                    elif self.args.model == "PatchTST_attn_weight_global_indiv" or self.args.model == "PatchTST_attn_weight_corr_dmodel_indiv":
                        device = self._acquire_device()
                        channel_num = -1
                                                
                        v_l = self.model.model.backbone.encoder.v_l.to(device)
                        L_n = self.model.model.backbone.encoder.L_n.to(device)
                        new_acf_lst = self.model.model.new_acf_lst.float().to(device)
                        w_s_lst = self.model.model.backbone.encoder.w_s_lst.to(device)
                        # (2)然后对权重矩阵乘上forecastability先验
                        w_s_weights = [w_s.weight.clone() for w_s in w_s_lst]
                        w_s_weights = [torch.mul(w_s, new_acf_lst[idx, :]) for idx, w_s in enumerate(w_s_weights)]
                        # (2).2同时计算梯度
                        w_s_grads = [w_s.weight.grad for w_s in w_s_lst]
                        # (3)然后将乘上forecastability先验后的中间向量乘上来
                        # ! 别忘记这里需要对self.w_s做softmax！！！
                        import torch.nn.functional as F
                        mask_global = [torch.mm(v_l, F.softmax(w_s)) for w_s in w_s_weights]  # mask_global: nvars个[patch_num, patch_num]
                        print("1.w_s_weights[-1]:", w_s_weights[-1])
                        print("2.F.softmax(w_s_weights[-1]):", F.softmax(w_s_weights[-1]))
                        print("3.torch.mm(v_l, F.softmax(w_s_weights[-1])):", torch.mm(v_l, F.softmax(w_s_weights[-1])))
                        print("mask_global_1[-1, 0, :]:", mask_global[channel_num][0, :])
                        mask_global = [torch.mm(tmp_mask, L_n) for tmp_mask in mask_global]  # mask_global: nvars个[patch_num, patch_num] or nvars个[max_q_len, q_len]
                        mask_global = torch.stack(mask_global)  # 默认为float, shape为[nvars x patch_num x patch_num]
                        print("new_acf_lst[-1]:", new_acf_lst[channel_num])
                        print("mask_global[-1, 0, :]:", mask_global[channel_num, 0, :])
                        # (4) 打印梯度
                        # print(len(w_s_lst))
                        print(w_s_grads[0].shape)
                        print("w_s_weights[-1]:", w_s_weights[channel_num])
                        print("w_s_grads[-1]:", w_s_grads[channel_num])
                        
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
                
                time_after_backward = time.time()
                time_backward += (time_after_backward - time_after_model)
            

            # # 打印出运行和前向&后向消耗的时间？
            # print("time_data:", time_data)
            # print("time_forward:", time_forward)
            # print("time_backward:", time_backward)
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            time_before_vali = time.time()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            # 打印出验证集消耗的时间
            # print("time_vali", time.time() - time_before_vali)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                
            # ! 这个函数应当是在每个epoch结束时就需要调用一次
            if self.args.model == "PatchTST_attn_weight_corr_dmodel_indiv":
                # 其shape为[channel, patch_num]
                corr_dmodel = self.calc_corr_dmodel()
                corr_dmodel = torch.Tensor(corr_dmodel).float()
                if epoch % 10 == 0:
                    print(corr_dmodel.shape)
                    print(corr_dmodel)
                # 直接更新到new_acf_lst上
                self.model.model.new_acf_lst = corr_dmodel

        # 这里为了防止异常，需要做一些修改，要在torch.load后加上map_location='cuda:0'
        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            # 这里为了防止异常，需要做一些修改，要在torch.load后加上map_location='cuda:0'
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        # 如果是PatchTST_attn_weight_global
        # 首先从子模块中取出掩码矩阵对应的可学习参数w_s
        # 其维度为[1, patch_num]
        if self.args.model == "PatchTST_attn_weight_global":
            # print(self.model)
            device = self._acquire_device()
            v_l = self.model.model.backbone.encoder.v_l.to(device)
            w_s = self.model.model.backbone.encoder.w_s.to(device)
            L_n = self.model.model.backbone.encoder.L_n.to(device)
            # ! 别忘记这里需要对self.w_s做softmax！！！
            import torch.nn.functional as F
            mask_global = torch.mm(v_l, F.softmax(w_s.weight))
            mask_global = torch.mm(mask_global, L_n)  # mask: [patch_num, patch_num] or [max_q_len, q_len]
            print(mask_global[0])
        elif self.args.model == "PatchTST_attn_weight_global_indiv" or self.args.model == "PatchTST_attn_weight_corr_dmodel_indiv":
            device = self._acquire_device()
            v_l = self.model.model.backbone.encoder.v_l.to(device)
            w_s_lst = self.model.model.backbone.encoder.w_s_lst.to(device)
            L_n = self.model.model.backbone.encoder.L_n.to(device)
            new_acf_lst = self.model.model.new_acf_lst.float().to(device)
            # (2)然后对权重矩阵乘上forecastability先验
            w_s_weights = [w_s.weight.clone() for w_s in w_s_lst]
            w_s_weights = [torch.mul(w_s, new_acf_lst[idx, :]) for idx, w_s in enumerate(w_s_weights)]
            # (3)然后将乘上forecastability先验后的中间向量乘上来
            # ! 别忘记这里需要对self.w_s做softmax！！！
            import torch.nn.functional as F
            mask_global = [torch.mm(v_l, F.softmax(w_s)) for w_s in w_s_weights]  # mask_global: nvars个[patch_num, patch_num]
            mask_global = [torch.mm(tmp_mask, L_n) for tmp_mask in mask_global]  # mask_global: nvars个[patch_num, patch_num] or nvars个[max_q_len, q_len]
            mask_global = torch.stack(mask_global)  # 默认为float, shape为[nvars x patch_num x patch_num]
            # print("new_acf_lst:", new_acf_lst)
            # print("mask_global[:, 0, :]:", mask_global[:, 0, :])
            


        preds = []
        trues = []
        inputx = []
        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # # 打印一下模型中的权重参数
        # if "weighted" in self.args.model:
        #     self.model.eval()
        #     print("Mask Weight:")
        #     print(self.model.model.weight_mask.mask_weight)
        #     mask = self.model.model.weight_mask.mask_weight.weight
        #     segment_len = self.args.d_model
        #     # segment_len = self.args.d_model if "concat" in self.args.model else self.args.d_pred
        #     # 打印gate和mask等的相关的信息
        #     torch.set_printoptions(profile="full")
        #     # print(f"gate in WeightMask: {gate}")
        #     mask_for_print = mask[:, :, ::segment_len]  # 由于多个channel共享相同的权重值，所以间隔着只取出第一个就ok了
        #     # mask_for_print = mask[:, :, :]  # 由于多个channel共享相同的权重值，所以间隔着只取出第一个就ok了
        #     mask_for_print = mask_for_print.mean(dim=0)  # 对当前batch的所有样本取平均值？
        #     print(f"mask_for_print in WeightMask: {mask_for_print}")
        #     # print(mask.shape)
        #     torch.set_printoptions(profile="default") # reset
        
        # 获得的总patch数
        patch_num = int((self.args.seq_len - self.args.patch_len) / self.args.stride + 1)
        # 默认是需要做padding的？
        # padding大小事实上只是一个stride的大小！
        if self.args.padding_patch == 'end': # can be modified to general case
            patch_num += 1

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'MoE' in self.args.model:
                            outputs, aux_loss = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'random' in self.args.model:
                        import random
                        # test的时候也是直接保留全部patches
                        # random_len = 0
                        # idx = random.randint(0, len(self.best_patch_num)-1)
                        unmasked_patch_num = self.best_patch_num[-1]
                        random_len = patch_num - unmasked_patch_num
                        
                        outputs = self.model(batch_x, random_len)
                    elif 'MoE' in self.args.model:
                        outputs, aux_loss = self.model(batch_x)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        
        print("preds.shape:", preds.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        
        # print(mae, mse, rmse, mape, mspe, rse, corr)

        # # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # # np.save(folder_path + 'x.npy', inputx)
        
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            # 这里为了防止异常，需要做一些修改，要在torch.load后加上map_location='cuda:0'
            # self.model.load_state_dict(torch.load(best_model_path))
            self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'MoE' in self.args.model:
                            outputs, aux_loss = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'MoE' in self.args.model:
                        outputs, aux_loss = self.model(batch_x)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
