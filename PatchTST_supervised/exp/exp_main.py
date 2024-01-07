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
from models import PatchTST_attn_weighted
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

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
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
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # def attn_plot(self, setting):
    #     test_data, test_loader = self._get_data(flag='test')
        
    #     # 这里为了防止异常，需要做一些修改，要在torch.load后加上map_location='cuda:0'
    #     print('loading model')
    #     # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
        
    #     # 跑测试数据？
    #     self.test(setting, test=1)
        
    #     cache = get_local.cache
        
    #     return cache


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
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

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                # ! 注意：这里应当是loss.item()，而不是直接append上loss自身！
                # ! 应为这样会直接将pytorch的计算图都给存储下来了！！！
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    @profile
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            time_data = 0
            time_forward = 0
            time_backward = 0
            
            # 测试发现dataloader加载数据的时间会随着epoch增加变得越来越长
            time_0 = time.time()
            for _, (_, _, _, _) in enumerate(train_loader):
                a = 0
                a += 1
            print("enumerating time:", time.time() - time_0)

            self.model.train()
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
                    if 'MoE' in self.args.model:
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
