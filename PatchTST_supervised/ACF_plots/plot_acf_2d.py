import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler

# 定义多变量之间ACF的函数：
def multivariate_autocorr(matrix, lag):
    # matrix为[timesteps, channel]的输入数据
    n, m = matrix.shape
    means = np.mean(matrix, axis=0)
    variances = np.var(matrix, axis=0)

    autocov_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            autocov_matrix[i, j] = np.sum((matrix[:n-lag, i] - means[i]) * (matrix[lag:, j] - means[j])) / (n - lag)

    autocorr_matrix = autocov_matrix / variances.reshape(1, -1)

    return autocorr_matrix


# 设置超参
NLAGS = 100


# 遍历所有数据集
for dataset_name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "national_illness", "weather", "traffic", "electricity"]:
    # 读数据
    folder = "../dataset"
    fname = f"{dataset_name}.csv"
    file_name = f"{folder}/{fname}"
    
    if dataset_name in ["ETTm1", "ETTm2", "weather", "national_illness"]:
        NLAGS = 200
    else:
        NLAGS = 100

    data_df = pd.read_csv(file_name)
    # 去掉第一列date列
    cols_data = data_df.columns[1:]
    data_df = data_df[cols_data]

    # 统计总列数
    data_cols = data_df.columns
    print("data_cols:", data_cols)
    total_channels = len(data_cols)

    # 获取整个序列数据为numpy格式
    data_matrix = data_df.values
    
    # 限制总channel数量不超过10
    THRESHOLD = 8
    if total_channels >= THRESHOLD:
        data_cols = data_cols[:THRESHOLD]
        total_channels = THRESHOLD
        data_matrix = data_matrix[:, :THRESHOLD]
        

    # 做归一化？
    split_ratio = 0.6 if "ETT" in fname else 0.7
    num_train = int(len(data_matrix) * split_ratio)
    scaler = StandardScaler()
    scaler.fit(data_matrix[:num_train])
    data_matrix = scaler.transform(data_matrix)


    acf_matrix_lst = []
    # 计算滞后为1的ACF值矩阵
    for lag in range(1, NLAGS+1):
        lag_acf_matrix = multivariate_autocorr(data_matrix, lag)
        print(lag_acf_matrix.shape)
        print(f"Lag {lag} ACF Matrix:\n{lag_acf_matrix}")
        acf_matrix_lst.append(lag_acf_matrix.flatten())

    acf_matrix_lst = np.array(acf_matrix_lst)
    acf_matrix_lst = acf_matrix_lst.T


    def visualize_acf(weight_map, channels, figure_name):
        matplotlib.rcParams.update({'font.size': 7}) # 改变所有字体大小，改变其他性质类似
        # plt.figure()
        # ax = plt.gca()
        fig, axes = plt.subplots(channels, 1)
        # Plot the heatmap
        for i in range(channels):
            im = axes[i].imshow(weight_map[i*channels:(i+1)*channels])
            # Create colorbar
            cbar = axes[i].figure.colorbar(im, ax=axes[i])
        
        print("Saving figures...")
        plt.savefig(figure_name)
        plt.show()

    # 画图展示
    visualize_acf(weight_map=acf_matrix_lst, channels=total_channels, figure_name=f"./{fname.split('.')[0]}_2d.pdf")