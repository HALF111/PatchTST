import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler

# 设置超参
NLAGS = 200
cur_channel = 7

# 读数据
folder = "../dataset"
fname = "ETTh1.csv"
file_name = f"{folder}/{fname}"

data_df = pd.read_csv(file_name)
# 去掉第一列date列
cols_data = data_df.columns[1:]
data_df = data_df[cols_data]

data_cols = data_df.columns
print("data_cols:", data_cols)

# 获取当前列
cur_col = data_cols[cur_channel-1]
print("cur_col:", cur_col)

selected_series = data_df[cur_col].values

# 做归一化？
split_ratio = 0.6 if "ETT" in fname else 0.7
num_train = int(len(selected_series) * split_ratio)
scaler = StandardScaler()
scaler.fit(selected_series[:num_train].reshape(-1, 1))
selected_series = scaler.transform(selected_series.reshape(-1, 1))


# acf, confint, qstat, pvalues = sm.tsa.stattools.acf(price_percentage)
acf = sm.tsa.stattools.acf(selected_series, nlags=NLAGS)
print(f"acf: {acf}")
# print(f"confint: {confint}")
# print(f"qstat: {qstat}")
# print(f"pvalues: {pvalues}")

# # 绝对值
# acf_abs = abs(acf)
# print(f"acf_abs: {acf_abs}")

# # 取出top-k峰值对应的lag
# k = 10
# top_indices = np.argsort(acf_abs)[-k:][::-1]
# print(f"top_indices: {top_indices}")

# 画图
plot_acf(selected_series, lags=NLAGS, title=f"{fname.split('.')[0]}_ACF_channel{cur_channel:02d}")

file_name = f"./{fname.split('.')[0]}_channel{cur_channel:02d}.pdf"
plt.savefig(file_name)