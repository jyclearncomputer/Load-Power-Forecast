from ceemdan_ffttransformerlstm_modified import run_differentlookback
import pandas as pd

#设定超参数值
lookback_len=[5,7,9,11,13,15,20,25,30,50]
epochs=[10,20,30,40,50,60,70,80]
lr=[0.0001,0.0005,0.001,0.005,0.01]

#创建三个空参列表，用于存储误差、lookback_len、epochs、lr
lookbacklen_error = []
epochs_error = []
lr_error = []


# for i in lookback_len:
#     temp_list=[]
#     r2, mape,rmse=run_differentlookback(i,50,0.001)
#     temp_list.append(r2)
#     temp_list.append(mape)
#     temp_list.append(rmse)
#     lookbacklen_error.append(temp_list)

#将lookbacklen_error列表转换为DataFrame，
# df=pd.DataFrame(lookbacklen_error)
# #存储到csv文件中,列名为r2, mape,rmse
# df.to_csv(r"D:\F\电网预测\结果图\误差列表\lookbacklen_error.csv")

# for i in epochs:
#     temp_list=[]
#     r2, mape,rmse=run_differentlookback(13,i,0.001)
#     temp_list.append(r2)
#     temp_list.append(mape)
#     temp_list.append(rmse)
#     epochs_error.append(temp_list)
#
# #将lookbacklen_error列表转换为DataFrame，
# df=pd.DataFrame(epochs_error)
# #存储到csv文件中,列名为r2, mape,rmse
# df.to_csv(r"D:\F\电网预测\结果图\误差列表\epochs_error.csv")

for i in lr:
    temp_list=[]
    r2, mape,rmse=run_differentlookback(13,50,i)
    temp_list.append(r2)
    temp_list.append(mape)
    temp_list.append(rmse)
    lr_error.append(temp_list)

#将lookbacklen_error列表转换为DataFrame，
df=pd.DataFrame(lr_error)
#存储到csv文件中,列名为r2, mape,rmse
df.to_csv(r"D:\F\电网预测\结果图\误差列表\lr_error.csv")
