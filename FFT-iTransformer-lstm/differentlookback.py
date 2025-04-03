import pandas as pd

from ceemdan_ffttransformerlstm_modified import run_differentlookback
import matplotlib.pyplot as plt

lookback_len=[5,10,20,30,40,50,60,70,80,90,100]
error_list=[]
for i in lookback_len:
    temp_list=[]
    r2, mape,rmse=run_differentlookback(i)
    temp_list.append(r2)
    temp_list.append(mape)
    temp_list.append(rmse)
    error_list.append(temp_list)

df=pd.DataFrame(error_list)
df.to_csv(r"D:\F\电网预测\结果图\different lookback.csv", index=False)