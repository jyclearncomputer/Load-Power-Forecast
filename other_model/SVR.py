import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

# 加载数据
data = pd.read_csv(r'D:\F\电网预测\数据集\daily data.csv').iloc[50:]  # 请将电力数据.csv 替换为实际数据集路径
data.columns=["date","elec"]
# 数据预处理（假设数据有 'date' 和 'elec' 列）
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# 电力消耗数据作为目标值
y = data['elec'].values  # 电力消耗值

# 滑动窗口函数，生成特征和目标
def create_features_and_target(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])  # 用之前的 window_size 个值作为特征
        y.append(data[i + window_size])    # 用当前时间步的值作为目标
    return np.array(X), np.array(y)

# 设置滑动窗口大小
window_size = 10  # 例如使用前5个时间步的电力消耗预测下一时间步

# 使用滑动窗口生成特征和目标值
X, y = create_features_and_target(y, window_size)

# 归一化特征和目标值
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# 创建 SVR 模型，使用径向基函数（RBF）核
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)

# 训练 SVR 模型
svr_model.fit(X_train, y_train)

# 预测
y_train_pred = svr_model.predict(X_train)
y_test_pred = svr_model.predict(X_test)

# 反归一化预测结果
y_train_pred_inv = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
y_test_pred_inv = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 计算评价指标
train_mse = mean_squared_error(y_train_inv, y_train_pred_inv)
test_mse = mean_squared_error(y_test_inv, y_test_pred_inv)

real = np.array(y_test_inv)
pred = np.array(y_test_pred_inv)

mse = mean_squared_error(real, pred)
mae = mean_absolute_error(real, pred)
r2 = r2_score(real, pred)
MAPE = np.mean(np.abs((real-pred)/real))
rmse = np.sqrt(mse)


print(f'MSE: {mse}, MAE: {mae}, R²: {r2}, MAPE: {MAPE}, RMSE: {rmse}')


# 可视化预测结果
plt.figure(figsize=(10, 6))

# 绘制实际电力消耗数据
# plt.plot(data.index[window_size:], scaler_y.inverse_transform(y_scaled.reshape(-1, 1)), label='Actual Data', color='blue')


# 绘制训练集的预测结果
# plt.plot(data.index[window_size:len(y_train_pred) + window_size], y_train_pred_inv, label='Train Prediction', color='green')

# 绘制测试集的预测结果
plt.plot(data.index[len(y_train_pred) + window_size:], y_test_pred_inv, label='Test Prediction')
plt.plot(data.index[len(y_train_pred) + window_size:], y_test_inv, label='Test Data')

plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.title('SVR Prediction vs Actual Data')
plt.legend()
plt.show()

#将预测结果保存
pred_df = pd.DataFrame({'Date': data.index[len(y_train_pred) + window_size:], 'Predicted': y_test_pred_inv})
real_df = pd.DataFrame({'Date': data.index[len(y_train_pred) + window_size:], 'Real': y_test_inv})
result = pd.concat([real_df, pred_df], axis=1)
real_df.to_csv(r'D:\F\电网预测\结果图\对比实验\预测数据\svr.csv', index=False)