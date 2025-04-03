import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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

# 将数据转换为张量并进行 reshape，符合 LSTM 输入格式： (batch_size, seq_len, input_size)
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, window_size, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, window_size, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 网络
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏层状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化记忆状态

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))  # out 的形状: (batch_size, seq_length, hidden_size)

        # 取最后时间步的输出
        out = self.fc(out[:, -1, :])  # 取最后时间步的 hidden state
        return out
# 初始化 LSTM 模型参数
input_size = 1
hidden_size = 128  # 隐藏层大小
num_layers = 2    # LSTM 层数
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model = model.to(torch.device('cpu'))  # 如果有 GPU，替换为 torch.device('cuda')

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# 训练模型
def train(model, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # 梯度归零
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}')

# 训练 100 个 epochs
train(model, train_loader, num_epochs=200)
# 模型评估
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train).detach().numpy()
    y_test_pred = model(X_test).detach().numpy()

# 反归一化预测结果
y_train_pred_inv = scaler_y.inverse_transform(y_train_pred)
y_test_pred_inv = scaler_y.inverse_transform(y_test_pred)
y_train_inv = scaler_y.inverse_transform(y_train.numpy().reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1))

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
plt.title('LSTM Prediction vs Actual Data')
plt.legend()
plt.show()

#将预测数据保存下来
predictions = pd.DataFrame(y_test_pred_inv, index=data.index[len(y_train_pred) + window_size:], columns=['Prediction'])
real = pd.DataFrame(y_test_inv, index=data.index[len(y_train_pred) + window_size:], columns=['Real'])
result = pd.concat([real, predictions], axis=1)
result.to_csv(r"D:\F\电网预测\结果图\对比实验\预测数据\lstm_result.csv")
