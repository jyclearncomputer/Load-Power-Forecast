import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.layers import Conv1D, Activation, Dropout, Dense, LSTM, Flatten, MaxPooling1D
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 示例数据
data = pd.read_csv(r'D:\F\电网预测\数据集\daily data.csv').iloc[50:]
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
values = df['total_electricity'].values.reshape(-1,1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

window_size = 13
# 创建数据集
def create_dataset(dataset, window_size=window_size):
    X, Y = [], []
    for i in range(len(dataset)-window_size):
        X.append(dataset[i:(i+window_size), 0])
        Y.append(dataset[i+window_size, 0])
    return np.array(X), np.array(Y)

# 创建数据集
X, Y = create_dataset(scaled_data, window_size=window_size)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# 重塑输入数据以适应CNN-LSTM模型
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# 构建CNN-LSTM模型
model = Sequential()
model.add(Conv1D(filters=512, kernel_size=1, activation='relu', input_shape=(1, window_size)))
model.add(MaxPooling1D(pool_size=1))
# model.add(Dropout(0.2))
model.add(LSTM(64, input_shape=(1, window_size), return_sequences=True))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1))

# 查看模型结构
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# 训练模型
model.fit(X_train, Y_train, epochs=200, batch_size=100, verbose=1)

# 进行预测
Y_pred = model.predict(X_test)

# 反归一化预测结果和实际值
Y_pred = scaler.inverse_transform(Y_pred)
Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))

# 计算评估指标
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
mape = mean_absolute_percentage_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

# 输出评估指标
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"RMSE: {rmse:.4f}")

index = range(len(Y_test))
# 画测试集的实际值
plt.figure(figsize=(10,6))
plt.plot(index, Y_test,  label='Testing Data')
# 画预测值
plt.plot(index, Y_pred, label='Predicted Value')
plt.show()

#将预测数据保存
df_pred = pd.DataFrame(Y_pred)
df_pred.to_csv(r'D:\F\电网预测\结果图\对比实验\预测数据\CNN-LSTM.csv')