import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"D:\F\电网预测\数据集\daily data.csv").iloc[50:]
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Prepare data for RNN
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences
seq_length = 15
X, y = create_sequences(scaled_data, seq_length)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build RNN model
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.1, batch_size=100, epochs=200)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions
predictions_inverse = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test)

# Evaluate the predictions
mse_rnn = mean_squared_error(y_test_inverse, predictions_inverse)
mae_rnn = mean_absolute_error(y_test_inverse, predictions_inverse)
r2_rnn = r2_score(y_test_inverse, predictions_inverse)
rmse = np.sqrt(mse_rnn)
mape = np.mean(np.abs((y_test_inverse - predictions_inverse) / y_test_inverse)) * 1

# Plot the predictions
index = range(len(predictions_inverse))
plt.figure(figsize=(14, 7))
plt.plot(index, y_test_inverse, label='Actual Data')
plt.plot(index, predictions_inverse, label='Predicted Data')
plt.title('RNN Model Predictions')
plt.xlabel('Date')
plt.ylabel('Total Electricity')
plt.legend()
plt.show()

print(f'MSE: {mse_rnn}, MAE: {mae_rnn}, R²: {r2_rnn}, MAPE: {mape}, RMSE: {rmse}')

#将预测数据保存
predictions_inverse = pd.DataFrame(predictions_inverse,columns=['predictions'])
y_test_inverse = pd.DataFrame(y_test_inverse,columns=['real'])
result = pd.concat([y_test_inverse, predictions_inverse], axis=1)
result.to_csv(r"D:\F\电网预测\结果图\对比实验\预测数据\rnn.csv")
