import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv(r'D:\F\电网预测\数据集\daily data.csv').iloc[50:]
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Split the data into training and testing sets (80:20 split)
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]



# Initialize a list to hold predictions
predictions = []

# Set the initial training data
train_data = train.copy()

# Iterate over each time step in the test set
for i in range(len(test)):
    # Fit the ARIMA model
    model = ARIMA(train_data, order=(3,0,3))
    model_fit = model.fit()

    # Forecast the next value
    forecast = model_fit.forecast(steps=1)

    # Append the forecast to the predictions list
    predictions.append(forecast.values)
    print(forecast.values)
    new_row = pd.DataFrame({'total_electricity': test.iloc[i]['total_electricity']},
                           index=[test.index[i]])
    train_data = pd.concat([train_data, new_row])

# Convert predictions to a DataFrame for easier evaluation
predictions = pd.DataFrame(predictions, index=test.index, columns=['Predicted'])

# Evaluate the predictions
real = np.array(test)
pred = np.array(predictions)

mse = mean_squared_error(real, pred)
mae = mean_absolute_error(real, pred)
r2 = r2_score(real, pred)
MAPE = np.mean(np.abs((real - pred) / real))
rmse = np.sqrt(mse)

# Plot the predictions
plt.figure(figsize=(14, 7))
plt.plot(test.index, real, label='Actual Data')
plt.plot(test.index, pred, label='Predicted Data', linestyle='--')
plt.title('ARIMA Model Moving Window Predictions')
plt.xlabel('Date')
plt.ylabel('Total Electricity')
plt.legend()
plt.show()

#将预测数据和一起保存下来
result = pd.concat([test, predictions], axis=1)
result.to_csv(r"D:\F\电网预测\结果图\对比实验\预测数据\arima.csv")

print(f'MSE: {mse}, MAE: {mae}, R²: {r2}, MAPE: {MAPE}, RMSE: {rmse}')

