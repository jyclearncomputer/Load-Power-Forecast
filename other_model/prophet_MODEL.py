import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'D:\F\电网预测\数据集\daily data.csv').iloc[50:]
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Prepare the data for Prophet
df = data.reset_index().rename(columns={'date': 'ds', 'total_electricity': 'y'})

# Split the data into training and testing sets (80:20 split)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Fit the Prophet model
model = Prophet()
model.fit(train_df)

# Make predictions
future = model.make_future_dataframe(periods=len(test_df), freq='D')
forecast = model.predict(future)

# Extract predictions for the test period
predictions_df = forecast.iloc[train_size:][['ds', 'yhat']]

# Merge predictions with actual test values for evaluation
eval_df = test_df[['ds', 'y']].merge(predictions_df, on='ds', how='left')

# Calculate evaluation metrics
mse = mean_squared_error(eval_df['y'], eval_df['yhat'])
mae = mean_absolute_error(eval_df['y'], eval_df['yhat'])
mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'])) * 100
r2 = 1 - (np.sum((eval_df['y'] - eval_df['yhat'])**2) / np.sum((eval_df['y'] - np.mean(eval_df['y']))**2))

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}%')
print(f'R²: {r2}')

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(eval_df['ds'], eval_df['y'], label='Actual Data', marker='o', linestyle='-')
plt.plot(eval_df['ds'], eval_df['yhat'], label='Predicted Data', marker='x', linestyle='--')
plt.title('Prophet Model Predictions')
plt.xlabel('Date')
plt.ylabel('Total Electricity')
plt.legend()
plt.show()

# Plot components
model.plot_components(forecast)
plt.show()