import torch.nn as nn
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from iTransformerFFT import iTransformerFFT
from datetime import datetime
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuantileLoss(nn.Module):
    def __init__(self, tau):
        super(QuantileLoss, self).__init__()
        self.tau = tau

    def forward(self, y_pred, y_true):
        delta = y_true - y_pred
        loss = torch.max((self.tau - 1) * delta, self.tau * delta)
        return torch.mean(loss)


# Step 1: Load and prepare data
def load_data(file_path):
    """
    Load data from a CSV file and convert the date column to datetime.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame with loaded data.
    """
    try:
        data = pd.read_csv(file_path).iloc[50:]
        data['date'] = pd.to_datetime(data['date'])
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


# Step 2: Preprocess data
def preprocess_data(data, lookback_len):
    """
    Preprocess data by scaling power data and creating sequences with corresponding targets.

    Parameters:
    - data (pd.DataFrame): DataFrame with the data.
    - lookback_len (int): Number of previous time steps to use for prediction.

    Returns:
    - torch.Tensor: Tensor of sequences.
    - torch.Tensor: Tensor of targets.
    - StandardScaler: Scaler fitted on the power data.
    """
    scaler = StandardScaler()
    data['total_electricity'] = scaler.fit_transform(data[['total_electricity']])  # Scale power data

    sequences = []
    targets = []
    for i in range(len(data) - lookback_len):
        sequence = data.iloc[i:i + lookback_len][['total_electricity']].values
        target = data.iloc[i + lookback_len]['total_electricity']  # Target is the day after the sequence
        sequences.append(sequence)
        targets.append(target)

    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), scaler


# Step 3: Initialize the model
def initialize_model(num_variates, lookback_len, num_tokens_per_variate, depth, dim, pred_length, dim_head, heads ,lstm_hidden_dim):
    """
    Initialize the iTransformer2D model with provided parameters.

    Parameters:
    - num_variates (int): Number of input variates.
    - lookback_len (int): Length of the lookback window.
    - num_time_tokens (int): Number of time tokens.
    - depth (int): Depth of the model.
    - dim (int): Dimension of the model.
    - pred_length (int): Length of the prediction window.
    - dim_head (int): Dimension of each head.
    - heads (int): Number of heads.

    Returns:
    - iTransformer2D: Initialized model.
    """
    model = iTransformerFFT(
        num_variates=num_variates,
        lookback_len=lookback_len,
        num_tokens_per_variate=num_tokens_per_variate,
        depth=depth,
        dim=dim,
        pred_length=pred_length,
        dim_head=dim_head,
        heads=heads,
        use_reversible_instance_norm=True,
        lstm_hidden_dim= lstm_hidden_dim
        # use reversible instance normalization, proposed here https://openreview.net/forum?id=cGDAkQo1C0p . may be redundant given the layernorms within iTransformer (and whatever else attention learns emergently on the first layer, prior to the first layernorm). if i come across some time, i'll gather up all the statistics across variates, project them, and condition the transformer a bit further. that makes more sense
    )
    return model


# Step 4: Train the model
def train_model(model, train_data, train_targets, epochs, lr):
    """
    Train the model using Mean Squared Error loss and Adam optimizer.

    Parameters:
    - model (iTransformer2D): The model to train.
    - train_data (torch.Tensor): Training data.
    - train_targets (torch.Tensor): Training targets.
    - epochs (int): Number of epochs to train for.
    - lr (float): Learning rate.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    # 定义 Quantile Loss
    tau = 0.37
    quantile_loss = QuantileLoss(tau)
    # 定义优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []

    model.train()
    for epoch in range(epochs):
        predictions = model(train_data)

        # Check if predictions is a dictionary and extract the relevant output
        if isinstance(predictions, dict):
            first_key = list(predictions.keys())[0]
            predictions = predictions.get(first_key, None)
            if predictions is None:
                raise ValueError(f"Expected key '{first_key}' not found in model predictions")

        # Extract the relevant feature from the predictions
        predictions = predictions[:, -1, 0]  # Use the last time step's power prediction

        # loss = quantile_loss(predictions, train_targets)
        loss = loss_fn(predictions, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        logging.info(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Save the trained model
    # torch.save(model.state_dict(), r"D:\F\电网预测\iTransformer-main\model\itransformer_modelFFT40_10.pth")

    return train_losses


# Step 5: Evaluate the model
def evaluate_model(model, test_data, test_targets, scaler):
    """
    Evaluate the model using the test data.

    Parameters:
    - model (iTransformer2D): The trained model.
    - test_data (torch.Tensor): Testing data.
    - test_targets (torch.Tensor): Testing targets.
    - scaler (StandardScaler): Scaler fitted on the power data.

    Returns:
    - torch.Tensor: Predictions made by the model.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
        if isinstance(predictions, dict):
            first_key = list(predictions.keys())[0]
            predictions = predictions[first_key]
        predictions = predictions[:, -1, 0]  # Use the last time step's power prediction

    # Calculate metrics
    test_targets = test_targets.numpy()
    predictions = predictions.numpy()

    test_targets = scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    mse = mean_squared_error(test_targets, predictions)
    mae = mean_absolute_error(test_targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_targets, predictions)
    mape = np.mean(np.abs((test_targets - predictions) / test_targets)) * 1

    logging.info(f"Test MSE: {mse}")
    logging.info(f"Test MAE: {mae}")
    logging.info(f"Test RMSE: {rmse}")
    logging.info(f"Test R2: {r2}")
    logging.info(f"Test MAPE: {mape}")

    return predictions,test_targets


# Step 6: Visualize predictions
def visualize_predictions(test_targets, predictions, scaler):
    """
    Plot actual vs. predicted values of power.

    Parameters:
    - test_targets (torch.Tensor): Actual values.
    - predictions (torch.Tensor): Predictions made by the model.
    - scaler (StandardScaler): Scaler fitted on the power data.
    """
    actual = test_targets
    predicted = predictions

    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Power')
    plt.plot(predicted, label='Predicted Power')
    plt.legend()
    plt.show()


# Step 7: Plot training loss
def plot_training_loss(train_losses):
    """
    Plot training loss over epochs.

    Parameters:
    - train_losses (list): List of training loss values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()



# Example usage
if __name__ == "__main__":
    file_path = r"D:\F\电网预测\数据集\daily data.csv"  # Replace with your data file path
    lookback_len = 10 # Example lookback length
    num_variates = 1  # Only power data3
    num_tokens_per_variate = 3  # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
    depth = 6
    dim = 256
    dim_head = 32
    heads = 8
    pred_length = 1  # Example prediction length
    lstm_hidden_dim = 128
    lr = 0.001
    epochs = 50

    data = load_data(file_path)
    sequences, targets, scaler = preprocess_data(data, lookback_len)

    # Split data into train and test sets
    train_data, test_data, train_targets, test_targets = train_test_split(
        sequences, targets, test_size=0.2, random_state=42, shuffle=False)


    model = initialize_model(
        num_variates=num_variates,
        lookback_len=lookback_len,
        num_tokens_per_variate=num_tokens_per_variate,
        depth=depth,
        dim=dim,
        pred_length=pred_length,
        dim_head=dim_head,
        heads=heads,
        lstm_hidden_dim=lstm_hidden_dim
    )
    #
    train_losses = train_model(model, train_data, train_targets, epochs,lr)
    # plot_training_loss(train_losses)

    # Load the saved model for evaluation and prediction
    # model.load_state_dict(torch.load(r"D:\F\电网预测\iTransformer-main\model\itransformer_modelFFT40_10.pth"))


    predictions, test_targets = evaluate_model(model, test_data, test_targets, scaler)
    visualize_predictions(test_targets, predictions, scaler)
