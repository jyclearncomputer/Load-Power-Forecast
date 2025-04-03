"""models needed to create the Neural Nets"""
import torch.nn as nn
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from attention import Attention

"""needed for scaling and calculating the performance of the model"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
import math
"""needed for the decomposition of the time series"""
from PyEMD import CEEMDAN
"""needed for downloading and wrangling data"""
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
import random
import warnings
from keras.models import Sequential
from keras.layers import Conv1D, Activation, Dropout, Dense, LSTM, Flatten, MaxPooling1D
import keras.optimizers

from iTransformerFFT import iTransformerFFT
from run_lstmitransformer import train_model,evaluate_model

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
def error(y_true,y_pred):
    return (y_pred-y_true) /y_true


"""used to stop warnings from showing up in the console"""
warnings.filterwarnings('ignore')
random.seed(88888)

"""CONSTANTS THAT ARE USED THROUGHOUT THE MODEL"""
# Example lookback length
num_variates = 1
num_tokens_per_variate = 3
depth = 6
dim = 256
dim_head = 32
heads = 8
pred_length = 1  # Example prediction length
lstm_hidden_dim = 128
# epoch = 50
# lr = 0.001

SPLIT = 0.8


data=pd.read_csv(r"D:\F\电网预测\数据集\daily data.csv").iloc[:] #读取日用电量数据
data.columns=['date','elec']
# data['elec'].iloc[:int(len(data)*SPLIT)].plot()
data['elec'].plot()
plt.xlabel("day")
plt.ylabel("Power consumption")
plt.legend()
plt.savefig(r"D:\F\电网预测\结果图\对比实验\earlydata.pdf", format='pdf', bbox_inches='tight')
plt.show()

def get_CEEMD_residue(data: pd.DataFrame):
    """
    Complete Ensemble EMD with Adaptive Noise (CEEMDAN) performs an EEMD
    The difference is that the information about the noise is shared among all workers

    :returns:
    IMFs : numpy array
        All the Intrinsic Mode Functions that make up the original stock price
    residue : numpy array
        The residue from the recently analyzed stock price
    """

    data_np = data.to_numpy()

    ceemd = CEEMDAN()
    ceemd.extrema_detection = "parabol"
    ceemd.ceemdan(data_np)
    IMFs, residue = ceemd.get_imfs_and_residue()

    nIMFs = IMFs.shape[0]

    plt.figure(figsize=(18, 12))
    plt.subplot(nIMFs + 2, 1, 1)

    plt.plot(data, 'r')
    plt.ylabel("Power consumption")
    plt.xlabel('day')

    plt.subplot(nIMFs + 2, 1, nIMFs + 2)
    plt.plot(data.index, residue)
    plt.ylabel("Residue")

    for n in range(nIMFs):
        plt.subplot(nIMFs + 2, 1, n + 2)
        plt.plot(data.index, IMFs[n], 'g')
        plt.ylabel("eIMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=4)

    plt.tight_layout()
    # plt.show()

    return IMFs, residue, nIMFs


def plot_IMFs(IMFs: np.ndarray, residue: np.ndarray, num_IMFs: int, data: pd.DataFrame):
    """
    This function aims to reconstruct the Time Series using the IMFs

    :param IMFs: The IMFs returned from using any of the decomposition functions above
    :param residue: The residue returned from using any of the decomposition functions above
    :param num_IMFs: The number of IMFs you want to reconstruct your data. A value of 2 means the last two IMFs
    :return: None
    """

    sum_IMFs = sum(IMFs[-num_IMFs:])
    sum_IMFs += residue

    plt.figure(figsize=(12, 10))
    plt.plot(data.index, data, label="Electricity value")
    plt.plot(data.index, sum_IMFs, label=f"Last {num_IMFs} IMFs")
    plt.legend(loc="upper left")
    plt.show()


def create_dataset(dataset: np.ndarray, lookback_len):
    '''将数据处理成训练数据和验证数据两大类
    dataset:数据集'''
    dataX, dataY = [], []

    for i in range(len(dataset) - lookback_len ):
        look_back_data = dataset[i:(i + lookback_len), 0]
        dataX.append(look_back_data)
        dataY.append(dataset[i + lookback_len, 0])

    return np.array(dataX), np.array(dataY)

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

    sequences = []
    targets = []
    for i in range(len(data) - lookback_len):
        sequence = data[i:i + lookback_len]
        target = data[i + lookback_len]  # Target is the day after the sequence
        sequences.append(sequence)
        targets.append(target)

    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def itransformerFFTlstmmode(lookback_len):
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
        lstm_hidden_dim=lstm_hidden_dim
        # use reversible instance normalization, proposed here https://openreview.net/forum?id=cGDAkQo1C0p . may be redundant given the layernorms within iTransformer (and whatever else attention learns emergently on the first layer, prior to the first layernorm). if i come across some time, i'll gather up all the statistics across variates, project them, and condition the transformer a bit further. that makes more sense
    )

    return model

def LSTM_CNN_CBAM(dataset: np.ndarray, layer, i:int,lookback_len,lr):
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))

    # Normalize the data -- using Min and Max values in each subsequence to normalize the values
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    # Split data into training and testing set
    train_size = int(len(dataset) * SPLIT)
    test_size = len(dataset) - train_size
    train, test = dataset[:train_size, :], dataset[train_size:, :]

    trainX, trainY = create_dataset(train,lookback_len)
    testX, testY = create_dataset(test, lookback_len)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print(len(trainX), len(testX))
    # create and fit the LSTM-CNN-CBAM network
    model = Sequential()
    model.add(LSTM(layer, input_shape=(1, lookback_len), return_sequences=True))
    model.add(Conv1D(filters=512, kernel_size=1, activation='relu', input_shape=(1, lookback_len)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1))
    adam = keras.optimizers.adam_v2.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=700, batch_size=100, verbose=1, validation_split=0.1)

    # model=keras.models.load_model(f'bestmodel{i}.h5')
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    testing_error = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    return testPredict, testY


def train_models(dataset: np.ndarray, model,lookback_len,epochs,lr):

    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))

    # Normalize the data -- using Min and Max values in each subsequence to normalize the values
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    # Split data into training and testing set
    train_size = int(len(dataset) * SPLIT)
    test_size = len(dataset) - train_size
    train, test = dataset[:train_size, :], dataset[train_size:, :]

    train_data, train_targets = preprocess_data(train,lookback_len)
    test_data, test_targets = preprocess_data(test,lookback_len)

    print(len(train_data), len(test_data))

    train_model(model=model, train_data=train_data, train_targets=train_targets, epochs=epochs, lr=lr)
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


    # invert predictions
    testPredict = scaler.inverse_transform(predictions.reshape(-1,1)).flatten()
    testY = scaler.inverse_transform(test_targets.reshape(-1,1)).flatten()

    # print(testPredict, testY)
    testing_error = np.sqrt(mean_squared_error(testY, testPredict))

    return testPredict, testY, testing_error,model


def run_model(IMFs,lookback_len,epochs,lr):


    IMF_predict_list = []
    error_list = []

    for i, IMF in enumerate(IMFs):
        print(type(IMF))
        print(f"IMF number {i + 1}")
        if i==0:
            model = itransformerFFTlstmmode(lookback_len)
            IMF_predict, IMF_test, testing_error, model1 = train_models(IMF, model,lookback_len,epochs,lr)
        else:
            IMF_predict, IMF_test = LSTM_CNN_CBAM(IMF, layer=lstm_hidden_dim,i=i,lookback_len=lookback_len,lr=lr)

        # error_list.append(testing_error)
        IMF_predict_list.append(IMF_predict)
        # model.save(f"bestmodel{i}.h5")

    return IMF_predict_list, error_list



def visualize_results(IMF_predict_list, error_list,lookback_len):
    for i, v in enumerate(IMF_predict_list):
        if i==0:
            IMF_predict_list[i] = v
        else:
            IMF_predict_list[i] = v[: ,0]

    final_prediction = []
    for i in range(len(IMF_predict_list[0])):

        element = 0

        for j in range(len(IMF_predict_list)):
            element += IMF_predict_list[j][i]

        final_prediction.append(element)

    data_plot = data.elec.astype("float32")
    data_plot = np.reshape(data_plot.to_numpy(), (-1, 1))

    train_size = int(len(data_plot) * SPLIT)
    print("train_size:",train_size)
    test_size = len(data_plot) - train_size
    data_plot_train, data_plot_test = data_plot[:train_size], data_plot[train_size:]

    data_plot_testX, data_plot_testY = create_dataset(data_plot_test,lookback_len)

    # 保存数据
    save_data = {
        "real": data_plot_testY,
        "CEEMDAN-fftitransformer": final_prediction
    }
    # df = pd.DataFrame(save_data)
    # df.to_csv(r"D:\F\电网预测\结果图\CEEMDAN-fftitransformer.csv", index=False)

    # Calculate the error

    pred = np.array(final_prediction)
    real = np.array(data_plot_testY)
    mse = mean_squared_error(real, pred)
    mae = mean_absolute_error(real, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(real, pred)
    MAPE = mape(real, pred)

    # err = error(real, pred)
    print("r2:", r2)
    print("MAPE:", MAPE)
    print("RMSE:", rmse)
    print("mse:", mse)
    print("mae:", mae)
    print(final_prediction)
    # print("ERR:",err*100,"%")

    # plot lines
    index = range(len(data.index[train_size+lookback_len :]))
    fig, ax1 = plt.subplots(1, 1, figsize=(18, 12))
    ax1.set_xlabel('天数')
    ax1.set_ylabel('收盘价')
    ax1.plot(index, final_prediction, label="预测值")
    ax1.plot(index, data_plot_testY.tolist(), label="真实值")
    ax1.legend()
    plt.show()
    return r2, MAPE, rmse

def run_differentlookback(lookback,epochs,lr):

    IMFs, residue, n = get_CEEMD_residue(data["elec"])

    IMF_predict_list, error_list = run_model(IMFs,lookback,epochs,lr)

    r2, MAPE, RMSE=visualize_results(IMF_predict_list, error_list, lookback)

    return r2,MAPE,RMSE
if __name__ == '__main__':
    run_differentlookback(5,50,0.0001)
