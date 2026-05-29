import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from lstm_model import TectonicLSTM

def load_model_data():
    """
    Loads data into a DataFrame from the model_input.csv file
    """
    df = pd.read_csv("outputs/model_input.csv")
    return df

def normalize_data(df):
    """
    Normalizes wx, wy, and wz into values from 0 to 1 for LSTM
    Parameters:
        df: the original unnormalized dataframe
    Return:
        normalized_df : the new normalized dataframe
        max_vals: array of the max values of the data in wx, wy, wz
        min_vals: array of the min values of the data in wx, wy, wz
    """
    normalized_df = df.copy()

    min_vals = df[['wx', 'wy', 'wz']].min()
    max_vals = df[['wx', 'wy', 'wz']].max()

    normalized_df['wx'] = (df['wx'] - min_vals['wx']) / (max_vals['wx'] - min_vals['wx'])
    normalized_df['wy'] = (df['wy'] - min_vals['wy']) / (max_vals['wy'] - min_vals['wy'])
    normalized_df['wz'] = (df['wz'] - min_vals['wz']) / (max_vals['wz'] - min_vals['wz'])

    return normalized_df, max_vals, min_vals

def denormalize_data(normalized_predictions_df):
    """
    """

def create_sequences(normalized_df, window_size=10):
    """
    Builds the sliding window sequences that the LSTM will train on
    Parameters:
        normalized_df: the normalized dataframe
        window_size  : the number of items in the window of input to produce one 
                       item of output
    Return:
        np_inputs : numpy array of shape (num_sequences, window_size, 3)
                    where 3 = wx, wy, wz 
        np_outputs: numpy array of shape (num_sequeces, 3)
                    the velocity vector immediately after each input window
    """
    inputs = []
    outputs = []

    for plate_id, plate_df in normalized_df.groupby('plate_id'):
        for i in range(plate_df.shape[0]-window_size-1):
            window = plate_df[['wx', 'wy', 'wz']].iloc[i: i+window_size].values
            inputs.append(window)
            outputs.append(plate_df[['wx', 'wy', 'wz']].iloc[i+window_size].values)

    np_inputs = np.array(inputs)
    np_outputs = np.array(outputs)
    return np_inputs, np_outputs

def train_model(model, X, y, epochs=150, batch_size=32, learning_rate=0.001):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            # ...
            optimizer.zero_grad() # resetting gradients
        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: ...")
        

if __name__ == "__main__":
    """"""