import pandas as pd
import numpy as np
import torch

def load_model_data():
    """
    Loads data into a DataFrame from the model_input.csv file
    """
    df = pd.read_csv("outputs/model_input.csv")
    return df

if __name__ == "__main__":
    """"""