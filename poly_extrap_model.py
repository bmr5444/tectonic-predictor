import pandas as pd
import numpy as np

def load_model_data():
    """
    Loads the plate data with the velocity vectors
    into a DataFrame
    """
    df = pd.read_csv("model_input.csv")
    return df

def polynomial_curve_fit(df):
    """
    Uses wx, wy, and wz to find a polynomial curve fit for the
    data used to to predict velocity vectors in the future
    """

if __name__ == "__main__":
    """"""