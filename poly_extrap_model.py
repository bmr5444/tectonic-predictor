import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #may not need this

def load_model_data():
    """
    Loads the plate data with the velocity vectors
    into a DataFrame
    """
    df = pd.read_csv("outputs/model_input.csv")
    return df

def polynomial_curve_fit(df, future_times, degree=4):
    """
    Uses wx, wy, and wz to find a polynomial curve fit for the
    data used to to predict velocity vectors in the future

    Parameters:
        df           :
        future_times :
        degree       :
    Returns:
        Dataframe of predicted wx, wy, and wz for each plate at each future time
    """
    predictions = []

    for plate_id, plate_df in df.groupby('plate_id'):
        plate_df = plate_df.sort_values('time_ma')
        time = plate_df['time_ma'].values
        wx = plate_df['wx'].values
        wy = plate_df['wy'].values
        wz = plate_df['wz'].values

        wx_coeffs = np.polyfit(time, wx, degree)
        wx_func = np.poly1d(wx_coeffs)
        wy_coeffs = np.polyfit(time, wy, degree)
        wy_func = np.poly1d(wy_coeffs)
        wz_coeffs = np.polyfit(time, wz, degree)
        wz_func = np.poly1d(wz_coeffs)

        for t in future_times:
            predictions.append({
                'plate_id': plate_id,
                'time_ma': t,
                'wx': wx_func(t),
                'wy': wy_func(t),
                'wz': wz_func(t),
            })
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    df = load_model_data()

    future_times = list(range(0, -155, -5))
    predictions = polynomial_curve_fit(df, future_times)

    print(predictions.head(20))
    print(f"\nTotal predictions: {predictions.shape[0]}")
    
    predictions.to_csv("outputs/polynomial_predictions.csv", index=False)
    print("\n Saved to outputs/predictions.csv")