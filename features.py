import pandas as pd
import numpy as np

#1. Load plate_histories.csv
#2. Flip time so it runs oldest → present (410 → 0 becomes index 0 → 82)
#3. Convert pole_lat/pole_lon → Cartesian x, y, z
#4. Compute velocity: difference in x,y,z between each timestep
#5. Save as model_input.csv

def load_plate_histories():
    """
    Read from plate histories file and load the data
    """
    df = pd.read_csv("outputs/plate_histories.csv")
    return df

def convert_to_cartesian(df):
    """
    Converts the latitudes and longitudes into cartesian rotation vectors (wx, wy, wz)
    It represents the plate's angular velocity at each Ma time step
    """
    lat_rad = np.radians(df['pole_lat'])
    lon_rad = np.radians(df['pole_lon'])
    df['omega'] = df.groupby('plate_id')['angle_deg'].diff().abs() / 5
    omega_rad = np.radians(df['omega'])

    df['wx'] = omega_rad * np.cos(lat_rad) * np.cos(lon_rad)
    df['wy'] = omega_rad * np.cos(lat_rad) * np.sin(lon_rad)
    df['wz'] = omega_rad * np.sin(lat_rad)

    return df

if __name__ == "__main__":
    
    df = load_plate_histories()
    df = convert_to_cartesian(df)
    
    df.to_csv("outputs/plate_velocities.csv", index=False)
    print("Saved to outputs/plate_velocities.csv")

    model_df = df[['plate_id', 'time_ma', 'wx', 'wy', 'wz']].dropna()
    model_df.to_csv("outputs/model_input.csv", index=False)
    print("Saved to ouputs/model_input.csv")

    print(model_df.head(20))
    print(f"\nTotal rows: {model_df.shape[0]}")