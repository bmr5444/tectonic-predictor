import pygplates
import pandas as pd
import numpy as np

def load_rotation_model():
    """
    Loads the .rot files into PyGPlates
    """
    return pygplates.RotationModel([
        "data/Young_etal_2018_GeosciFrontiers/Rotations/Global_250-0Ma_Young_et_al.rot",
        "data/Young_etal_2018_GeosciFrontiers/Rotations/Global_410-250Ma_Young_et_al.rot"
    ])

def extract_plate_history(rotation_model, plate_id, time_start=410, time_end=0, time_step=5):
    """
    """
    records = []
    times = np.arange(time_start, time_end - time_step, -time_step)
    for time in times:
        rotation = rotation_model.get_rotation(
            to_time = time,
            moving_plate_id = plate_id,
            anchor_plate_id = 0
        )
        if rotation is None or rotation.represents_identity_rotation():
            continue

        euler_pole, angle = rotation.get_euler_pole_and_angle()
        pole_lat, pole_lon = euler_pole.to_lat_lon()

        records.append({
            'plate_id': plate_id,
            'time_ma': time,
            'pole_lat': pole_lat,
            'pole_lon': pole_lon,
            'angle_deg': np.degrees(angle),
        })
    
    return pd.DataFrame(records)
    

def extract_all_plates(rotation_model, plate_ids, time_start=410, time_end=0, time_step=5):
    all_dfs = []
    
    for pid in plate_ids:
        df = extract_plate_history(rotation_model, pid, time_start, time_end, time_step)
        if not df.empty:
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":

    rotation_model = load_rotation_model()

    PLATE_IDS = [
        101,
        201,
        301,
        401,
        501,
        601,
        701,
        714,
        801,
        802,
        902,
    ]

    df = extract_all_plates(rotation_model, PLATE_IDS)
    print(df.head(20))
    print(f"\nTotal rows: {df.shape[0]}")
    print(f"Plates extracted: {df['plate_id'].unique()}")
    print(f"Time range: {df['time_ma'].max()} Ma to {df['time_ma'].min()} Ma")

    df.to_csv("outputs/plate_histories.csv", index=False)
    print("\nSaved to outputs/plate_histories.csv")