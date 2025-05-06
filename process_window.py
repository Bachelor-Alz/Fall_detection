import numpy as np
import pandas as pd

def process_window(df: pd.DataFrame, window_size: int, step_size: int, include_metadata: bool = False) -> pd.DataFrame:
    """Extracts features from the data using NumPy for improved performance."""
    columns_to_use = ['gyro_x_list', 'gyro_y_list', 'gyro_z_list', 
                      'accel_x_list', 'accel_y_list', 'accel_z_list']
    

    data = df[columns_to_use].values  

    if len(data) < window_size:
        return pd.DataFrame()

    num_windows = max(1, (len(data) - window_size) // step_size)  
    features = []

    # Pre-extract metadata if needed
    if include_metadata:
        time_data = df['time'].values
        filename = df['filename'].iloc[0]
        start_time = df['start_time'].iloc[0]
        end_time = df['end_time'].iloc[0]

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = data[start_idx:end_idx]
        feature_dict = {}

        if include_metadata:
            process_meta_data(window_size, time_data, filename, start_time, end_time, start_idx, end_idx, feature_dict)

        for j, col in enumerate(columns_to_use):
            process_col(window_size, window, feature_dict, j, col)

        accel_mag = np.linalg.norm(window[:, 3:6], axis=1)  
        gyro_mag = np.linalg.norm(window[:, 0:3], axis=1)

        for mag, prefix in zip([accel_mag, gyro_mag], ['accel_magnitude', 'gyro_magnitude']):
            process_mag_col(window_size, feature_dict, mag, prefix)

        features.append(feature_dict)

    return pd.DataFrame(features)

def process_meta_data(window_size, time_data, filename, start_time, end_time, start_idx, end_idx, feature_dict):
    time_window = time_data[start_idx:end_idx]
    is_fall = 0

    if not pd.isna(start_time) and not pd.isna(end_time):  
        valid_mask = (time_window >= start_time) & (time_window <= end_time)
        if np.sum(valid_mask) / window_size >= 0.6:
            is_fall = 1

    feature_dict.update({
        'is_fall': is_fall,
        'filename': filename,
        'start_time': time_window[0],
        'end_time': time_window[-1]
    })

def process_mag_col(window_size, feature_dict, mag, prefix):
    feature_dict.update({
        f'{prefix}_mean': np.mean(mag),
        f'{prefix}_std': np.std(mag),
        f'{prefix}_min': np.min(mag),
        f'{prefix}_max': np.max(mag),
        f'{prefix}_median': np.median(mag),
        f'{prefix}_iqr': np.percentile(mag, 75) - np.percentile(mag, 25),
        f'{prefix}_energy': np.sum(mag**2),
        f'{prefix}_rms': np.sqrt(np.mean(mag**2)),
        f'{prefix}_sma': np.sum(np.abs(mag)) / window_size,
        f'{prefix}_absum': np.sum(np.abs(mag))
    })

def process_col(window_size, window, feature_dict, j, col):
    col_data = window[:, j]
    feature_dict.update({
        f'{col}_mean': np.mean(col_data),
        f'{col}_std': np.std(col_data),
        f'{col}_min': np.min(col_data),
        f'{col}_max': np.max(col_data),
        f'{col}_median': np.median(col_data),
        f'{col}_iqr': np.percentile(col_data, 75) - np.percentile(col_data, 25),
        f'{col}_energy': np.sum(col_data**2),
        f'{col}_rms': np.sqrt(np.mean(col_data**2)),
        f'{col}_sma': np.sum(np.abs(col_data)) / window_size,
        f'{col}_absum': np.sum(np.abs(col_data)),
        f'{col}_skew': ((col_data - np.mean(col_data))**3).mean() / (np.std(col_data)**3 + 1e-8),
        f'{col}_kurtosis': ((col_data - np.mean(col_data))**4).mean() / (np.std(col_data)**4 + 1e-8),
        f'{col}_DDM': np.max(col_data) - np.min(col_data),
        f'{col}_GMM': np.sqrt((np.max(col_data) - np.min(col_data))**2 + (np.argmax(col_data) - np.argmin(col_data))**2),
        f'{col}_MD': np.max(np.diff(col_data)) if len(col_data) > 1 else 0
    })
