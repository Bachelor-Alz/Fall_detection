from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import httpx
from fastapi import FastAPI, HTTPException
import uvicorn
from scipy.ndimage import gaussian_filter1d
import time

model : RandomForestClassifier = joblib.load("fall_detection_model.joblib")
pca : PCA = joblib.load("pca_transformation.joblib")
raw_scaler : StandardScaler = joblib.load("combined_scaler.joblib")
features_scaler : StandardScaler = joblib.load("features_scaler.joblib")

WINDOW_SIZE = 80
OVERLAP = 75

app = FastAPI()

class IMUDataPoint(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float



@app.post("/predict")
async def predict(data: List[IMUDataPoint]): 
    if not data:
        raise HTTPException(status_code=400, detail="No IMU data provided.")
    
    try:
        df = pd.DataFrame([{ 'accel_x_list': point.ax, 'accel_y_list': point.ay, 'accel_z_list': point.az,
                             'gyro_x_list': point.gx, 'gyro_y_list': point.gy, 'gyro_z_list': point.gz } 
                            for point in data])
        
        if len(data) < WINDOW_SIZE:
            raise HTTPException(status_code=422, detail=f"Data length must be at least {WINDOW_SIZE} samples.")

        df = pre_process(df)
        features_df = extract_features(df)
        features_df = features_scaler.transform(features_df)
        pca_features = pca.transform(features_df)
        pca_features = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca_features.shape[1])])
        prediction = model.predict(pca_features)
        prediction_list = prediction.tolist()

        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:5171/AI/compute", json=prediction_list)
            response.raise_for_status()

        return prediction_list
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def pre_process(df: pd.DataFrame):
    """Pre-processes data: resample to 50Hz, scale sensor data, and apply Gaussian smoothing per filename"""
    columns_to_scale = ['accel_x_list', 'accel_y_list', 'accel_z_list',
                        'gyro_x_list', 'gyro_y_list', 'gyro_z_list']
    
    # Scale the raw sensor data
    df[columns_to_scale] = raw_scaler.transform(df[columns_to_scale])

    for col in columns_to_scale:
        df[col] = gaussian_filter1d(df[col].values, sigma=5)

    return df

def extract_features(df: pd.DataFrame):
    """Extracts features from the data using NumPy for improved performance."""
    step_size = WINDOW_SIZE - OVERLAP
    columns_to_use = ['gyro_x_list', 'gyro_y_list', 'gyro_z_list', 
                      'accel_x_list', 'accel_y_list', 'accel_z_list']
    
    # Convert DataFrame to NumPy array
    data = df[columns_to_use].values  # Shape: (num_samples, num_features)
    num_windows = (len(data) - WINDOW_SIZE) // step_size  # Number of windows to extract
    features = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + WINDOW_SIZE
        window = data[start_idx:end_idx]  # Shape: (WINDOW_SIZE, num_features)
        feature_dict = {}

        for j, col in enumerate(columns_to_use):
            col_data = window[:, j]
            feature_dict[f'{col}_mean'] = np.mean(col_data)
            feature_dict[f'{col}_std'] = np.std(col_data)
            feature_dict[f'{col}_min'] = np.min(col_data)
            feature_dict[f'{col}_max'] = np.max(col_data)
            feature_dict[f'{col}_median'] = np.median(col_data)
            feature_dict[f'{col}_iqr'] = np.percentile(col_data, 75) - np.percentile(col_data, 25)
            feature_dict[f'{col}_energy'] = np.sum(col_data**2)
            feature_dict[f'{col}_rms'] = np.sqrt(np.mean(col_data**2))
            feature_dict[f'{col}_sma'] = np.sum(np.abs(col_data)) / WINDOW_SIZE
            feature_dict[f'{col}_skew'] = ((col_data - np.mean(col_data))**3).mean() / (np.std(col_data)**3 + 1e-8)
            feature_dict[f'{col}_kurtosis'] = ((col_data - np.mean(col_data))**4).mean() / (np.std(col_data)**4 + 1e-8)
            feature_dict[f'{col}_absum'] = np.sum(np.abs(col_data))
            feature_dict[f'{col}_DDM'] = np.max(col_data) - np.min(col_data)
            feature_dict[f'{col}_GMM'] = np.sqrt((np.max(col_data) - np.min(col_data))**2 + 
                                                 (np.argmax(col_data) - np.argmin(col_data))**2)
            feature_dict[f'{col}_MD'] = np.max(np.diff(col_data)) if len(col_data) > 1 else 0

        # Compute magnitude features
        accel_mag = np.linalg.norm(window[:, 3:6], axis=1)  # Using columns accel_x, accel_y, accel_z
        gyro_mag = np.linalg.norm(window[:, 0:3], axis=1)  # Using columns gyro_x, gyro_y, gyro_z

        for mag, prefix in zip([accel_mag, gyro_mag], ['accel_magnitude', 'gyro_magnitude']):
            feature_dict[f'{prefix}_mean'] = np.mean(mag)
            feature_dict[f'{prefix}_std'] = np.std(mag)
            feature_dict[f'{prefix}_min'] = np.min(mag)
            feature_dict[f'{prefix}_max'] = np.max(mag)
            feature_dict[f'{prefix}_median'] = np.median(mag)
            feature_dict[f'{prefix}_iqr'] = np.percentile(mag, 75) - np.percentile(mag, 25)
            feature_dict[f'{prefix}_energy'] = np.sum(mag**2)
            feature_dict[f'{prefix}_rms'] = np.sqrt(np.mean(mag**2))
            feature_dict[f'{prefix}_sma'] = np.sum(np.abs(mag)) / WINDOW_SIZE

        features.append(feature_dict)

    return pd.DataFrame(features)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)