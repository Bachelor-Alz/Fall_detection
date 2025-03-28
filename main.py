from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

# Load pre-trained model and PCA
model = joblib.load("fall_detection_model.joblib")
pca = joblib.load("pca_transformation.joblib")

app = FastAPI()

WINDOW_SIZE = 80
OVERLAP = 75

class IMUDataPoint(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

class IMUBatch(BaseModel):
    data: List[IMUDataPoint]

def pre_process(df: pd.DataFrame):
    """Pre-processes data: resample to 50Hz, scale sensor data, and apply Gaussian smoothing per filename"""
    columns_to_scale = ['accel_x_list', 'accel_y_list', 'accel_z_list',
                        'gyro_x_list', 'gyro_y_list', 'gyro_z_list']

    scaler = StandardScaler()
    df.loc[:, columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    for col in columns_to_scale:
        df[col] = gaussian_filter1d(df[col].values, sigma=5)

    return df

def extract_features(df: pd.DataFrame):
    """Extracts features from the data"""
    step_size = WINDOW_SIZE - OVERLAP
    features = []
    columns_to_use = ['gyro_x_list', 'gyro_y_list', 'gyro_z_list', 
                    'accel_x_list', 'accel_y_list', 'accel_z_list']
    
    for i in range(0, len(df) - WINDOW_SIZE + 1, step_size):
        window = df.iloc[i:i + WINDOW_SIZE]
        feature_dict = {}

        for col in columns_to_use:
            col_data = window[col]
            feature_dict[f'{col}_mean'] = col_data.mean()
            feature_dict[f'{col}_std'] = col_data.std()
            feature_dict[f'{col}_min'] = col_data.min()
            feature_dict[f'{col}_max'] = col_data.max()
            feature_dict[f'{col}_median'] = col_data.median()
            feature_dict[f'{col}_iqr'] = col_data.quantile(0.75) - col_data.quantile(0.25)
            feature_dict[f'{col}_energy'] = np.sum(col_data**2)
            feature_dict[f'{col}_rms'] = np.sqrt(np.mean(col_data**2))
            feature_dict[f'{col}_sma'] = np.sum(np.abs(col_data)) / WINDOW_SIZE
            feature_dict[f'{col}_skew'] = col_data.skew()
            feature_dict[f'{col}_kurtosis'] = col_data.kurtosis()
            feature_dict[f'{col}_absum'] = np.sum(np.abs(col_data))
            feature_dict[f'{col}_DDM'] = col_data.max() - col_data.min()
            feature_dict[f'{col}_GMM'] = np.sqrt((col_data.max() - col_data.min())**2 + 
                                                (np.argmax(col_data) - np.argmin(col_data))**2)
            feature_dict[f'{col}_MD'] = max(np.diff(col_data))

        accel_magnitude = np.sqrt(window['accel_x_list']**2 + window['accel_y_list']**2 + window['accel_z_list']**2)
        gyro_magnitude = np.sqrt(window['gyro_x_list']**2 + window['gyro_y_list']**2 + window['gyro_z_list']**2)

        for mag, prefix in zip([accel_magnitude, gyro_magnitude], ['accel_magnitude', 'gyro_magnitude']):
            feature_dict[f'{prefix}_mean'] = mag.mean()
            feature_dict[f'{prefix}_std'] = mag.std()
            feature_dict[f'{prefix}_min'] = mag.min()
            feature_dict[f'{prefix}_max'] = mag.max()
            feature_dict[f'{prefix}_median'] = mag.median()
            feature_dict[f'{prefix}_iqr'] = mag.quantile(0.75) - mag.quantile(0.25)
            feature_dict[f'{prefix}_energy'] = np.sum(mag**2)
            feature_dict[f'{prefix}_rms'] = np.sqrt(np.mean(mag**2))
            feature_dict[f'{prefix}_sma'] = np.sum(np.abs(mag)) / WINDOW_SIZE

        features.append(feature_dict)

    return pd.DataFrame(features)


@app.post("/predict")
async def predict(data: IMUBatch):
    try:
        df = pd.DataFrame([{
            'accel_x_list': point.accel_x,
            'accel_y_list': point.accel_y,
            'accel_z_list': point.accel_z,
            'gyro_x_list': point.gyro_x,
            'gyro_y_list': point.gyro_y,
            'gyro_z_list': point.gyro_z
        } for point in data.data])

        df = pre_process(df)
        features_df = extract_features(df)
        pca_features = pca.transform(features_df)
        prediction = model.predict(pca_features)
        prediction_list = prediction.tolist()

        return {"prediction": prediction_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

