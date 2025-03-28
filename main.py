from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import httpx
from fastapi import FastAPI, HTTPException
import uvicorn


model = joblib.load("fall_detection_model.joblib")
pca = joblib.load("pca_transformation.joblib")

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
        df = pd.DataFrame([{
            'accel_x_list': point.ax,
            'accel_y_list': point.ay,
            'accel_z_list': point.az,
            'gyro_x_list': point.gx,
            'gyro_y_list': point.gy,
            'gyro_z_list': point.gz
        } for point in data])

        df = pre_process(df)
        features_df = extract_features(df)
        pca_features = pca.transform(features_df.values)
        prediction = model.predict(pca_features)
        prediction_list = prediction.tolist()

        # Send the result to a different API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://healthdevice_app:5171/AI/compute",  
                json=prediction_list
            )
            response.raise_for_status() 

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))



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
    WINDOW_SIZE = 80
    OVERLAP = 75
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)