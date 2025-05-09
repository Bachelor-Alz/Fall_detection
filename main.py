import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import httpx
import uvicorn
from scipy.ndimage import gaussian_filter1d

from process_window import process_window

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

class RequestBody(BaseModel):
    mac: str
    readings: List[IMUDataPoint]

class ResponseBody(BaseModel):
    Mac: str
    Predictions: List[int]

@app.post("/predict", response_model=ResponseBody)
async def predict(data: RequestBody, background_tasks: BackgroundTasks): 
    if not data.readings:
        raise HTTPException(status_code=400, detail="No IMU data provided.")
    
    try:
        df = pd.DataFrame([{ 'accel_x_list': point.ax, 'accel_y_list': point.ay, 'accel_z_list': point.az,
                             'gyro_x_list': point.gx, 'gyro_y_list': point.gy, 'gyro_z_list': point.gz } 
                            for point in data.readings])
        
        if len(data.readings) < WINDOW_SIZE:
            raise HTTPException(status_code=422, detail=f"Data length must be at least {WINDOW_SIZE} samples.")

        df = pre_process(df)
        step_size = WINDOW_SIZE - OVERLAP
        features_df = process_window(df, WINDOW_SIZE, step_size)
        features_df = features_scaler.transform(features_df)
        pca_features = pca.transform(features_df)
        pca_features = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca_features.shape[1])])
        prediction = model.predict(pca_features)
        prediction_list = prediction.tolist()
        body = {"mac": data.mac, "predictions": prediction_list}
        background_tasks.add_task(send_prediction, body)

        return body

    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

async def send_prediction(body):
    print("Sending to .NET backend:", json.dumps(body))
    async with httpx.AsyncClient() as client:
        await client.post("http://localhost:5171/AI/compute", json=body)

def pre_process(df: pd.DataFrame):
    """Pre-processes data: resample to 50Hz, scale sensor data, and apply Gaussian smoothing per filename"""
    columns_to_scale = ['accel_x_list', 'accel_y_list', 'accel_z_list',
                        'gyro_x_list', 'gyro_y_list', 'gyro_z_list']
    
    df[columns_to_scale] = raw_scaler.transform(df[columns_to_scale])

    for col in columns_to_scale:
        df[col] = gaussian_filter1d(df[col].values, sigma=5)

    return df


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
