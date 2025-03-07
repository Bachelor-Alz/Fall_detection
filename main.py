from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class IMUData(BaseModel):
    accelerometer: list[float]  # [x, y, z]
    gyroscope: list[float]      # [x, y, z]

@app.post("/imu")
async def receive_imu_data():
    return True

