from fastapi import FastAPI
import joblib

model = joblib.load("fall_detection_model.pkl")
app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    X = pd.DataFrame(data)
    y_pred = model.predict(X)
    return  y_pred[0]

