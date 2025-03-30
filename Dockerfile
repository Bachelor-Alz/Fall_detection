FROM python:3.10-slim

WORKDIR /app

COPY fall_detection_model.joblib /app
COPY pca_transformation.joblib /app
COPY features_scaler.joblib /app
COPY combined_scaler.joblib /app
COPY main.py /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 9999

# Run app.py when the container launches
CMD ["python", "main.py"]
