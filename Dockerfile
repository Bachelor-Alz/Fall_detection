FROM python:3.13-slim@sha256:21e39cf1815802d4c6f89a0d3a166cc67ce58f95b6d1639e68a394c99310d2e5

WORKDIR /app

# Copy necessary files into the container
COPY fall_detection_model.joblib /app
COPY pca_transformation.joblib /app
COPY features_scaler.joblib /app
COPY combined_scaler.joblib /app
COPY main.py /app
COPY process_window.py /app
COPY requirements-api.txt /app
COPY requirements-dev.txt /app

# Combine the libraries from requirements-api.txt with their versions from requirements-dev.txt
RUN awk 'NR==FNR {split($1, a, "=="); libs[a[1]]=$0; next} $1 in libs {print libs[$1]}' /app/requirements-dev.txt /app/requirements-api.txt > /app/requirements.txt

RUN cat /app/requirements.txt

# Install dependencies from the generated requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 9999

# Run the API with uvicorn using 4 workers when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9999", "--workers", "4"]
