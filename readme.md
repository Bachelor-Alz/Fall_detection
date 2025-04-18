# IMU fall detector

This project is a FastAPI-based application designed to receive IMU (Inertial Measurement Unit) data through a POST endpoint and classify falls.

## Setting Up Virtual Environment

1. **Create a Virtual Environment**:
   In your project folder, run the following command to create a virtual environment:

```bash
python3 -m venv venv
```

2. Activate the Vitual Enviroemnt

   - Windows

   ```bash
   .\venv\Scripts\activate
   ```

   - Mac

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements-dev.txt
   ```

## Running the project

```bash
uvicorn main:app --port 9999
```

for changes to be reflected immediatly use:

```bash
uvicorn main:app --reload --port 9999
```

If you want to make the API faster you can utilise multiple CPU cores

```bash
uvicorn main:app --host 0.0.0.0 --port 9999 --workers 4
```

## Swagger Documentation

Swagger documentation can be found at http://127.0.0.1:8000/docs

## Docker

You can build the API with the Dockerfile using the command:

```bash
docker build -t image-name .
```

After building the image you can run the app in a docker container using the command

```bash
docker run -d -p 9999:9999 image-name
```
