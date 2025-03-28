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
   pip install -r requirements.txt
   ```

## Running the project

```bash
uvicorn main:app
```

for changes to be reflected immediatly use:

```bash
uvicorn main:app --reload
```

## Swagger Documentation

Swagger documentation can be found at http://127.0.0.1:8000/docs