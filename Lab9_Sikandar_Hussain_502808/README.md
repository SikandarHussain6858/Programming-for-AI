# Lab 9: Dockerizing a FastAPI Application

## 1. Create a FastAPI-based API
A basic FastAPI application was developed in `main.py`. The code creates a simple application instance and defines a root endpoint (`/`). 

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "AI Service Running"}
```

## 2. Write a Dockerfile
A `Dockerfile` was created based on `python:3.9-slim` to install the requirements and expose the API on port 8000 using `uvicorn`. 

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --default-timeout=120 --retries=5 -r /app/requirements.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 3. Build Docker image
The image was built on the local machine using the following command:
```bash
docker build -t ai-service .
```

## 4. Run container successfully 
The container was started on port 8000 mapping to 8000 on the host:
```bash
docker run -p 8000:8000 ai-service
```

## 5. Test API endpoints
The API was tested from the host machine using PowerShell. It successfully returned the expected message.
```bash
Invoke-RestMethod -Uri http://localhost:8000/
# Output:
# message
# -------
# AI Service Running
```

## 6. Handle at least 2 container-level issues
During development, the following 2 container-level issues were successfully handled:

1. **ASGI App Missing Attribute Error:** 
   * **Issue:** `ERROR: Error loading ASGI app. Attribute "app" not found in module "main".`
   * **Cause:** The `main.py` file was completely empty, so `uvicorn` could not find the `app` instance to serve.
   * **Fix:** The `main.py` file was populated with the actual complete FastAPI code, and the image was rebuilt.

2. **Network Interface Access Mismatch (`ERR_ADDRESS_INVALID`):**
   * **Issue:** After running the server and binding it to `0.0.0.0`, attempting to access `http://0.0.0.0:8000/` in the Windows web browser resulted in an `ERR_ADDRESS_INVALID` error.
   * **Cause:** Inside the container, `0.0.0.0` correctly tells the server to bind to all available network interfaces. However, from the host context, browsers (especially on Windows) do not know how to route to `0.0.0.0`.
   * **Fix:** Accessed the application by pointing the browser to `http://localhost:8000/` and `http://127.0.0.1:8000/` instead.

## 7. Verify API works inside container
By using `docker exec` against the running container runtime, the API was tested internally using an inline python HTTP request. The container successfully responded to localhost requests from within its own network:
```bash
# Get container ID
docker ps

# Execute HTTP request inside the container natively
docker exec <container_id> python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/').read().decode())"
# Output: {"message":"AI Service Running"}
```

## 8. Document steps
This document outlines all the stages required for the completion of the Docker deployment and validation steps above.
