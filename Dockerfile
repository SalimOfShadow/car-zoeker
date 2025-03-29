FROM python:3.9-slim

WORKDIR /app
# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*
# Copy only the requirements file first to leverage Docker cache efficiently
COPY requirements.txt .

COPY ./models /app/models

RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

ENV PYTHONPATH=/app/src

# Copy the rest of the application code
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 7020

# Command to run the FastAPI app using Uvicorn inside the container
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7020"]