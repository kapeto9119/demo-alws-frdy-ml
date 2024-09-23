# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Install system dependencies
# libgl1-mesa-glx is required for matplotlib
# libglib2.0-0 is required for pyarrow
# libheif-examples and libheif-dev are required for pyheif
# rm -rf /var/lib/apt/lists/* is used to clean up the package manager cache
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libheif-examples \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

# Establish the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port that the app runs on
EXPOSE 8000

# FastAPI runs on port 8000 by default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]