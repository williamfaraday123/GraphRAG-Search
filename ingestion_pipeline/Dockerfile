# Use the official lightweight Python image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# 1. Create a virtual environment
RUN python -m venv /opt/venv

# 2. Activate the virtual environment (add to PATH)
ENV PATH="/opt/venv/bin:$PATH"

# 3. Install system dependencies (if needed for specific python packages like gcc)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc

# 4. Copy only requirements first (for Docker layer caching)
COPY requirements.txt .

# 5. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY . .

# Define the default command to run the pipeline
# Assuming your main script is named main.py
CMD ["python", "main.py"]