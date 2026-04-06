# Use the official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# 1. Create a virtual environment
RUN python -m venv /opt/venv

# 2. Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# 3. Copy requirements first
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code
COPY . .

# Expose port if your query processor runs as an API (e.g., FastAPI/Flask)
# EXPOSE 8000

# Default command
CMD ["python", "main.py"]