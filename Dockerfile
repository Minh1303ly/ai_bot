# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip list

# Copy the application code
COPY . .

# Check environment variables and files
RUN echo "Checking environment: PORT=${PORT}" > /app/env_check.txt
RUN ls -la /app >> /app/env_check.txt

# Expose the port (optional, for documentation)
EXPOSE 8000

# Run the application with Gunicorn, with debug logging and timeout
CMD ["sh", "-c", "echo 'Starting container with PORT: ${PORT}' && gunicorn --bind 0.0.0.0:${PORT} --timeout 120 --log-level debug app:application || echo 'Gunicorn failed with exit code $?'"]