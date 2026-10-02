# Match the versions CI tests on; 3.14-slim also satisfies requires-python but
# was untested here.
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# scripts/train.py imports scripts.data_prep, which needs the repository root on
# the path. Without this the container fails at startup with
# ModuleNotFoundError: No module named 'scripts'.
ENV PYTHONPATH=/app

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Default command
CMD ["python", "scripts/train.py"]
