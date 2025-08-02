FROM python:3.10-slim-bullseye

WORKDIR /app

# Upgrade system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set correct permissions
RUN mkdir -p /app/data/cache /app/data/documents && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]