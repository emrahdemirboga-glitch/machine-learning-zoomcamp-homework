# Start from the base image
FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /code

# Copy pyproject.toml and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir poetry uvicorn && \
    poetry install --no-root

# Copy your FastAPI app
COPY app.py .

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
