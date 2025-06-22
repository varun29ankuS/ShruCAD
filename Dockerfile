# Dockerfile

# Start from a standard Python 3.10 environment
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# --- Install System Dependencies ---
# This is the crucial step to install the Tesseract engine itself
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# Copy our project's dependency list
COPY requirements.txt .

# Install our Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our application code
COPY . .

# Expose the port that the app will run on
EXPOSE 10000

# --- The command to run when the container starts ---
# This replaces the "Start Command" in the Render UI
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:10000", "main:app"]
