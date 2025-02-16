# Dockerfile

# 1. Choose a base image
FROM python:3.11-slim

# 2. Create a working directory inside the container
WORKDIR /app

# 3. Set PYTHONPATH so /app is on the search path
ENV PYTHONPATH=/app

# 4. Copy requirements.txt into the container
COPY requirements.txt /app

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire project code into the container
COPY . /app

# 7. Default command: run the training script
CMD ["python", "src/train.py", "--epochs", "5", "--batch_size", "16"]