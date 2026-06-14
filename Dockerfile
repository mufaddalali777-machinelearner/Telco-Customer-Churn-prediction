FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install directly (No compilers allowed!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code and models
COPY . .