# Start from the official Python image
FROM python:3.11-slim
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    update-ca-certificates

# Set the working directory inside the container
WORKDIR /app

# Copy all code into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command to run when container starts
CMD ["python", "signal_test.py"]

