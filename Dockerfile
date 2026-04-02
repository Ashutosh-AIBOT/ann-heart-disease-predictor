# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# Simplified to only essential packages for Streamlit + PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using CPU-only torch to minimize image size for Hugging Face Spaces
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and switch to it for Hugging Face security
RUN useradd -m -u 1000 user

# Copy the rest of the application code as the non-root user
# This ensures that the user has write permissions where needed
COPY --chown=user:user . .

# Set environment variables for the non-root user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Expose the port that Streamlit will run on
EXPOSE 7860

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
