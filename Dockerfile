# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy all source code and data
COPY . /app

# Expose port 8501 for Streamlit
EXPOSE 8501

# Default command runs the dynamic Streamlit dashboard
CMD ["streamlit", "run", "src/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
