FROM python:3.12-slim

# Create a working directory
WORKDIR /app

# Install dependencies first for better caching
COPY salesforce-qwen-agent/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app source code
COPY salesforce-qwen-agent /app/

# Ensure the cache/uploads directory exists and has right permissions for Hugging Face
RUN mkdir -p /app/uploads/.cache && chmod -R 777 /app/uploads

# Expose port
EXPOSE 7860

# Run uvicorn directly (bypassing the local python script so it doesn't open a browser)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
