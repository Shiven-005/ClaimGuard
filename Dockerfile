FROM python:3.11-slim

WORKDIR /app

# Install system build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY pyproject.toml .
COPY src/ ./src/
COPY app/ ./app/
COPY api/ ./api/
COPY artifacts/ ./artifacts/
COPY README.md .

# Install editable package
RUN pip install --no-cache-dir -e .

# Expose Hugging Face Space port
EXPOSE 7860

# Run Streamlit app on 0.0.0.0:7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
