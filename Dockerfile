# aviationstack-mcp requires Python >=3.13, so we can't use 3.12.
FROM python:3.13-slim

# System packages: build-essential for any wheels that compile,
# and curl for the container healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer — only re-runs if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project in, then install the local aviationstack package
# from its committed source (the command we already verified works locally).
COPY . .
RUN pip install --no-cache-dir ./aviationstack-mcp

# Streamlit serves on 8501
EXPOSE 8501

# Healthcheck hits Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Bind to 0.0.0.0 so the container is reachable from outside
CMD ["streamlit", "run", "frontend.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]