FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Streamlit config — disable telemetry and set port
RUN mkdir -p ~/.streamlit
RUN echo '[server]\nport = 8080\nheadless = true\nenableCORS = false\n\n[browser]\ngatherUsageStats = false' \
    > ~/.streamlit/config.toml

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]