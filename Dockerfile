FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y cron git vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Install ComfyUI dependencies
WORKDIR /app/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# Return to main app directory
WORKDIR /app

COPY requirements_dev.txt .
RUN pip install --no-cache-dir -r requirements_dev.txt

COPY workflow.sh .
COPY utils ./utils
COPY workflow ./workflow
RUN chmod +x workflow.sh

# Copy crontab file and set up the cron job
COPY crontab /etc/cron.d/my-crontab
RUN chmod 0644 /etc/cron.d/my-crontab && \
    crontab /etc/cron.d/my-crontab

# Install Chromium, ChromeDriver, dependencies for Selenium, and poppler-utils
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    vim \
    curl \
    unzip \
    chromium \
    chromium-driver \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Chrome
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_PATH=/usr/lib/chromium/

# Install Firefox and GeckoDriver
RUN apt-get update && apt-get install -y \
    firefox-esr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install GeckoDriver
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.33.0/geckodriver-v0.33.0-linux64.tar.gz \
    && tar -xvzf geckodriver-v0.33.0-linux64.tar.gz \
    && chmod +x geckodriver \
    && mv geckodriver /usr/bin/ \
    && rm geckodriver-v0.33.0-linux64.tar.gz

# Copy the rest of your application
COPY . .

RUN mkdir -p /app/imgs && \
    mkdir -p /app/data && \
    mkdir -p /app/data/arxiv_text && \
    mkdir -p /app/data/nonllm_arxiv_text && \
    mkdir -p /app/data/arxiv_first_page && \
    mkdir -p /app/data/arxiv_chunks && \
    mkdir -p /app/data/arxiv_large_chunks

# Set environment variables
ENV PROJECT_PATH=/app
ENV MODELS_PATH=/app/ComfyUI/models
ENV COMFY_PATH=/app/ComfyUI

# Start cron in the foreground
CMD ["cron", "-f"]