FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y cron git vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Chromium, ChromeDriver, dependencies for Selenium, and poppler-utils
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    libnss3 \
    libxss1 \
    libasound2 \
    libxtst6 \
    libgtk-3-0 \
    libgconf-2-4 \
    libgbm-dev \
    libdbus-glib-1-2 \
    libdbus-1-3 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Chrome
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_PATH=/usr/lib/chromium/

# Install Firefox and GeckoDriver
RUN apt-get update && apt-get install -y \
    firefox-esr \
    libgtk-3-0 \
    libdbus-glib-1-2 \
    libxt6 \
    libpci3 \
    libegl1 \
    libegl1-mesa \
    libgl1-mesa-dri \
    xvfb \
    libxrender1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    libnss3 \
    libxrandr2 \
    libxdamage1 \
    libxfixes3 \
    libx11-xcb1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxinerama1 \
    libxkbcommon0 \
    libxshmfence1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install GeckoDriver
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz \
    && tar -xvzf geckodriver-v0.35.0-linux64.tar.gz \
    && chmod +x geckodriver \
    && mv geckodriver /usr/local/bin/ \
    && rm geckodriver-v0.35.0-linux64.tar.gz

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
ENV GECKODRIVER_PATH=/usr/local/bin/geckodriver
ENV PROJECT_PATH=/app
ENV MODELS_PATH=/app/ComfyUI/models
ENV COMFY_PATH=/app/ComfyUI

# Create entrypoint script
RUN echo '#!/bin/bash\nXvfb :99 -screen 0 1920x1080x24 &\nexport DISPLAY=:99\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["cron", "-f"]

# Copy the update script
COPY update_and_restart.sh .
RUN chmod +x update_and_restart.sh