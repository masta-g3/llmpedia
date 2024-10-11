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

# Install Firefox
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    libgtk-3-0 \
    libdbus-glib-1-2 \
    libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && wget -O firefox.tar.bz2 "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US" \
    && tar -xjf firefox.tar.bz2 \
    && mv firefox /opt/firefox \
    && ln -s /opt/firefox/firefox /usr/bin/firefox \
    && rm firefox.tar.bz2

# Install Geckodriver
RUN GECKODRIVER_VERSION="v0.35.0" \
    && wget -O geckodriver.tar.gz "https://github.com/mozilla/geckodriver/releases/download/${GECKODRIVER_VERSION}/geckodriver-${GECKODRIVER_VERSION}-linux64.tar.gz" \
    && tar -xzf geckodriver.tar.gz \
    && chmod +x geckodriver \
    && mv geckodriver /usr/bin/ \
    && rm geckodriver.tar.gz

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
