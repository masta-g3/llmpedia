FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y cron git && \
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
RUN chmod +x workflow.sh

# Create cron log file
RUN touch /var/log/cron.log

CMD ["/bin/bash", "-c", "cron && tail -f /var/log/cron.log"]