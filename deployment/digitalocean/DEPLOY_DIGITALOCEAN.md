# Deploying LLMpedia on DigitalOcean with Nginx and Systemd

This guide provides step-by-step instructions to deploy the LLMpedia Streamlit application on a DigitalOcean Ubuntu droplet using Nginx as a reverse proxy and systemd to manage the application process.

## Prerequisites

1.  **DigitalOcean Account**: You need an account with DigitalOcean.
2.  **Domain Name**: A registered domain name (e.g., `llmpedia.ai`).
3.  **SSH Access**: Ability to SSH into your Droplet.
4.  **Populated Database**: Access to the PostgreSQL database populated by the `llmpedia_workflows` pipeline.

## Step 1: Create and Configure Droplet

1.  **Create Droplet**: Log in to DigitalOcean and create a new Ubuntu Droplet (e.g., Ubuntu 22.04 LTS). Choose a size appropriate for your expected traffic (a basic $6/month droplet might be sufficient to start).
2.  **SSH Key**: Add your SSH key for secure access.
3.  **Initial Server Setup**: Connect to your Droplet via SSH:
    ```bash
    ssh root@YOUR_DROPLET_IP
    ```
    - Create a new non-root user with sudo privileges (replace `your_user`):
      ```bash
      adduser your_user
      usermod -aG sudo your_user
      # Optional: Copy SSH keys for the new user
      rsync --archive --chown=your_user:your_user ~/.ssh /home/your_user
      ```
    - Log in as the new user:
      ```bash
      exit # Log out from root
      ssh your_user@YOUR_DROPLET_IP
      ```
    - Set up a basic firewall (UFW):
      ```bash
      sudo ufw allow OpenSSH
      sudo ufw allow 'Nginx Full' # Allows HTTP (80) and HTTPS (443)
      sudo ufw enable
      ```

## Step 2: Install Dependencies

1.  **Update System**: Keep your system up-to-date:
    ```bash
    sudo apt update
    sudo apt upgrade -y
    ```
2.  **Install Python, Git, Nginx, and PostgreSQL Dev Libraries**:
    ```bash
    ## libpq-dev is needed for psycopg2-binary build
    sudo apt install -y python3 python3-dev git nginx curl libpq-dev
    ```
3.  **Install uv (Python Package Manager)**:
    ```bash
    curl -sSf https://astral.sh/uv/install.sh | sh
    ## Add uv to your PATH if the installer doesn't do it automatically
    export PATH="$HOME/.cargo/bin:$PATH"
    ## Add this line to your .bashrc or .zshrc to make it permanent
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    ```
4.  **Install Certbot (for SSL)**:
    ```bash
    sudo apt install -y certbot python3-certbot-nginx
    ```

## Step 3: Deploy Application Code

1.  **Clone Repository**: Clone your `llmpedia` repository into the user's home directory (or another preferred location):
    ```bash
    cd ~
    git clone <your_repository_url> llmpedia
    cd llmpedia
    ```
2.  **Create Virtual Environment**: Set up a virtual environment using uv:
    ```bash
    uv venv
    source .venv/bin/activate
    ```
3.  **Install Dependencies**: Install the required Python packages using uv:
    ```bash
    uv pip install -r requirements.txt
    ```
    *Note: `streamlit` is included in `requirements.txt`.*

4.  **Configure Environment Variables**:
    - Copy the template:
      ```bash
      cp .env.template .env
      ```
    - Edit the `.env` file with your actual database credentials, LLM API keys, and AWS keys:
      ```bash
      nano .env
      ```
      *(Save and close: Ctrl+X, then Y, then Enter)*

5.  **Create Static Directory for Nginx**: Create a directory where Nginx will serve static files like the logo.
    ```bash
    sudo mkdir -p /var/www/llmpedia/static
    sudo chown $USER:$USER /var/www/llmpedia/static
    ```

6.  **Upload Logo**: Upload your `logo.png` from your local machine to the server using `scp`. Replace `path/to/local/logo.png` with the actual path on your computer and `your_user@YOUR_DROPLET_IP` accordingly.
    ```bash
    # Run this command from your LOCAL machine, not the server
    scp path/to/local/logo.png your_user@YOUR_DROPLET_IP:/var/www/llmpedia/static/logo.png
    ```

7.  **Test the App (Optional)**: You can test if the app runs directly:
    ```bash
    # Back on the server SSH session
    cd ~/llmpedia # Ensure you are in the app directory
    source .venv/bin/activate
    streamlit run app.py
    ```
    Access it via `http://YOUR_DROPLET_IP:8501` in your browser. Stop it with `Ctrl+C`.
    *Note: You might need to temporarily allow port 8501 in UFW (`sudo ufw allow 8501`) for this test and remove it later (`sudo ufw delete allow 8501`).*

## Step 4: Configure Nginx

1.  **Copy Nginx Configuration**: Copy the provided Nginx configuration template (`deployment/digitalocean/nginx_streamlit.conf`) to the Nginx sites-available directory.
    ```bash
    sudo cp deployment/digitalocean/nginx_streamlit.conf /etc/nginx/sites-available/llmpedia
    ```
2.  **Edit Configuration**: Open the copied file and replace `your_domain.com www.your_domain.com` with your actual domain name(s) (e.g., `llmpedia.ai www.llmpedia.ai`):
    ```bash
    sudo nano /etc/nginx/sites-available/llmpedia
    ```
    *(Save and close)*

3.  **Set Static File Permissions**: Set the correct ownership and permissions for the static directory so Nginx can read the files.
    ```bash
    sudo chown -R www-data:www-data /var/www/llmpedia
    sudo chmod -R 755 /var/www/llmpedia
    ```

4.  **Enable Site**: Create a symbolic link to enable the configuration:
    ```bash
    sudo ln -s /etc/nginx/sites-available/llmpedia /etc/nginx/sites-enabled/
    ```
5.  **Remove Default (Optional but Recommended)**: Remove the default Nginx configuration link if it exists:
    ```bash
    sudo rm /etc/nginx/sites-enabled/default
    ```
6.  **Test Nginx Configuration**: Check for syntax errors:
    ```bash
    sudo nginx -t
    ```
7.  **Restart Nginx**: Apply the changes:
    ```bash
    sudo systemctl restart nginx
    ```

## Step 5: Configure DNS

1.  **Go to Domain Registrar**: Log in to the service where you purchased your domain name.
2.  **Create A Record**: Find the DNS management section and create an 'A' record:
    - **Host/Name**: `@` (or your domain name, `llmpedia.ai`)
    - **Value/Points to**: Your Droplet's IP address
3.  **Create CNAME Record (Optional but Recommended)**: Create a 'CNAME' record for `www`:
    - **Host/Name**: `www`
    - **Value/Points to**: `@` (or your domain name, `llmpedia.ai`)
4.  **Wait for Propagation**: DNS changes can take time to propagate (minutes to hours).

## Step 6: Set Up SSL with Certbot

1.  **Obtain Certificate**: Run Certbot to automatically obtain an SSL certificate and configure Nginx for HTTPS. Replace `llmpedia.ai` and `www.llmpedia.ai` with your actual domain names:
    ```bash
    sudo certbot --nginx -d llmpedia.ai -d www.llmpedia.ai
    ```
2.  **Follow Prompts**: Certbot will ask for your email address (for renewal notices) and ask you to agree to the terms of service. It will also ask if you want to redirect HTTP traffic to HTTPS (recommended).
3.  **Automatic Renewal**: Certbot automatically sets up a cron job or systemd timer to renew your certificates before they expire.

## Step 7: Set Up Systemd Service

1.  **Copy Service File**: Copy the provided systemd service template (`deployment/digitalocean/streamlit_app.service`) to the systemd system directory:
    ```bash
    sudo cp deployment/digitalocean/streamlit_app.service /etc/systemd/system/streamlit_app.service
    ```
2.  **Edit Service File**: Modify the service file to match your setup:
    ```bash
    sudo nano /etc/systemd/system/streamlit_app.service
    ```
    - Replace `your_user` and `your_group` with the username and group you created in Step 1.
    - Replace `/path/to/your/llmpedia` with the *absolute path* to your application directory (e.g., `/home/your_user/llmpedia`). Make sure this path is correct for both `WorkingDirectory` and `EnvironmentFile`.
    - Ensure the path to `streamlit` in `ExecStart` is correct. If you installed it within the virtual environment, the path might be `/home/your_user/llmpedia/.venv/bin/streamlit`. You can verify with `which streamlit` *while the venv is active*.
    *(Save and close)*

3.  **Reload Systemd**: Inform systemd about the new service file:
    ```bash
    sudo systemctl daemon-reload
    ```
4.  **Enable Service**: Make the service start automatically on boot:
    ```bash
    sudo systemctl enable streamlit_app.service
    ```
5.  **Start Service**: Start the application service now:
    ```bash
    sudo systemctl start streamlit_app.service
    ```
6.  **Check Status**: Verify that the service is running correctly:
    ```bash
    sudo systemctl status streamlit_app.service
    ```
    - Look for `active (running)`. If there are errors, check the logs:
      ```bash
      sudo journalctl -u streamlit_app.service -f
      ```
      (Press `Ctrl+C` to exit the log stream).

## Step 8: Access Your Application

Open your web browser and navigate to `https://your_domain.com` (e.g., `https://llmpedia.ai`). Your LLMpedia application should now be live!

## Maintenance

- **Updating the App**: `cd ~/llmpedia`, `git pull`, `source .venv/bin/activate`, `uv pip install -r requirements.txt` (if needed), `sudo systemctl restart streamlit_app.service`.
- **Renewing SSL**: Certbot should handle this automatically. You can test renewal with `sudo certbot renew --dry-run`.
- **Viewing Logs**: `sudo journalctl -u streamlit_app.service -f`. 