
FROM python:3.11-slim

# Create non-root user that matches Spaces runtime (uid 1000), set HOME
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install dependencies as the user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code with correct ownership
COPY --chown=user . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]

# NOTE: Disable XSRF protection for easier external access in order to make batch predictions
