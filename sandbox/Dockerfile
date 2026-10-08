# Image for sprout.sandbox_server: Python 3.12 with common data libraries, plus Node.js.
# Network is switched off at run time (docker run --network none).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends nodejs coreutils sqlite3 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pandas matplotlib sympy pytest beautifulsoup4
RUN useradd -m -u 1000 runner
USER runner
WORKDIR /workspace
