FROM python:3.12-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		build-essential \
		curl \
	&& rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -e .[api]

# Keep this container idle; run pipeline commands manually with docker compose exec.
CMD ["tail", "-f", "/dev/null"]
