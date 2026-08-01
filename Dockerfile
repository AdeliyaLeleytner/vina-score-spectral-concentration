FROM python:3.12.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       make texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . /work
RUN pip install --no-cache-dir -r requirements.lock

CMD ["make", "all"]
