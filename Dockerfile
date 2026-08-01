FROM python:3.12.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       make texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.lock /work/requirements.lock
RUN pip install --no-cache-dir -r requirements.lock
COPY . /work

CMD ["make", "all"]
