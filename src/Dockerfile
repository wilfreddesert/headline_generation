FROM python:3.8-slim-buster as base

FROM base as builder
RUN apt-get update && \
    apt-get upgrade -y
RUN apt-get install -y build-essential gcc g++
WORKDIR /app
RUN python -m venv /venv 
ENV PATH="/app/venv/bin:$PATH"

COPY ./requirements.txt ./

RUN . /venv/bin/activate && \    
    pip install --upgrade pip && \    
    mkdir /libs && cd /libs && \    
    pip download -r /app/requirements.txt

RUN . /venv/bin/activate && \
    cd /app && \
    pip install -r /app/requirements.txt

COPY . .

FROM base
RUN apt-get update && \
    apt-get upgrade -y
WORKDIR /
COPY --from=builder / /
CMD . /venv/bin/activate && ./train.py