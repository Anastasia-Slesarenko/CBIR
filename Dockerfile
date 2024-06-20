FROM python:3.10-slim

RUN mkdir /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY ./requirements.txt /app

WORKDIR /app

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["/bin/bash", "-c", "python ./lib/load_artifacts/start.py;python ./main.py"]
