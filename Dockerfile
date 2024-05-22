FROM python:3.11-slim

RUN mkdir /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY ./requirements.txt /app

WORKDIR /app

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000
CMD ["fastapi", "run", "app.py"]
