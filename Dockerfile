FROM python:3.9-slim

RUN mkdir /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY ./requirements.txt /app

WORKDIR /app

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000
CMD ["python", "./bin/main.py"]



# RUN mkdir /app

# WORKDIR /app

# RUN python -m pip install fastapi==0.111.0

# COPY . /app

# EXPOSE 8000
# CMD ["fastapi", "run", "./bin/test.py"]
