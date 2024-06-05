FROM python:3.10-slim

RUN mkdir /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY ./requirements.txt /app

WORKDIR /app

RUN python -m pip install --no-cache-dir torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000
CMD ["python", "./main.py"]



# RUN mkdir /app

# WORKDIR /app

# RUN python -m pip install fastapi==0.111.0

# COPY . /app

# EXPOSE 8000
# CMD ["fastapi", "run", "./bin/test.py"]
