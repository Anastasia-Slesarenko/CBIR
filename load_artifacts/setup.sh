# #!/bin/bash

# Update package list and install necessary packages
# sudo apt update
# sudo apt install -y jq curl tar 
# sudo apt-get install -y docker-compose
# sudo apt install -y python3-pip
sudo apt install python3.8-venv

# data_ydisk_url="https://disk.yandex.ru/d/6mlmK_XWqlK08A"
# app_volume_path="app"
# main_docker_compose_file="app/docker-compose.yml"
# postgresql_service="postgres"
# main_app_service="retrieval_service"
prestart_path=`pwd`

# API_ENDPOINT="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
# response=$(curl -s "$API_ENDPOINT$data_ydisk_url")
# data_download_link=$(echo "$response" | jq -r '.href')

# # Download tar file
# curl -L "$data_download_link" -o my_file.tar

# # Unzip tar file to specified path
# cd ~
# mkdir -p $app_volume_path
# tar -xf $prestart_path/my_file.tar -C $app_volume_path/

# # create postgresql from docker-compose
# cd ~
# sudo docker-compose -f $main_docker_compose_file up --build -d $postgresql_service

# Create and activate Python virtual environment
cd ~
cd $prestart_path
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 start.py
deactivate
rm -rf venv

# # create postgresql from docker-compose
# docker compose -f $main_docker_compose_file up --build -d $main_app_service