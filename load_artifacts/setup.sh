# #!/bin/bash

# Update package list and install necessary packages
sudo apt update
sudo apt install -y jq curl tar
sudo apt-get install -y docker-compose
sudo apt install -y python3-pip
sudo apt install python3.8-venv

prestart_path=`pwd`
cd ..
app_path=`pwd`
data_ydisk_url="https://disk.yandex.ru/d/6mlmK_XWqlK08A"
app_volume_path=$app_path/data
main_docker_compose_file=$app_path/docker-compose.yml

# # create postgresql from docker-compose
# cd ~
# sudo docker-compose -f $main_docker_compose_file up --build -d

API_ENDPOINT="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
response=$(curl -s "$API_ENDPOINT$data_ydisk_url")
data_download_link=$(echo "$response" | jq -r '.href')

# Download tar file
cd $prestart_path
curl -L "$data_download_link" -o my_file.tar

# Unzip tar file to specified path
cd ~
mkdir -p $app_volume_path
tar -xf $prestart_path/my_file.tar -C $app_volume_path/
rm -rf $prestart_path/my_file.tar

# # Create and activate Python virtual environment
# cd ~
# cd $prestart_path
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# python3 start.py
# deactivate
# rm -rf venv
