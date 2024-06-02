# #!/bin/bash

# Update package list and install necessary packages
# sudo apt update
# sudo apt install -y jq curl tar
# sudo apt-get install -y docker-compose
# sudo apt install -y software-properties-common
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt install -y python3.10
# sudo apt install -y python3-pip
# sudo apt install -y python3.10-venv python3.10-distutils

prestart_path=`pwd`
project_path="$(dirname "$prestart_path")"
data_ydisk_url="https://disk.yandex.ru/d/ADNIpXT6fJx74g"
app_volume_path=$project_path/data
main_docker_compose_file=$project_path/docker-compose.yml
postgresql_service="postgres"
main_app_service="retrieval_service"

mkdir -p $app_volume_path

sudo chown -R $USER $app_volume_path
sudo chmod -R 700 $app_volume_path

# # create postgresql from docker-compose
cd $project_path
sudo docker-compose -f docker-compose.yml up --build -d $postgresql_service

API_ENDPOINT="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
response=$(curl -s "$API_ENDPOINT$data_ydisk_url")
data_download_link=$(echo "$response" | jq -r '.href')

# # Download tar file
cd $prestart_path
curl -L "$data_download_link" -o my_file.tar

# # Unzip tar file to specified path
# # mkdir -p $app_volume_path
sudo tar -xf $prestart_path/my_file.tar -C $app_volume_path/
rm -rf $prestart_path/my_file.tar

# # Create and activate Python virtual environment
cd $prestart_path
python3.10 -m venv venv
source venv/bin/activate
pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python3.10 start.py
deactivate
rm -rf venv

# run app
cd $project_path
sudo docker-compose -f docker-compose.yml up --build -d $main_app_service
