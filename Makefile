# Короткие команды для сборки, запуска и проверки сервиса.
# Compose v1 по умолчанию; для v2 — make up COMPOSE="docker compose"
COMPOSE ?= docker-compose
SERVICE := retrieval_service

.PHONY: help up gpu down catalog index test lint format

help:  ## список команд
	@awk 'BEGIN{FS=":.*## "} /^[a-z][a-z-]*:.*## /{printf "  make %-9s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up:  ## запустить сервис (CPU)
	$(COMPOSE) up -d --build

gpu:  ## запустить сервис на GPU (нужен NVIDIA Container Toolkit)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

down:  ## остановить сервис
	$(COMPOSE) down

catalog:  ## скачать каталог с Яндекс.Диска в data/
	$(COMPOSE) exec $(SERVICE) python -c "import requests, tarfile; api='https://cloud-api.yandex.net/v1/disk/public/resources/download'; href=requests.get(api, params={'public_key':'https://disk.yandex.ru/d/FrMFRUVfAaknsA'}).json()['href']; r=requests.get(href, stream=True); r.raw.decode_content=True; tarfile.open(fileobj=r.raw, mode='r|*').extractall('/app/data')"

index:  ## эмбеддинги в PostgreSQL и сборка FAISS-индекса
	$(COMPOSE) exec $(SERVICE) bash -c "cd load_artifacts && python start.py"
	$(COMPOSE) restart $(SERVICE)

test:  ## прогнать тесты в контейнере
	$(COMPOSE) -f docker-compose-test.yml up --build --exit-code-from $(SERVICE)_test

lint:  ## проверить код (ruff)
	ruff check .
	ruff format --check .

format:  ## отформатировать код (ruff)
	ruff format .
