# Поиск по картинке
Финальный проект Академии Аналитиков Авито

## Описание проекта
У пользователей иногда возникает потребность в быстром и удобном поиске похожих товаров по фото. Поэтому целью проекта является разработка сервиса с веб-интерфейсом, куда можно отправить картинку и получить выдачу из объявлений с подходящими товарами.

## Название команды
Triplet A

## Команда проекта
Слесаренко Анастасия

## Запуск сервисов в продакшен режиме
### 1. Запуск и настройка FastAPI, PostgreSQL в Docker контейнерах
```bash
docker-compose -f docker-compose.yml up --build -d
```

### 2. Остановка и удаление Docker сервисов
```bash
docker-compose -f docker-compose.yml down --remove-orphans
```

## Запуск сервиса в режиме разработки
### 1. Запуск и настройка PostgreSQL в Docker контейнерах
```bash
docker-compose -f docker-compose-dev.yml up --build -d
```

### 2. Создание и активация виртуального окружения в `venv` и установка всех библиотек
```
pip install -r requirements-dev.txt
```

### 3. Запуск приложения FastAPI:
```bash
fastapi dev app.py
```
### 4. Остановка приложения FastAPI с помощью Ctrl+C

### 5. Остановка и удаление Docker сервисов
```
docker-compose -f docker-compose-dev.yml down --remove-orphans
```

## Запуск pytest
Запуск всех тестов pytest в tests/:
```
pytest -vv tests/ --show-capture=all -W ignore::DeprecationWarning
```