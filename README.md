# Поиск по картинке
Финальный проект Академии Аналитиков Авито

## Описание проекта
У пользователей иногда возникает потребность в быстром и удобном поиске похожих товаров по фото. Поэтому целью проекта является разработка сервиса с веб-интерфейсом, куда можно отправить картинку и получить выдачу из объявлений с подходящими товарами.

## Название команды
Triplet A

## Команда проекта
Слесаренко Анастасия

## Запуск сервиса в продакшен режиме
### 1.1. Запуск с подготовленной базой данных и индексами FAISS
```bash
cd CBIR/load_artifacts && source setup.sh
```

### 1.2. Запуск с пустой базой данных
```bash
docker-compose -f docker-compose.yml up --build -d
```

### 2. Остановка и удаление Docker сервисов
```bash
docker-compose -f docker-compose.yml down --remove-orphans
```

## Запуск сервиса для тестирования
### 1. Запуск и настройка PostgreSQL в Docker контейнерах
```bash
docker-compose -f docker-compose-test.yml up --build -d
```
### 2. Остановка и удаление Docker сервисов
```bash
docker-compose -f docker-compose-test.yml down --remove-orphans
```