import logging
from typing import Generator
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import Error
import numpy as np

from settings import (
    HOSTNAME,
    USERNAME,
    PASSWORD,
    DATABASE_NAME,
    PORT,
)


def get_postgres_connection():
    """Устанавливает соединения с pg"""
    print(HOSTNAME)

    try:
        connection = psycopg2.connect(
            user=USERNAME,
            password=PASSWORD,
            host=HOSTNAME,
            port=PORT,
            database=DATABASE_NAME,
        )
        return connection

    except (Exception, Error) as e:
        logging.error("Failed to connect to the database: %s", e)
        return {"error": e}, 400


def create_tables_structure() -> None:
    """Создает таблицу, если ее не существует"""
    create_table_query = """
        CREATE TABLE IF NOT EXISTS image_descriptor (
            id SERIAL PRIMARY KEY,
            embedding FLOAT ARRAY NOT NULL,
            source TEXT NOT NULL,
            item_url TEXT NOT NULL,
            title TEXT NOT NULL
        );
    """

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(create_table_query)
            connection.commit()


def save_embeddings(insert_embeddings: list) -> None:
    """Записывает данные о изображениях и их эмбеддинги в таблицу"""
    insert_query = """
        INSERT INTO image_descriptor (embedding, source, item_url, title)
        VALUES %s;
    """

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, insert_query, insert_embeddings)
            connection.commit()


def count_rows() -> int:
    """Считает количество строк в таблице"""
    query = """
        SELECT COUNT(*)
        FROM image_descriptor;
    """

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            res = cursor.fetchone()
    return res[0]


def get_batch_from_pg(limit: int, offset: int) -> np.array:
    """Отдает батч заданного размера"""
    query = """
        SELECT embedding
        FROM image_descriptor
        ORDER BY id
        LIMIT %s
        OFFSET %s;
    """

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (limit, offset))
            res = cursor.fetchall()
    return np.array(res).squeeze(1)


def get_all_emb_from_pg(
    batch_size: int,
) -> Generator[np.array, None, None]:
    """Генератор, который извлекает эмбеддинги из базы данных пакетами."""
    n_rows = count_rows()
    for step in range(0, n_rows, batch_size):
        yield get_batch_from_pg(limit=batch_size, offset=step)


def get_image_by_index(index_list: list[int]) -> list[str]:
    """Принимает ранжированные индексы похожих изображений и
    возвращает ссылки на изображения в указанном порядке"""
    query = []
    for idx in index_list:
        query.append(
            f"""(
            SELECT
                source,
                item_url,
                title
            FROM image_descriptor
            ORDER BY id
            LIMIT 1 offset {idx}
        )"""
        )
    query = "\nunion all\n".join(query)
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    return rows
