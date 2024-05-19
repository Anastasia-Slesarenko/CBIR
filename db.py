import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import Error
import numpy as np
import logging
from typing import Generator

HOSTNAME = "localhost"
USERNAME = "postgres"
PASSWORD = "postgres"
DATABASE_NAME = "db"
PORT = 5432


def get_postgres_connection():

    try:
        connection = psycopg2.connect(
            user=USERNAME,
            password=PASSWORD,
            host=HOSTNAME,
            port=PORT,
            database=DATABASE_NAME,
        )
        logging.info("Datebase connection established")
        return connection

    except (Exception, Error) as e:
        logging.error("Failed to connect to the database: %s", e)
        return {"error": e}, 400


def create_tables_structure() -> None:

    create_table_query = """
        CREATE TABLE if not exists image_descriptor (
            id SERIAL,
            embedding FLOAT ARRAY NOT NULL,
            path TEXT NOT NULL
        );
    """

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(create_table_query)
            connection.commit()
            logging.info("The table has been created successfully")


def save_embeddings(insert_embeddings: list) -> None:

    insert_query = """
        INSERT INTO image_descriptor (embedding, path)
        VALUES %s;
        """

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, insert_query, insert_embeddings)
            connection.commit()
            logging.info("The data has been successfully inserted into the table")


def count_rows() -> int:

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


def get_all_emb_from_pg(batch_size: int) -> Generator[np.array, None, None]:
    n_rows = count_rows()
    for step in range(0, n_rows, batch_size):
        yield get_batch_from_pg(limit=batch_size, offset=step)


def get_image_by_index(index_list: list[int]) -> list[str]:
    """принимает ранжированные индексы похожих изображений
    возвращает ссылки на изображения в указанном порядке"""
    query = []
    for idx in index_list:
        query.append(
            f"""(
            select
                path
            from image_descriptor
            order by id
            LIMIT 1 offset {idx}
        )"""
        )
    query = "\nunion all\n".join(query)
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    return rows
