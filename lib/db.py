import logging
from typing import Generator
from psycopg2.extras import execute_values
from psycopg2 import Error
from psycopg2.pool import SimpleConnectionPool
import numpy as np


class Storage:
    def __init__(self, user, password, host, port, database):
        self._pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            user=user,
            password=password,
            host=host,
            port=port,
            dbname=database,
        )

    def disconnect(self) -> None:
        self._pool.closeall()

    def get_connection(self):
        try:
            return self._pool.getconn()
        except (Exception, Error) as e:
            logging.error("Failed to connect to the database: %s", e)
            return None

    def create_tables_structure(self) -> None:
        """Создает таблицу, если ее не существует"""
        create_table_query = """
            CREATE TABLE IF NOT EXISTS image_descriptor (
                id SERIAL PRIMARY KEY,
                image_id INT NOT NULL,
                embedding FLOAT ARRAY NOT NULL,
                item_url TEXT NOT NULL,
                title TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS image_id_btree ON image_descriptor (image_id);
        """
        connection = self.get_connection()
        with connection.cursor() as cursor:
            cursor.execute(create_table_query)
            connection.commit()
        self._pool.putconn(connection)

    def save_embeddings(self, insert_embeddings: list) -> None:
        """Записывает данные о изображениях и их эмбеддинги в таблицу"""
        insert_query = """
            INSERT INTO image_descriptor (image_id, embedding, item_url, title)
            VALUES %s;
        """
        connection = self.get_connection()
        with connection.cursor() as cursor:
            execute_values(cursor, insert_query, insert_embeddings)
            connection.commit()
        self._pool.putconn(connection)

    def count_rows(self) -> int:
        """Считает количество строк в таблице"""
        query = """
            SELECT COUNT(*)
            FROM image_descriptor;
        """
        connection = self.get_connection()
        with connection.cursor() as cursor:
            cursor.execute(query)
            res = cursor.fetchone()
        self._pool.putconn(connection)
        return res[0]

    def get_batch_from_pg(
        self, limit: int, offset: int
    ) -> tuple[list, np.array]:
        """Отдает батч заданного размера"""
        query = """
            SELECT image_id, embedding
            FROM image_descriptor
            ORDER BY id
            LIMIT %s
            OFFSET %s;
        """
        connection = self.get_connection()
        with connection.cursor() as cursor:
            cursor.execute(query, (limit, offset))
            res = cursor.fetchall()
        self._pool.putconn(connection)
        ids = np.array([np.float32(el[0]) for el in res])
        batch = np.array([el[1] for el in res])
        return ids, batch

    def get_all_emb_from_pg(
        self,
        batch_size: int,
    ) -> Generator[np.array, None, None]:
        """Генератор, который извлекает эмбеддинги из базы данных пакетами."""
        n_rows = self.count_rows()
        for step in range(0, n_rows, batch_size):
            yield self.get_batch_from_pg(limit=batch_size, offset=step)

    def get_image_by_index(self, index_list: list[int]) -> list[str]:
        """Принимает ранжированные индексы похожих изображений и
        возвращает ссылки на изображения в указанном порядке"""
        query = f"""(
            SELECT
                image_id,
                item_url,
                title
            FROM image_descriptor
            WHERE image_id in ({', '.join([str(idx) for idx in index_list])})
        )"""
        connection = self.get_connection()
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        self._pool.putconn(connection)
        image_id_dict = {row[0]: row for row in rows}
        order_rows = [image_id_dict[image_id] for image_id in index_list]
        return order_rows
