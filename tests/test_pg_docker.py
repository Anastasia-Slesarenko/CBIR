import pytest


@pytest.mark.order(1)
def test_pg_connection(mock_db):
    connection = mock_db.get_postgres_connection()
    assert not isinstance(connection, tuple)
    connection.close()
