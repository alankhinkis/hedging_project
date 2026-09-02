"""The query cache is the reproducibility guarantee: Stage 4 will be re-run many times and
must never re-hit WRDS. These tests pin the two properties that make the cache safe --
cosmetic SQL changes must not force a re-pull, and a change in *meaning* always must.
"""

from __future__ import annotations

from vrp.wrds_conn import query_key


SQL = "SELECT permno, date FROM crsp.dsf WHERE permno IN %(permnos)s"


def test_whitespace_does_not_change_the_key():
    reformatted = """
        SELECT permno, date
        FROM   crsp.dsf
        WHERE  permno IN %(permnos)s
    """
    assert query_key(SQL, {"permnos": (1, 2)}) == query_key(reformatted, {"permnos": (1, 2)})


def test_parameter_order_does_not_change_the_key():
    a = query_key(SQL, {"permnos": (1, 2), "a": "2019-01-01"})
    b = query_key(SQL, {"a": "2019-01-01", "permnos": (1, 2)})
    assert a == b


def test_different_parameters_change_the_key():
    assert query_key(SQL, {"permnos": (1, 2)}) != query_key(SQL, {"permnos": (1, 3)})


def test_different_sql_changes_the_key():
    other = SQL.replace("crsp.dsf", "crsp.msf")
    assert query_key(SQL, {"permnos": (1,)}) != query_key(other, {"permnos": (1,)})


def test_tuple_and_list_parameters_agree():
    """psycopg2 needs a tuple for IN; a caller passing a list must not orphan the cache."""
    assert query_key(SQL, {"permnos": (1, 2)}) == query_key(SQL, {"permnos": [1, 2]})
