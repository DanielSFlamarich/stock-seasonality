import typing

import awswrangler as wr
import pandas as pd


def get_sql_query(path: str, option: str = "r") -> str:
    """
    Get SQL query from a file using a context manager.

    Args:
        path (str): Path to the query file.
        option (str): File open mode. Default is 'r'.

    Returns:
        str: SQL text query.
    """
    with open(path, option) as q:
        query = q.read()
        return query


def get_data(
    glue_connection: str,
    sql: str,
    connection_type: str = "redshift",
) -> typing.Optional[pd.DataFrame]:
    """
    Get a DataFrame by executing SQL via a Glue connection.

    Args:
        glue_connection (str): Glue catalog connection name.
        sql (str): SQL statement to execute.
        connection_type (str, optional): "mysql" or "redshift". Defaults to "redshift".

    Returns:
        pd.DataFrame | None: Query results.
    """
    if connection_type == "redshift":
        with wr.redshift.connect(glue_connection) as con_db:
            df = wr.redshift.read_sql_query(sql, con=con_db)
            return df
    elif connection_type == "mysql":
        with wr.mysql.connect(glue_connection) as con_db:
            df = wr.mysql.read_sql_query(sql, con=con_db)
            return df


def execute_multiple_raw_sqls(
    sql_stmts: typing.List[str],
    glue_db_connection: str,
    connection_type: str,
):
    """
    Execute raw SQL queries without returning data.

    Args:
        sql_stmts (List[str]): List of SQL queries.
        glue_db_connection (str): Glue catalog connection name.
        connection_type (str): "mysql" or "redshift".
    """
    if connection_type == "redshift":
        with wr.redshift.connect(glue_db_connection) as con_db:
            with con_db.cursor() as cursor:
                for stmt in sql_stmts:
                    cursor.execute(stmt)
    elif connection_type == "mysql":
        with wr.mysql.connect(glue_db_connection) as con_db:
            with con_db.cursor() as cursor:
                for stmt in sql_stmts:
                    cursor.execute(stmt)


def print_multiple_raw_sqls(sql_stmts: typing.List[str]):
    """
    Print each SQL statement from a list.

    Args:
        sql_stmts (List[str]): List of SQL statements.
    """
    for stmt in sql_stmts:
        print(stmt)


def save_dataframe_to_s3_parquet(df: pd.DataFrame, destination_file_path: str):
    """
    Save a DataFrame to S3 as a Parquet file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        destination_file_path (str): S3 path where the Parquet file will be stored.
    """
    wr.s3.to_parquet(df, path=destination_file_path)
