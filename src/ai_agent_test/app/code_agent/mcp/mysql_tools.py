import subprocess
from typing import Annotated, List, Optional, Dict
from mcp.server.fastmcp import FastMCP
from pydantic import Field, BaseModel
import pymysql

mcp = FastMCP()

class Response(BaseModel):
    success: bool
    database: str
    table: str
    data: Optional[dict] | Optional[list]

MYSQL_CONFIG = {
    "host": "192.168.64.2",
    "port": 3306,
    "user": "root",
    "password": "root",
    'charset': 'utf8mb4',
}

def get_connection(db):
    config = MYSQL_CONFIG.copy()
    if db:
        config['database'] = db
    try:
        connection = pymysql.connect(**config)
        return connection
    except Exception as e:
        msg = f'mysql connection error: {str(e)}'
        raise Exception(msg)

def execute_query(command, database=None, parmas: Optional[list | tuple] = None):
    connection = get_connection(database)
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(command, parmas)
            if cursor.description is not None:
                # SELECT / SHOW / DESCRIBE 等有结果集的语句
                result = cursor.fetchall()
            else:
                # INSERT / UPDATE / DELETE 等没有结果集的语句
                result = {
                    "affected_rows": cursor.rowcount,
                    "insert_id": cursor.lastrowid,
                }
        connection.commit()
        return result
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

@mcp.tool(name="mysql_list_databases", description="列举MySQL中包含哪些数据库")
def mysql_list_databases():
    try:
        result = execute_query("show databases")
        databases = [row['Database'] for row in result]
        return databases
    except Exception as e:
        msg = f'mysql list databases error: {str(e)}'
        raise Exception(msg)

@mcp.tool(name="mysql_list_tables", description="获取指定数据库中的所有表")
def mysql_list_tables(database):
    try:
        result = execute_query("show tables", database=database)
        tables = [list(row.values())[0] for row in result]
        print(tables)
        return Response(
            success=True,
            database=database,
            table='',
            data=tables
        )
    except Exception as e:
        msg = f'mysql list tables error: {str(e)}'
        raise Exception(msg)

@mcp.tool(name="mysql_describe_tables", description="获取表结构信息")
def mysql_describe_tables(database, table):
    try:
        result = execute_query(f"describe {table}", database=database)
        return Response(
            success=True,
            database=database,
            table=table,
            data=result
        )
    except Exception as e:
        msg = f'mysql describe tables error: {str(e)}'
        return msg

@mcp.tool(name="mysql_execute_query", description="执行sql查询语句")
def mysql_execute_query(command, database=None, parmas: Optional[list] = None):
    try:
        params_tuple = tuple(parmas) if parmas else None
        result = execute_query(command, database=database, parmas=params_tuple)
        return Response(
            success=True,
            database=database,
            table='',
            data=result
        )
    except Exception as e:
        msg = f'mysql execute_query error: {str(e)}'
        return msg

@mcp.tool(name="mysql_insert_data", description="向指定数据库的表中插入一条数据")
def mysql_insert_data(database: str, table: str, data: Dict[str, str]):
    if not data:
        return "mysql insert data error: data is empty"

    # 表名、列名无法参数化,用反引号包裹并转义内部反引号
    def quote(name: str) -> str:
        return "`" + str(name).replace("`", "``") + "`"

    columns = ", ".join(quote(k) for k in data.keys())
    placeholders = ", ".join(["%s"] * len(data))
    sql = f"INSERT INTO {quote(table)} ({columns}) VALUES ({placeholders})"

    try:
        result = execute_query(sql, database=database, parmas=tuple(data.values()))
        return Response(
            success=True,
            database=database,
            table=table,
            data=result
        )
    except Exception as e:
        return f'mysql insert data error: {str(e)}'

@mcp.tool(name="mysql_update_data", description="根据条件更新指定数据库表中的数据,where 中的多个条件用 AND 连接")
def mysql_update_data(database: str, table: str, data: Dict[str, str], where: Dict[str, str]):
    if not data:
        return "mysql update data error: data is empty"
    if not where:
        return "mysql update data error: where is empty, refuse to update the whole table"

    # 表名、列名无法参数化,用反引号包裹并转义内部反引号
    def quote(name: str) -> str:
        return "`" + str(name).replace("`", "``") + "`"

    set_clause = ", ".join([f"{quote(k)} = %s" for k in data.keys()])
    where_clause = " AND ".join([f"{quote(k)} = %s" for k in where.keys()])
    command = f"UPDATE {quote(table)} SET {set_clause} WHERE {where_clause}"

    set_params = tuple(data.values())
    where_params = tuple(where.values())
    params = set_params + where_params

    try:
        result = execute_query(command, database=database, parmas=params)
        return Response(
            success=True,
            database=database,
            table=table,
            data=result
        )
    except Exception as e:
        return f'mysql update data error: {str(e)}'

@mcp.tool(name="mysql_delete_data", description="根据条件删除指定数据库表中的数据,data 中的多个条件用 AND 连接")
def mysql_delete_data(database: str, table: str, data: Dict[str, str]):
    if not data:
        return "mysql delete data error: data is empty, refuse to delete the whole table"

    # 表名、列名无法参数化,用反引号包裹并转义内部反引号
    def quote(name: str) -> str:
        return "`" + str(name).replace("`", "``") + "`"

    where_clause = " AND ".join([f"{quote(k)} = %s" for k in data.keys()])
    command = f"DELETE FROM {quote(table)} WHERE {where_clause}"
    params = tuple(data.values())

    try:
        result = execute_query(command, database=database, parmas=params)
        return Response(
            success=True,
            database=database,
            table=table,
            data=result
        )
    except Exception as e:
        return f'mysql delete data error: {str(e)}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
    # print(mysql_list_databases())
    # print(mysql_list_tables("test"))
    # print(mysql_describe_tables('test', 'user'))
    # mysql_execute_query('select * from user where name=%s', database='test', parmas=["wxy"])
    # print(mysql_insert_data("test", "user", {"id": "4", "name": "olivia"}))
    # mysql_update_data("test", "user", {"name": "lily"}, {"id": "3"})