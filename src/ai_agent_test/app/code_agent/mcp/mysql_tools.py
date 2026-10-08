from typing import Optional, Dict, Annotated, List
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import pymysql
import re
from pydantic import Field

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

@mcp.tool(name="mysql_create_database", description="创建数据库,如果已存在则不重复创建")
def mysql_create_database(database_name: str, charset: str = "utf8mb4"):
    # 库名、字符集无法参数化,这里做白名单校验,避免拼接 SQL 时被注入
    if not re.fullmatch(r"[A-Za-z0-9_$]+", database_name):
        return "mysql create database error: invalid database name, only letters, digits, '_' and '$' are allowed"
    if not re.fullmatch(r"[A-Za-z0-9_]+", charset):
        return "mysql create database error: invalid charset"

    command = f"CREATE DATABASE IF NOT EXISTS `{database_name}` DEFAULT CHARACTER SET {charset}"

    try:
        execute_query(command)
        return Response(
            success=True,
            database=database_name,
            table='',
            data={"charset": charset}
        )
    except Exception as e:
        return f'mysql create database error: {str(e)}'

@mcp.tool(
    name="mysql_create_table",
    description=(
        "在指定数据库中创建数据表(已存在则不重复创建),创建成功后返回实际的表结构。"
        "调用前请先用 mysql_list_databases 确认数据库存在。"
    ),
)
def mysql_create_table(
    database: Annotated[str, Field(
        description="要在哪个数据库中建表,该数据库必须已存在",
        examples=["test"],
    )],
    table_name: Annotated[str, Field(
        description="表名,只能包含字母、数字、下划线和 $",
        examples=["user"],
    )],
    table_columns: Annotated[List[str], Field(
        description=(
            "建表语句中的字段定义列表,每个元素是一个字段(或索引/约束)的完整定义,"
            "格式为 '字段名 类型 [约束] [COMMENT ...]'。不要包含外层括号和表名,也不要带分号"
        ),
        examples=[[
            "id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键'",
            "name VARCHAR(50) NOT NULL COMMENT '姓名'",
            "age INT COMMENT '年龄'",
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'",
        ]],
        min_length=1,
    )],
    table_schema: Annotated[str, Field(
        description=(
            "表级选项,拼接在字段定义的右括号之后,如存储引擎、字符集、表注释。"
            "不需要时保持默认值即可"
        ),
        examples=["ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表'"],
    )] = "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
):
    if not database:
        return "mysql create table error: database is empty"
    if not re.fullmatch(r"[A-Za-z0-9_$]+", table_name):
        return "mysql create table error: invalid table name, only letters, digits, '_' and '$' are allowed"
    if not table_columns or not all(c and c.strip() for c in table_columns):
        return "mysql create table error: table_columns is empty or contains blank items"
    if any(";" in c for c in table_columns) or ";" in table_schema:
        return "mysql create table error: ';' is not allowed in table_columns or table_schema"

    columns_sql = ",\n    ".join(c.strip() for c in table_columns)
    command = f"CREATE TABLE IF NOT EXISTS `{table_name}` (\n    {columns_sql}\n) {table_schema}"

    try:
        execute_query(command, database=database)
        columns = execute_query(f"DESCRIBE `{table_name}`", database=database)
        return Response(
            success=True,
            database=database,
            table=table_name,
            data=columns
        )
    except Exception as e:
        return f'mysql create table error: {str(e)}'

@mcp.tool(
    name="mysql_execute_command",
    description=(
        "执行特定的 SQL 语句,如:变更表结构(ALTER TABLE 新增/修改/删除字段、添加索引)、"
        "增删改数据等写操作,执行后会自动提交。"
        "查询数据请使用 mysql_execute_query;"
        "单条数据的增删改优先使用 mysql_insert_data / mysql_update_data / mysql_delete_data。"
        "一次只能执行一条语句,语句末尾不要带分号。"
    ),
)
def mysql_execute_command(
    database: Annotated[str, Field(
        description="要在哪个数据库中执行该语句,该数据库必须已存在",
        examples=["test"],
    )],
    command: Annotated[str, Field(
        description="要执行的单条完整 SQL 语句,不要带分号",
        examples=[
            "ALTER TABLE `user` ADD COLUMN `email` VARCHAR(100) COMMENT '邮箱'",
            "ALTER TABLE `user` MODIFY COLUMN `age` SMALLINT",
            "ALTER TABLE `user` DROP COLUMN `email`",
            "CREATE INDEX idx_name ON `user` (`name`)",
        ],
    )],
):
    if not database:
        return "mysql execute command error: database is empty"
    if not command or not command.strip():
        return "mysql execute command error: command is empty"

    sql = command.strip().rstrip(";").strip()
    if ";" in sql:
        return "mysql execute command error: only one statement is allowed, ';' is not allowed"

    try:
        result = execute_query(sql, database=database)
        return Response(
            success=True,
            database=database,
            table='',
            data=result
        )
    except Exception as e:
        return f'mysql execute command error: {str(e)}'

if __name__ == "__main__":
    mcp.run(transport="stdio")