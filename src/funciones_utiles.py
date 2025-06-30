import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener configuración de la base de datos desde el archivo .env
DB_NAME = os.getenv("DB_NAME")
DB_PATH = os.getenv("DB_PATH")
DATA_PATH = os.getenv("DATA_PATH")



def cargar_csv_como_tabla(nombre_archivo):
    """
    Carga un archivo CSV como una tabla en DuckDB y devuelve el dataset como un DataFrame.
    
    Args:
        nombre_archivo (str): Nombre del archivo CSV a cargar (incluyendo extensión).
    
    Returns:
        pd.DataFrame: Dataset cargado desde el archivo CSV.
    """
    #Ruta a los datasets
    ruta_csv = os.path.join(DATA_PATH, nombre_archivo)
    
    #Saco la extension del archivo al nombre de tabla
    nombre_tabla = os.path.splitext(nombre_archivo)[0]

    #conexion a la base
    db_connection = duckdb.connect(f"{DB_PATH}/{DB_NAME}")
    
    #Create or replace
    query = f"""
    CREATE OR REPLACE TABLE {nombre_tabla} AS 
    SELECT * FROM read_csv_auto('{ruta_csv}');
    """
    db_connection.execute(query)
    db_connection.close()
    
    
    # Leer la tabla como DataFrame y devolverla
    return print(f"Tabla '{nombre_tabla}' creada o reemplazada en DuckDB.")

def ejecutar_query(query):
    """
    Ejecuta una consulta SQL en la base de datos DuckDB.
    
    Args:
        query (str): Consulta SQL a ejecutar.
    
    Returns:
        pd.DataFrame | None: Resultados de la consulta si return_result es True, de lo contrario None.
    """
    db_connection = duckdb.connect(f"{DB_PATH}/{DB_NAME}")

    df = db_connection.execute(query).df()
    
    # Si la consulta incluye un CREATE TABLE, extraer el nombre de la tabla y devolver los datos
    if "CREATE OR REPLACE TABLE" in query.upper():
        table_name = query.split("TABLE")[1].split("AS")[0].strip()
        df = db_connection.execute(f"SELECT * FROM {table_name}").df()
 
    db_connection.close()
    return df

def cargar_df_a_db(df, nombre_tabla):
    """
    Carga un DataFrame en DuckDB como una tabla, creando o reemplazando la tabla si ya existe.
    
    Args:
        df (pd.DataFrame): El DataFrame que se desea cargar en la base de datos.
        nombre_tabla (str): El nombre de la tabla en la base de datos.
    
    Returns:
        None
    """
    # Validar que el DataFrame no esté vacío
    if df.empty:
        raise ValueError("El DataFrame está vacío y no se puede cargar en la base de datos.")
    
    # Cargar el DataFrame en DuckDB
    # hacer que no se muestren los warnings de pandas
    pd.options.mode.chained_assignment = None
    db_connection = duckdb.connect(f"{DB_PATH}/{DB_NAME}")
    df.to_sql(nombre_tabla, con=db_connection, if_exists="replace", index=False)
    db_connection.close()
    return print(f"Tabla '{nombre_tabla}' creada o reemplazada en DuckDB.")
