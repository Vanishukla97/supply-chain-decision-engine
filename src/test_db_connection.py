
import getpass
import psycopg

password = getpass.getpass("Enter your PostgreSQL postgres password: ")

try:
    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="supply_chain_db",
        user="postgres",
        password=password
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), version();")
            database, version = cur.fetchone()

            print("\nDatabase connection successful!")
            print("Database:", database)
            print("Version:", version.split(",")[0])

except psycopg.OperationalError as error:
    print("\nConnection failed. Check the password and service.")
    print(error)
