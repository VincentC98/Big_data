from cassandra.cluster import Cluster
import json

# first install datastax cassandra driver for Python in the project directory
# Go to: D:\GDrive\F20\5D3.DS\TD\Colvalds\venv\Scripts\python.exe'.
# Run command: pip install cassandra-driver
from src.cassandra.connect_database import Cassandra_Connector


def run():

    key_space = "colvalds"
    table = "Restaurants"
    file = "C:\\Users\\smart\\OneDrive\\Desktop\\NYR.json"
    connector = Cassandra_Connector()
    session = connector.session
    session.execute("USE {}".format(key_space))

    with open(file) as f:
        for document in f:
            document = document.replace("'", "''")
            session.execute("""
                            insert into {} JSON '{}'
                            """.format(table, document))
    print("JSON inserted successfully on {} table".format(table))


if __name__ == '__main__':
    run()
