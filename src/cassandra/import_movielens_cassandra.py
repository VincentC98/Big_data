from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
import pandas as pandas

# first install datastax cassandra driver for Python in the project directory
# Go to: D:\GDrive\F20\5D3.DS\TD\Colvalds\venv\Scripts\python.exe'.
# Run command: pip install cassandra-driver
from src.cassandra.connect_database import Cassandra_Connector

key_space = "colvalds"
table = "Restaurants"
file_path = "/ressources/data/movies.csv"
connector = Cassandra_Connector()
session = connector.session
session.execute("USE {}".format(key_space))


def run_ddl():
    # Developer : Amine 2020-06-25 - 13:04
    input_data = pandas.read_csv(file_path, header=0)
    # print(input_data.head(5))
    # len(input_data)
    for i in range(len(input_data)):
        movie_id = input_data.iloc[i]['movieId']
        title = input_data.iloc[i]['title']
        #genres = input_data.iloc[i]['genres'].split('|')
        genres = input_data.iloc[i]['genres']
        session.execute("""
                        INSERT INTO movies(movie_id, title, genres)  
                        VALUES (%s, %s, %s)
                        """,
                        (movie_id, title, genres)
                        )
        print("Inserted successfully ", movie_id, title, genres)


def run_dml():
    movie_id = 1
    movie_lookup_stmt = session.prepare("SELECT * FROM movies WHERE movie_id=?")
    movie_lookup_stmt.consistency_level = ConsistencyLevel.QUORUM
    movie = session.execute(movie_lookup_stmt, [movie_id]).one()
    print(movie)


if __name__ == '__main__':
    #run_ddl()
    run_dml()