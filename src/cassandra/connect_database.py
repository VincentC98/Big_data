from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider


class Cassandra_Connector(object):

    cloud_config = {'secure_connect_bundle':
                    'D:\\GDrive\\F20\\5D3.DS\\TD\\Colvalds\\ressources\\''secure-connect-colval.zip'}

    def __init__(self):

        self.auth_provider = PlainTextAuthProvider('colval', 'Colval2020')
        self.cluster = Cluster(cloud=Cassandra_Connector.cloud_config, auth_provider=self.auth_provider)
        self.session: Session = self.cluster.connect()

        row = self.session.execute("select release_version from system.local").one()
        if row:
            print(row[0])
        else:
            print("An error occurred.")

# Output: 4.0.0.682
