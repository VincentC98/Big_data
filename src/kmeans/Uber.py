import os

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

os.environ['HADOOP_HOME'] = "C:\\Users\\chart\\Desktop\\Automne 2021\\Analyse d'affaire\\TD\\spark-3.1.2-bin-hadoop3.2\\bin"
ss = SparkSession\
                    .builder\
                    .appName("Uber LS App Driver...")\
                    .master("local[4]")\
                    .getOrCreate()

ss.sparkContext.setLogLevel("ERROR")
print(ss.sparkContext.getConf().getAll())


input_file = "../../resources/data/uber.csv"
model_file = "../../resources/model/uber.csv/uber_kmeans_model"

#chargez les données dans un dataframe avec les noms de colonnes indiqués
raw_data_df = ss.read.option("header", "true").option("delimiter", ",").option("inferschema", "true").format("csv").load(input_file).toDF("base", "dt", "lat", "lon")

raw_data_df.printSchema() #affiche le schéma
raw_data_df.show(2, truncate=False) #affiche les deux premières lignes
raw_data_df.createOrReplaceTempView("Uber")
sqlResult = ss.sql("SELECT COUNT(base) FROM Uber")
sqlResult.show() #affiche le nombre de lignes dans le dataframe

assembler = VectorAssembler(inputCols=["lat", "lon"], outputCol="features") # Création d'un vecteur de featueres composé des deux colonnes lat et lon
fact_table = assembler.transform(raw_data_df) #création de la table d'entrainement
fact_table.show(10, False) #affiche les dix premières lignes de la table d'entrainement

training_data, test_data = fact_table.randomSplit([0.8, 0.2]) # Partitionnement des données d'entrainement et données de test à raison de 90

# Trains a k-means model
kmeans: KMeans = KMeans().setK(8).setSeed(1).setFeaturesCol("features").setPredictionCol("prediction")
model: KMeansModel = kmeans.fit(training_data)


print("Cluster Centers: ")
for center in model.clusterCenters():
    print(center)

# Make predictions
predictions = model.transform(test_data)
predictions .show(5, False)
print("How many pickups occurred in each cluster?")
predictions.groupBy("prediction").count().show()

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
wcss = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(wcss))

print("Operating the prediction model:")
print(str(model.predict(Vectors.dense(234.0, 1652.0))))
print(str(model.predict(Vectors.dense(1035.0, 8041.0))))
ss.stop()

print("all ok.")

