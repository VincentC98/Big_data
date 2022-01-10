import os

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


os.environ['HADOOP_HOME'] = "C:\\Users\\chart\\Desktop\\Automne 2021\\Analyse d'affaire\\TD\\spark-3.1.2-bin-hadoop3.2\\bin"
spark = SparkSession\
                    .builder\
                    .appName("Uber LS App Driver...")\
                    .master("local[4]")\
                    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(spark.sparkContext.getConf().getAll())

# US Accidents (3.5 million records)
# A Countrywide Traffic Accident Dataset (2016 - 2020)
# data source : https://www.kaggle.com/sobhanmoosavi/us-accidents
# https://www.kaggle.com/siddharthcha519810/us-accidents-analysis
# https://leo-you.github.io/US_Accident_Analysis/

input_file = "../../resources/data/US_Accident_June20_500.csv"
model_file = "../../resources/model/us_accident_model"

raw_data_df = spark.read.option("header", "true").csv(input_file)

raw_data_df.printSchema()
print(f'Number of rows: {raw_data_df.count()}')  # 500
raw_data_df.show(5, truncate=False)
raw_data_df.createOrReplaceTempView("USAccident")

fact_table_query = 'SELECT City, County, ' \
                   'cast(Start_Lat as float) Start_Lat, ' \
                   'cast(End_Lat as float) End_Lat, ' \
                   'cast(Start_Lng as float) Start_Lng, ' \
                   'cast(End_Lng as float) End_Lng ' \
                   'FROM USAccident'

fact_table = spark.sql(fact_table_query)
fact_table.show()


# Combine multiple input columns to a Vector using Vector Assembler utility
features = ['Start_Lat', 'End_Lat', 'Start_Lng', 'End_Lng']
vectorAssembler = VectorAssembler(inputCols=features, outputCol='features')
fact_df = vectorAssembler.transform(fact_table)
fact_df = fact_df.select(['features'])
fact_df.show(5, truncate=False)


training_dataset, test_dataset = fact_df.randomSplit([0.9, 0.1])

kmeans = KMeans().setK(8).setFeaturesCol("features").setPredictionCol("prediction")
kmeans_model = kmeans.fit(training_dataset)

# Summarize the model over the training set and print out some metrics

print('Cluster centers are: ');
for center in kmeans_model.clusterCenters():
    print(center)


# Make predictions.
predictions = kmeans_model.transform(test_dataset)
# show all predictions
predictions.show(5)

print('How many car accidents occurred in each cluster?')
predictions.groupBy("prediction").count().show()

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
wcss = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(wcss))



spark.stop()
print("all ok.")
