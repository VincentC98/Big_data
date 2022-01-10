import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, IndexToString
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.environ['HADOOP_HOME'] = "C:\\Users\\chart\\Desktop\\Automne 2021\\Analyse d'affaire\\TD\\spark-3.1.2-bin-hadoop3.2"
spark = SparkSession\
                    .builder\
                    .appName("App Driver...")\
                    .master("local[4]")\
                    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(spark.sparkContext.getConf().getAll())

#  House Sales King County
# data source : https://www.kaggle.com/ambarish/tutorial-housesales-kingcounty-eda-modelling/data

input_file = "../resources/data/room_occupancy.csv"
model_file = "/resources/model/room_occupancy_model"

raw_data_df = spark.read.option("header", "true").csv(input_file)

raw_data_df.printSchema()
print(f'Number of rows: f{raw_data_df.count()}')
raw_data_df.show(5, truncate=False)