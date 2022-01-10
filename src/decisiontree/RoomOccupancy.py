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

input_file = "../../resources/data/room_occupancy.csv"
model_file = "/resources/model/room_occupancy_model"

raw_data_df = spark.read.option("header", "true").csv(input_file)

raw_data_df.printSchema()
print(f'Number of rows: f{raw_data_df.count()}')
raw_data_df.show(5, truncate=False)
raw_data_df.createOrReplaceTempView("RoomOccupancy")

fact_table_query = 'SELECT cast(Temperature as float) Temperature, ' \
        'cast(Humidity as float) Humidity, ' \
        'cast(Light as float) Light, ' \
        'cast(CO2 as float) CO2, ' \
        'cast(HumidityRatio as float) HumidityRatio, ' \
        'cast(Occupancy as int) Occupancy ' \
        'FROM RoomOccupancy'

fact_table = spark.sql(fact_table_query)
fact_table.show()

# Combine multiple input columns to a Vector using Vector Assembler utility
features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
vectorAssembler = VectorAssembler(inputCols=features, outputCol='Features')
fact_df = vectorAssembler.transform(fact_table)
fact_df = fact_df.select(['Features', 'Occupancy'])
fact_df.show(5, truncate=False)


training_df, test_df = fact_df.randomSplit([0.7, 0.3])
dt = DecisionTreeClassifier().setLabelCol('Occupancy').setFeaturesCol('Features')


# Chain indexers and tree in a Pipeline.
pipeline = Pipeline().setStages([dt])
# Train model. This also runs the indexers.
model = pipeline.fit(training_df)
predictions = model.transform(test_df)

# Select example rows to display
# Example records with Predicted Occupancy as 0
print("Example records with Predicted Occupancy as 0:")
predictions.select("features", "Occupancy", "prediction")\
    .where(F.col('prediction') == 0).show(10)

# Example records with Predicted Occupancy as 1
print("Example records with Predicted Occupancy as 1:")
predictions.select("features", "Occupancy", "prediction")\
    .where(F.col('prediction') == 1).show(10)

# Example records with In-correct predictions
print("Example records with Predicted Occupancy not equal to recorded Occupancy:")
predictions.select("features", "Occupancy", "prediction")\
    .where(F.col('prediction') != (F.col("Occupancy"))).show(10)

# ['Humidity', 'Light', 'CO2', 'HumidityRatio']
single_data =  [(23.1, 27.1, 419, 691, 0.00473937073052061, 1)]
single_data_Table = spark.createDataFrame(single_data, ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy'])
single_data_Table.show()
single_data_df = vectorAssembler.transform(single_data_Table)
single_data_df = single_data_df.select(['Features', 'Occupancy'])
single_data_df.show(5, truncate=False)
single_prediction = model.transform(single_data_df)
single_prediction.select("features", "Occupancy", "prediction").show()

spark.stop()
print("all ok.")
