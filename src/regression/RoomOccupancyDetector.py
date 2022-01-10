import os

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.environ['HADOOP_HOME'] = "C:\\Users\\chart\\Desktop\\Automne 2021\\Analyse d'affaire\\TD\\spark-3.1.2-bin-hadoop3.2\\bin"
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
print(f'Number of rows: f{raw_data_df.count()}')  # 21613
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

train_dataset, test_dataset = fact_df.randomSplit([0.7, 0.3])

lr = LogisticRegression(featuresCol='Features', labelCol='Occupancy', maxIter=10)
lrModel = lr.fit(train_dataset)
trainingSummary = lrModel.summary
print(trainingSummary)

# Make predictions.
predictions = lrModel.transform(test_dataset)

print("Example records with Predicted Occupancy as 0:")
predictions.select("features", "Occupancy", "prediction")\
    .where(F.col('prediction') == 0).show(10)

print("Example records with Predicted Occupancy as 1:")
predictions.select("features", "Occupancy", "prediction")\
    .where(F.col('prediction') == 1).show(10)

# show all predictions
# predictions.show(5)

spark.stop()
print("all ok.")
