import os

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession

os.environ['HADOOP_HOME'] = "C:\\Users\\chart\\Desktop\\Automne 2021\\Analyse d'affaire\\TD\\spark-3.1.2-bin-hadoop3.2"
spark = SparkSession \
    .builder \
    .appName("App Driver...") \
    .master("local[4]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(spark.sparkContext.getConf().getAll())

#  House Sales King County
# data source : https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv
# https://acadgild.com/blog/premium-insurance-policyholders-using-linear-regression-with-r

input_file = "../../resources/data/energy_data_complete.csv"
model_file = "../../resources/model/appliance_energy_model"

raw_data_df = spark.read.option("header", "true").csv(input_file)

raw_data_df.printSchema()
print(f'Number of rows: {raw_data_df.count()}')
raw_data_df.show(5, truncate=False)
raw_data_df.createOrReplaceTempView("ApplianceEnergy")

fact_table_query = 'SELECT ' \
                   'cast(Appliances as float) Appliance_Energy, ' \
                   'cast(T1 as float) T1, ' \
                   'cast(RH_1 as float) RH_1, ' \
                   'cast(T2 as float) T2, ' \
                   'cast(RH_2 as float) RH_2, ' \
                   'cast(T3 as float) T3, ' \
                   'cast(RH_3 as float) RH_3, ' \
                   'cast(T4 as float) T4, ' \
                   'cast(RH_4 as float) RH_4, ' \
                   'cast(T5 as float) T5, ' \
                   'cast(RH_5 as float) RH_5, ' \
                   'cast(T6 as float) T6, ' \
                   'cast(RH_6 as float) RH_6, ' \
                   'cast(T7 as float) T7, ' \
                   'cast(RH_7 as float) RH_7, ' \
                   'cast(T8 as float) T8, ' \
                   'cast(RH_8 as float) RH_8, ' \
                   'cast(T9 as float) T9, ' \
                   'cast(RH_9 as float) RH_9, ' \
                   'cast(T_out as float) T_OUT, ' \
                   'cast(Press_mm_hg as float) PRESS_OUT, ' \
                   'cast(RH_out as float) RH_OUT, ' \
                   'cast(Windspeed as float) WIND,  ' \
                   'cast(Visibility as float) VIS ' \
                   'FROM ApplianceEnergy'

fact_table = spark.sql(fact_table_query)
fact_table.show()

features = ["T1", "RH_1", "T2", "RH_2", "T3", "RH_3",
            "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8",
            "RH_8", "T9", "RH_9", "T_OUT", "PRESS_OUT", "RH_OUT", "WIND", "VIS"]
# Combine multiple input columns to a Vector using Vector Assembler utility
vectorAssembler = VectorAssembler(inputCols=features, outputCol='Features')
fact_df = vectorAssembler.transform(fact_table)
fact_df = fact_df.select(['Features', 'Appliance_Energy'])
fact_df.show(5, truncate=False)

train_dataset, test_dataset = fact_df.randomSplit([0.9, 0.1])

regression_stage = LinearRegression().setLabelCol("Appliance_Energy").setFeaturesCol("Features")
pipeline = Pipeline()
pipeline.setStages([regression_stage])
lrModel = pipeline.fit(train_dataset)

# Make predictions.
predictions = lrModel.transform(test_dataset)
# show all predictions
predictions.show(5)

spark.stop()
print("all ok.")

