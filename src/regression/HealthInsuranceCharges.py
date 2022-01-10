import os

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
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
# data source : https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv
# https://acadgild.com/blog/premium-insurance-policyholders-using-linear-regression-with-r

input_file = "../../resources/data/insurance.csv"
model_file = "../../resources/model/insurance_model"

raw_data_df = spark.read.option("header", "true").csv(input_file)

raw_data_df.printSchema()
print(f'Number of rows: {raw_data_df.count()}')  # 21613
raw_data_df.show(5, truncate=False)
raw_data_df.createOrReplaceTempView("HealthInsurance")


fact_table_query = 'SELECT cast(age as int) age, ' \
                   'cast( bmi as float) bmi, ' \
                   'cast(children as int) children, ' \
                   'cast(smoker as boolean) smoker, ' \
                   'cast(charges as float) charges ' \
                   'FROM HealthInsurance'

fact_table = spark.sql(fact_table_query)
fact_table.show()

# Combine multiple input columns to a Vector using Vector Assembler utility
features = ['age', 'bmi', 'children', 'smoker']
vectorAssembler = VectorAssembler(inputCols=features, outputCol='Features')
fact_df = vectorAssembler.transform(fact_table)
fact_df = fact_df.select(['Features', 'Charges'])
fact_df.show(5, truncate=False)

train_dataset, test_dataset = fact_df.randomSplit([0.9, 0.1])

lr = LinearRegression().setLabelCol("Charges").setFeaturesCol("Features")
lrModel = lr.fit(train_dataset)

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("Number of Iterations: %d" % trainingSummary.totalIterations)
print("objective History: %s" % str(trainingSummary.objectiveHistory))
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept (Ordonnée à origine): %s" % str(lrModel.intercept))
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)

# Make predictions.
predictions = lrModel.transform(test_dataset)
# show all predictions
predictions.show(5)

# ['age', 'bmi', 'children', 'smoker']
single_prediction = lrModel.predict(Vectors.dense(28, 29.4, 2, 0))
print(f'Prediction for values (28, 29.4, 2, 0) is: {single_prediction}')


spark.stop()
print("all ok.")
