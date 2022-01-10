import os
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.environ['HADOOP_HOME'] = "C:\\Users\\chart\\Desktop\\Automne 2021\\Analyse d'affaire\\TD\\spark-3.1.2-bin-hadoop3.2\\bin"
spark = SparkSession\
                    .builder\
                    .appName("App Driver...")\
                    .master("local[*]")\
                    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(spark.sparkContext.getConf().getAll())

#  House Sales King County
# data file : https://www.kaggle.com/ambarish/tutorial-housesales-kingcounty-eda-modelling/data

input_file = "../../resources/data/kc_house_data.csv"
model_file = "/resources/model/kc_house_model"

raw_data_df = spark.read.option("header", "true").csv(input_file)

raw_data_df.printSchema()
print(f'Number of rows: f{raw_data_df.count()}')  # 21613
raw_data_df.show(5, truncate=False)
raw_data_df.createOrReplaceTempView("KingCountyHouses")

query = 'SELECT zipcode, avg(price) avgPrice ' \
        'FROM KingCountyHouses ' \
        'GROUP BY zipcode ' \
        'ORDER BY avgPrice DESC'
result = spark.sql(query)
result.show()

fact_table_query = 'SELECT bedrooms, bathrooms, sqft_living, sqft_lot, price ' \
        'FROM KingCountyHouses'
fact_table = spark.sql(fact_table_query)
fact_table = fact_table.select(*(F.col(c).cast('float').alias(c) for c in fact_table.columns))
fact_table.show()

# label = price, features = [bedrooms, bathrooms, sqft_living, sqft_lot]
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
vectorAssembler = VectorAssembler(inputCols=features, outputCol='features')
fact_df = vectorAssembler.transform(fact_table)
fact_df = fact_df.select(['features', 'price'])
fact_df.show(5)

# check correlation between features
# import six
# for i in fact_df.columns:
#     if not(isinstance(fact_df.select(i).take(1)[0][0], six.string_types)):
#         print("Correlation to price for ", i, fact_df.stat.corr('price', i))

train_data, test_data = fact_df.randomSplit([0.75, 0.25])

lr = LinearRegression(featuresCol='features', labelCol='price',
                      maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(train_data)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# # Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))

trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
r2 = trainingSummary.r2
n = train_data.count()
p = len(train_data.columns)
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print("Adjusted r2: %f" % adjusted_r2)

# testing dataset
predictions = lrModel.transform(test_data)
x = ((predictions['price'] - predictions['prediction'])/predictions['price'])*100
predictions = predictions.withColumn('Accuracy', x)
predictions.select("prediction", "price", "Accuracy", "features").show()

# r — square value for the test dataset
predictions_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="r2")
print("R Squared (R2) on test data = %g" % predictions_evaluator.evaluate(predictions))

single_prediction = lrModel.predict(Vectors.dense(1.0, 2, 2000, 15000))
print(f'Prediction for values (1.0, 2, 2000, 15000) is: {single_prediction}')

spark.stop()
print("all ok.")
