import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
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

#  Academic and Employability Factors influencing placement
# data source : https://www.kaggle.com/benroshan/factors-affecting-campus-placement?select=Placement_Data_Full_Class.csv

input_file = "../../resources/data/Placement_Data_Full_Class.csv"
model_file = "/resources/model/student_placement_model"

raw_data_df = spark.read.option("header", "true").csv(input_file)

raw_data_df.printSchema()
print(f'Number of rows: {raw_data_df.count()}') # 215
raw_data_df.show(5, truncate=False)
raw_data_df.createOrReplaceTempView("StudentPlacement")

fact_table_query = 'SELECT cast(ssc_p as float) SecondaryPercent, ' \
                   'cast(hsc_p as float) PostSecondaryPercent, ' \
                   '    case    when hsc_s = \'Commerce\' then 1 ' \
                   '            when hsc_s = \'Science\' then 2 ' \
                   '            else 3 end PostSecondarySpecialisation, ' \
                   'cast(degree_p as float) DegreePercent, ' \
                   '    case    when degree_t = \'Comm&Mgmt\' then 1 ' \
                   '            when degree_t = \'Sci&Tech\' then 2 ' \
                   '            else 3 end DegreeType, ' \
                   '    case    when workex = \'Yes\' then true ' \
                   '            else false end WorkExperience, ' \
                   '    case    when specialisation = \'Mkt&HR\' then 1 ' \
                   '            else 2 end MBASpecialisation, ' \
                   'cast(etest_p as float) EmploymentTestPercent, ' \
                   'cast(mba_p as float) MBAPercent, ' \
                   'status as placement ' \
                   'FROM StudentPlacement'

fact_table = spark.sql(fact_table_query)
fact_table.show()

# Combine multiple input columns to a Vector using Vector Assembler utility
features = ["SecondaryPercent", "PostSecondaryPercent", "PostSecondarySpecialisation",
            "DegreePercent", "DegreeType", "WorkExperience", "MBASpecialisation",
            "EmploymentTestPercent", "MBAPercent"]
vectorAssembler = VectorAssembler(inputCols=features, outputCol='features')
fact_df = vectorAssembler.transform(fact_table)
fact_df.printSchema()
print('Fact dataframe is:')
fact_df.show(5, truncate=False)

# Indexing is done to improve the execution times as comparing indexes
# is much cheaper than comparing strings/floats
# Index labels, adding metadata to the label column (Placement). Fit on
# whole dataset to include all labels in index.
labelIndexer = StringIndexer().setInputCol("placement").setOutputCol("indexedLabel").fit(fact_df)
# Index features vector
featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(fact_df)

# Split the data into training and test sets (20% held out for testing
training_df, test_df = fact_df.randomSplit([0.7, 0.3])

# Train a Decision Tree model
dt = DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

# Convert indexed labels back to original labels
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

# Chain indexers and decision tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt, labelConverter])
# Train model.  This also runs the indexers.
model = pipeline.fit(training_df)
# Make predictions.
predictions = model.transform(test_df)

# Select example rows to display.
print(' Some predictions..')
predictions.select("predictedLabel", "placement", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator()\
                .setLabelCol("indexedLabel")\
                .setPredictionCol("prediction")\
                .setMetricName("accuracy")

accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)

# Select example rows to display
# Example records with Predicted Occupancy as 0
print("Example records with Predicted placement as Not Placed (0):")
predictions.select("features", "placement", "prediction")\
    .where(F.col('prediction') == 0).show(10)

# Example records with Predicted Occupancy as 1
print("Example records with Predicted placement as Placed (1):")
predictions.select("features", "placement", "prediction")\
    .where(F.col('prediction') == 1).show(10)

# Example records with In-correct predictions
print("Example records with Predicted placement is 1 while recorded placement is Not Placed:")
predictions.select("features", "placement", "prediction")\
    .where((F.col('prediction') == 1)).where((F.col("placement") == 'Not Placed')).show(10)

# Example records with In-correct predictions
print("Example records with Predicted placement is 0 while recorded placement is Placed:")
predictions.select("features", "placement", "prediction")\
    .where((F.col('prediction') == 0)).where((F.col("placement") == 'Placed')).show(10)


spark.stop()
print("all ok.")
