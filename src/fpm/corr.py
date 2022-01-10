import os

from pyspark.ml.linalg import SparseVector, DenseVector, Vectors
from pyspark.mllib.stat import Statistics
from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation, ChiSquareTest

os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
spark: SparkSession = SparkSession \
    .builder \
    .appName("FPM App Driver...") \
    .master("local[4]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.sparkContext.parallelize([
    (1, SparseVector(4, {0: 1.0, 3: -2.0})),
    (2, DenseVector([4.0, 5.0, 0.0, 3.0])),
    (3, DenseVector([6.0, 7.0, 0.0, 8.0])),
    (4, SparseVector(4, {0: 9.0, 3: 1.0})),
]).toDF(["row_num", "features"])

df.show()
# data.printSchema()
# data.show(truncate=False)

# find the frequent items that show up 40% of the time for each column
# one-pass algorithm proposed by Karp et al.
# freq = df.stat.freqItems(["a", "b", "c"], 0.4)
# e = freq.collect()[0]
# print(e)
# Correlation.corr(df, "features")

pearson_matrix = Correlation.corr(df, "features", "pearson").head()
print(f"Pearson correlation matrix:\n {str(pearson_matrix[0])}")
spearman_matrix = Correlation.corr(df, "features", "spearman").head()
print(f"Spearman correlation matrix:\n {str(spearman_matrix[0])}\n")

df1 = spark.sparkContext.parallelize([(0.0, 1.0), (1.0, 0.0)]).toDF(["x", "y"])
print(df1.stat.corr("x", "y"))

english_grades = spark.sparkContext.parallelize([56, 78, 45, 71, 62, 64, 58, 80, 76, 61], 2)
math_grades = spark.sparkContext.parallelize([66, 70, 40, 60, 65, 56, 59, 77, 67, 63], 2)

corrType = 'pearson'
corr = Statistics.corr(english_grades, math_grades, corrType)
print(f'Pearson correlation coefficient : {corr}')
corrType = 'spearman'
corr = Statistics.corr(english_grades, math_grades, corrType)
print(f'Spearman correlation coefficient : {corr}')

data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]
data_df = spark.createDataFrame(data, ["label", "features"])

r = ChiSquareTest.test(data_df, "features", "label").head()
# r.show(truncate=False)
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))