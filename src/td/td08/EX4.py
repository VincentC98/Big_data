import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def app_driver():
    os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
    spark = SparkSession.builder.appName("Uber LS App Driver...").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # print(spark.sparkContext.getConf().getAll())

    input_data = 'D:/GDrive/F20/5D3.DS/TD/TD08.PySpark.SQL/Vehicles.csv'
    vehicles_df = spark.read.csv(input_data, header=True)
    vehicles_df.limit(5).show(truncate=False)

    cars_2010 = vehicles_df.select("year", "comb08", "VClass")\
    .filter(F.col('VClass').contains('Cars'))\
    .filter(F.col('year') > 2010)\
    .withColumnRenamed('comb08','mpg')\
    .withColumnRenamed('VClass','class')
    cars_2010.limit(5).show(truncate=False)

    cars_2010_improved = cars_2010.groupBy("class", "year").agg(F.round(F.avg("mpg"), 2))\
    .withColumnRenamed("round(avg(mpg), 2)", "mpg")\
    .sort("class", "year")
    cars_2010_improved.limit(50).show(truncate=False)

    cars_rotated = cars_2010.groupBy("class").pivot("year").agg(F.round(F.avg("mpg"), 2)).sort("class");
    cars_rotated.limit(10).show(truncate=False)


app_driver()