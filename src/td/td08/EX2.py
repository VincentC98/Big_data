import os
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from pyspark.sql import functions as F


def is_odd(number: int):
    if number % 2 == 0:
        return True
    return False


def app_driver():
    os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
    spark = SparkSession.builder.appName("Uber LS App Driver...").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # print(spark.sparkContext.getConf().getAll())

    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    values_df = spark.createDataFrame(data=values, schema=IntegerType()).withColumnRenamed('value', 'digit')
    # values_df = spark.createDataFrame(values, schema=StructType([StructField("digit", IntegerType())]))
    # values_df.withColumnRenamed("", "digit")
    values_df.printSchema()
    values_df.show(truncate=False)
    print(values_df.dtypes)

    odd_values = values_df.filter((values_df.digit % 2) == 0).sort('digit', ascending=False)
    odd_values.show(truncate=False)

    digits_data = [(1, 'One', 'I'), (2, 'Two', 'II'), (3, 'Three', 'III'), (4, 'Four', 'IV')]
    digits_schema = StructType([StructField("Arabic", IntegerType()),
                                StructField("English", StringType()),
                                StructField("Roman", StringType())])
    digits_df = spark.createDataFrame(data=digits_data, schema=digits_schema)
    digits_df.show(truncate=False)

    df34 = digits_df\
        .where(F.col('Arabic') > 2)\
        .select(digits_df.columns[0:2])\
        .orderBy('English', ascending=True)
    df34.show(truncate=False)

    spark.stop()
    print('driver stop...')


app_driver()
