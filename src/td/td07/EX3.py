import os
from pyspark.sql import SparkSession


def to_lower_split(input):
    return input.lower().split()


def driver():
    os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
    spark = SparkSession.builder.appName("Uber LS App Driver...").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # print(spark.sparkContext.getConf().getAll())

    depts = [("Finance", 10), ("Marketing", 20), ("Sales", 30), ("IT", 40)]
    dept_columns = ["dept_name", "dept_id"]

    depts_rdd = spark.sparkContext.parallelize(depts)
    depts_rdd.foreach(print)

    df = depts_rdd.toDF(dept_columns)
    df.printSchema()
    df.show(truncate=False)

    spark.stop()


driver()