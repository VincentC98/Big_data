import os
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import when
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

    arrayStructureData = [
        (("James", "", "Smith"), ["Java", "Scala", "C++"], "OH", "M"),
        (("Anna", "Rose", ""), ["Spark", "Java", "C++"], "NY", "F"),
        (("Julia", "", "Williams"), ["CSharp", "VB"], "OH", "F"),
        (("Maria", "Anne", "Jones"), ["CSharp", "VB"], "NY", "M"),
        (("Jen", "Mary", "Brown"), ["CSharp", "VB"], "NY", "M"),
        (("Mike", "Mary", "Williams"), ["Python", "VB"], "OH", "M")
    ]

    arrayStructureSchema = StructType([
        StructField('name', StructType([
            StructField('firstname', StringType(), True),
            StructField('middlename', StringType(), True),
            StructField('lastname', StringType(), True)
        ])),
        StructField('languages', ArrayType(StringType()), True),
        StructField('state', StringType(), True),
        StructField('gender', StringType(), True)
    ])

    df = spark.createDataFrame(data=arrayStructureData, schema=arrayStructureSchema)
    df.printSchema()
    df.show(truncate=False)

    df.filter(df.state == "OH").show(truncate=False)

    dataDF = spark.createDataFrame([(66, "a", "4"),
                                (67, "a", "0"),
                                (70, "b", "4"),
                                (71, "d", "4")],
                                ("id", "code", "amt"))

    dataDF.withColumn("new_column",
       when((F.col("code") == "a") | (F.col("code") == "d"), "A")
      .when((F.col("code") == "b") & (F.col("amt") == "4"), "B")
      .otherwise("A1")).show()

app_driver()
