import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


def to_lower_split(input):
    return input.lower().split()


def driver():
    os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
    # .master("spark://192.168.2.10:7077")
    spark = SparkSession.builder.appName("Uber LS App Driver...").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # print(spark.sparkContext.getConf().getAll())

    data_science = ["scala", "python", "hadoop", "spark in action", "akka", "spark vs hadoop",
                    "pyspark", "pyspark and spark"]

    words_rdd = spark.sparkContext.parallelize(data_science)
    words_rdd.foreach(print)

    filtered_words = words_rdd.filter(lambda x: 'spark' in x)
    print(f'filtered RDD {filtered_words.collect()}')
    print(filtered_words.first())

    print(f'Number of entries containing spark word is {filtered_words.count()}')

    filtered_words.cache()
    print(filtered_words.persist().is_cached)
    spark.catalog.clearCache()

    spark.stop()


driver()
