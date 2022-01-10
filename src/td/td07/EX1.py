import os
from pyspark.sql import SparkSession


def driver():
    os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
    spark = SparkSession.builder.appName("Uber LS App Driver...").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # print(spark.sparkContext.getConf().getAll())

    tweets_file = 'D:/GDrive/F20/5D3.DS/TD/TD07.Spark.RDD/input.txt'

    # set threshold
    threshold = 2

    # read in text file and split each line into words
    input_data = spark.sparkContext.textFile(tweets_file)
    tokenized = input_data.flatMap(lambda line: line.split(" "))

    # count the occurrence of each word
    wordCounts = tokenized.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

    # filter out words with fewer than threshold occurrences
    filtered = wordCounts.filter(lambda pair: pair[1] >= threshold)

    for element in filtered.collect():
        print(element)

    spark.stop()
    print('driver stop...')


driver()