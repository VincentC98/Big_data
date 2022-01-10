import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def merge(a, b):
    return a + b


def app_driver():
    os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
    spark = SparkSession.builder.appName("Uber LS App Driver...").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # print(spark.sparkContext.getConf().getAll())

    input_data = 'D:/GDrive/F20/5D3.DS/TD/TD08.PySpark.SQL/Students.json'

    students_df = spark.read.json(input_data)
    students_sorted_columns = students_df.select(['id', 'fname', 'lname', 'email'])
    students_sorted_columns.show(5, truncate=False)

    students_names = students_df.select(['fname', 'lname'])
    students_names.limit(5).show(truncate=False)

    print(students_df.groupBy('gender').count().show())

    students_males_v1 = students_df.filter(F.col('gender') == 'Male')
    students_males_v1.limit(5).show(truncate=False)
    students_df.createOrReplaceTempView("Students")
    gender_filter = 'SELECT fname, lname, gender FROM Students WHERE gender == "Male" '
    students_males_v2 = spark.sql(gender_filter)
    students_males_v2.limit(5).show(truncate=False)

    students_fullnames = students_df.withColumn('fullname',
        F.concat(F.col('fname'), F.lit(' '), F.col('lname')))
    students_fullnames.limit(5).show(truncate=False)

    spark.stop()
    print('driver stop...')


app_driver()