import os
from pyspark.sql import SparkSession


def app_driver():
    os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
    spark = SparkSession.builder.appName("Uber LS App Driver...").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # print(spark.sparkContext.getConf().getAll())

    input_data = 'D:/GDrive/F20/5D3.DS/TD/TD08.PySpark.SQL/Employees.json'

    employees_df = spark.read.json(input_data)
    employees_df.printSchema()
    employees_df.show(truncate=False)

    employees_df.createOrReplaceTempView("Employee")
    salary_filter = 'SELECT name FROM Employee WHERE salary>3500'
    result_df = spark.sql(salary_filter)
    result_df.printSchema()
    result_df.show()

    age_filter = 'SELECT name FROM Employee WHERE age = 30'
    result_df = spark.sql(age_filter)
    result_df.printSchema()
    result_df.show()

    spark.stop()
    print('driver stop...')


app_driver()
