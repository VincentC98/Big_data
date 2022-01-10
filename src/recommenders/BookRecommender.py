import os

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType

os.environ['HADOOP_HOME'] = "C:\\Users\\chart\\Desktop\\Automne 2021\\Analyse d'affaire\\TD\\spark-3.1.2-bin-hadoop3.2"
spark = SparkSession\
                    .builder\
                    .appName("Uber LS App Driver...")\
                    .master("local[4]")\
                    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(spark.sparkContext.getConf().getAll())


# data source : https://www.kaggle.com/zygmunt/goodbooks-10k?select=sample_book.xml
# 10000 books x 912705 readings x 981756 ratings

books_file = "../../resources/data/bookstore/books.csv"
user_book_file = "../../resources/data/bookstore/to_read.csv"
ratings_file = "../../resources/data/bookstore/ratings.csv"

books_raw_data_df = spark.read.option("header", "true").option('delimiter', ',').csv(books_file)
user_book_raw_data_df = spark.read.option("header", "true").option('delimiter', ',').csv(user_book_file)
ratings_raw_data_df = spark.read.option("header", "true").option('delimiter', ',').csv(ratings_file)

books_raw_data_df.printSchema()
print(f'\nNumber of books: {books_raw_data_df.count()}')
books_raw_data_df.show(5, truncate=False)
books_raw_data_df.createOrReplaceTempView("Books")
print(books_raw_data_df.first())


user_book_raw_data_df.printSchema()
print(f'\nNumber of readings: {user_book_raw_data_df.count()}')  # 6040
user_book_raw_data_df.show(5, truncate=False)
user_book_raw_data_df.createOrReplaceTempView("To_Read")

ratings_raw_data_df.printSchema()
print(f'\nNumber of ratings: {ratings_raw_data_df.count()}')  # 1000209
ratings_raw_data_df.show(5, truncate=False)
ratings_raw_data_df.createOrReplaceTempView("Ratings")

fact_table_query = 'SELECT user_id, book_id, rating FROM Ratings'
fact_table = spark.sql(fact_table_query)
fact_df = fact_table.toDF('reader', 'book', 'rating')
fact_df = fact_df.withColumn("reader", fact_df["reader"].cast(IntegerType()))
fact_df = fact_df.withColumn("book", fact_df["book"].cast(IntegerType()))
fact_df = fact_df.withColumn("rating", fact_df["rating"].cast(FloatType()))
fact_df.show(5, truncate=False)

training_df, test_df = fact_df.randomSplit([0.9, 0.1])
als = ALS(userCol="reader", itemCol="book", ratingCol="rating")

# Build model using Alternating Least Squares (ALS)
recommender_model = als.fit(training_df)
print(recommender_model)


# test the model
recommender_model.setColdStartStrategy("drop")
predict_df = recommender_model.transform(test_df)
# Remove NaN values from prediction (due to SPARK-14489)
predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))
predicted_ratings_df.show(5)
predicted_ratings_df.createOrReplaceTempView("predictions")

# predictions are (reader, book, rating, prediction)
# display predictions (reader_id, book_id, rating, prediction, title)
query = 'select predictions.reader, predictions.book, predictions.rating, ' \
        'predictions.prediction, books.title ' \
        'FROM predictions ' \
        'INNER JOIN books ON predictions.book = books.id'

result = spark.sql(query)
result.show(10, truncate=False)

# Mean Squared Error(MSE) / (RMSE)
# Create an RMSE evaluator using the label and predicted columns
evaluator = RegressionEvaluator().setMetricName("rmse")\
                .setLabelCol("rating")\
                .setPredictionCol("prediction")
rmse = evaluator.evaluate(predict_df)
# On average the mean error is 0.89 that is the difference between
# the original rating and the predicted rating.
print(f'Root-mean-square error : {rmse}')

# get recommendations for 10 users
print('Recommendation for 10 users...')
ALS_recommendations = recommender_model.recommendForAllUsers(numItems=10)
ALS_recommendations.show(n=10, truncate=False)

spark.stop()
print("all ok.")
