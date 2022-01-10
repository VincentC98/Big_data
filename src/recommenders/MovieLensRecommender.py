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


# data source : https://grouplens.org/datasets/movielens/


movies_file = "../../resources/data/movielens/movies.csv"
users_file = "../../resources/data/movielens/users.csv"
ratings_file = "../../resources/data/movielens/ratings.csv"

movies_raw_data_df = spark.read.option("header", "true").option('delimiter', '::').csv(movies_file)
users_raw_data_df = spark.read.option("header", "true").option('delimiter', '::').csv(users_file)
ratings_raw_data_df = spark.read.option("header", "true").option('delimiter', '::').csv(ratings_file)

movies_raw_data_df.printSchema()
print(f'Number of movies: {movies_raw_data_df.count()}')  # 3883
movies_raw_data_df.show(5, truncate=False)
movies_raw_data_df.createOrReplaceTempView("Movies")

users_raw_data_df.printSchema()
print(f'Number of users: {users_raw_data_df.count()}')  # 6040
users_raw_data_df.show(5, truncate=False)
users_raw_data_df.createOrReplaceTempView("Users")

ratings_raw_data_df.printSchema()
print(f'Number of ratings: {ratings_raw_data_df.count()}')  # 1000209
ratings_raw_data_df.show(5, truncate=False)
ratings_raw_data_df.createOrReplaceTempView("Ratings")

fact_table_query = 'SELECT UserID, MovieID, Rating FROM Ratings'


fact_table = spark.sql(fact_table_query)
fact_table.show()

fact_df = fact_table.toDF('user', 'item', 'rating')
fact_df = fact_df.withColumn("user", fact_df["user"].cast(IntegerType()))
fact_df = fact_df.withColumn("item", fact_df["item"].cast(IntegerType()))
fact_df = fact_df.withColumn("rating", fact_df["rating"].cast(FloatType()))
fact_df.show(5, truncate=False)

seed = 1800009193
training_df, test_df = fact_df.randomSplit([0.9, 0.1], seed)
als = ALS().setMaxIter(5).setRegParam(0.01)\
        .setUserCol("user")\
        .setItemCol("item")\
        .setRatingCol("rating")

# Build model using Alternating Least Squares (ALS)
recommender_model = als.fit(training_df)
print(recommender_model)

# test the model
recommender_model.setColdStartStrategy("drop")
predict_df = recommender_model.transform(test_df)
# Remove NaN values from prediction (due to SPARK-14489)
predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))
predicted_ratings_df.show(5)

# Mean Squared Error(MSE) / (RMSE)
# Create an RMSE evaluator using the label and predicted columns
evaluator = RegressionEvaluator().setMetricName("rmse")\
                .setLabelCol("rating")\
                .setPredictionCol("prediction")
rmse = evaluator.evaluate(predict_df)
print(f'Root-mean-square error : {rmse}')


# Making Predictions for some users
# Get the top movies predictions for the most active user 4169
print('find recommendation for user 4169')
user_df = spark.table("Users").filter("UserId = 4169").select("UserID").withColumnRenamed("UserID", "user")
user_df.show()
topRecommendationsForUser_df = recommender_model.recommendForUserSubset(user_df, 20)
topRecommendationsForUser_df.show(truncate=False)
values = []
for row in topRecommendationsForUser_df.rdd.collect():
    recommendations = row['recommendations']
    for (item, rating) in recommendations:
        movie_title = spark.sql('SELECT Title FROM Movies WHERE MovieId=' + str(item)).first()
        values.append((movie_title, round(rating, 2)))

columns = ['movie_title', 'rating']
movie_rating_df = spark.createDataFrame(values, columns)
movie_rating_df.printSchema()
movie_rating_df.show(truncate=False)

spark.stop()
print("all ok.")
