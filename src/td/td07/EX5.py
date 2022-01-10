import os
from pyspark.sql import SparkSession
import json


class Tweet(object):

    def __init__(self, id, user, text, place, country, *args, **kwargs):
        self.user_id = id
        self.user_name = user
        self.text = text
        self.place = place
        self.country = country

    def __str__(self):
        return f'<Id={self.user_id}, Username={self.user_name}>'

    __repr__ = __str__


def tweet_decoder(json_input):
    loaded_json = json.loads(json_input)
    return Tweet(**loaded_json)


def app_driver():
    os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
    spark = SparkSession.builder.appName("Uber LS App Driver...").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # print(spark.sparkContext.getConf().getAll())

    tweets_file = '/ressources/data/reduced-tweets.json'

    # read data
    tweets = spark.sparkContext.textFile(tweets_file)
    # print(tweet_decoder(tweets.first()))

    # convert df of rows into rdd of tweet object using decoder
    tweets_rdd = tweets.map(tweet_decoder)
    # print(tweets_rdd.first())

    # count tweets by user_id : (user_id, number of tweets)
    tweets_by_user_rdd = tweets_rdd.map(lambda tweet: (tweet.user_id, 1)).reduceByKey(lambda a, b: a + b)
    for element in tweets_by_user_rdd:
        print(element)

    # filter tweets by user
    filtered = tweets_rdd.flatMap(lambda tweet: (tweet.user_id == tweets_by_user_rdd))
    print(filtered)

    spark.stop()


app_driver()
