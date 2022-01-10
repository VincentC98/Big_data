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


def driver():
    os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
    spark = SparkSession.builder.appName("Uber LS App Driver...").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # print(spark.sparkContext.getConf().getAll())

    tweets_file = '/ressources/data/reduced-tweets.json'

    # read as RDD of rows
    tweets = spark.sparkContext.textFile(tweets_file)
    # print(tweet_decoder(tweets.first()))

    # convert df of rows into rdd of tweet object using decoder
    tweets_rdd = tweets.map(tweet_decoder)
    # print(tweets_rdd.first())

    # find tweeters in Indonesia
    filtered_tweets = tweets_rdd.filter(lambda tweet: tweet.country == 'Indonesia')
    print(f'There are {filtered_tweets.count()} people tweeting form Indonisia')

    # find recipients people in tweets
    cited_people = filtered_tweets.flatMap(lambda tweet: tweet.text.split(" "))\
        .filter(lambda w: w.startswith('@') and len(w) > 1)
    print(f'There are {cited_people.count()} people are cited in tweets')
    # for element in cited_people.collect():
    # print(element)

    # Count how many times each person is cited
    people_counts = cited_people.map(lambda person: (person, 1)).reduceByKey(lambda a, b: a + b)
    # for element in people_counts.collect():
    # print(element)

    # Find the 10 most cited people by descending order
    top_ten_cited = people_counts.map(lambda pair: (pair[1], pair[0])).sortByKey(False).take(10)
    for element in top_ten_cited:
        print(element)

    spark.stop()


driver()
