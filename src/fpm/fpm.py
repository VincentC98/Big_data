
import os
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import split

os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
spark: SparkSession = SparkSession\
                        .builder\
                        .appName("FPM App Driver...")\
                        .master("local[4]")\
                        .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
# print(ss.sparkContext.getConf().getAll())

# data = [(1, ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt']),
#                 (2, ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt']),
#                 (3, ['Milk', 'Apple', 'Kidney Beans', 'Eggs']),
#                 (4, ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt']),
#                 (5, ['Corn', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs'])]
# transactions = data.map(lambda line: line.strip().split(' '))
# df = ss.createDataFrame(transactions, ["id", "items"])

input_file = "../../resources/data/market_basket.csv"

data = (spark.read.text(input_file).select(split("value", "\s+").alias("items")))
data.show(truncate=False)
fpGrowth = FPGrowth(itemsCol="items", minSupport=0.2, minConfidence=0.7)
fpm_model = fpGrowth.fit(data)
fpm_model.setPredictionCol("newPrediction")

# Display frequent itemsets.
fpm_model.freqItemsets.show(5)

# Display generated association rules.
fpm_model.associationRules.show(5)

# transform examines the input items against all the association rules
# and summarize the consequents as prediction
fpm_model.transform(data).show(5)

spark.stop()


