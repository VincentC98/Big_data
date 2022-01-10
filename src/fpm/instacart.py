
import os
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F
import time

os.environ['HADOOP_HOME'] = "C:\\Users\\chart\\Desktop\\Automne 2021\\Analyse d'affaire\\TD\\spark-3.1.2-bin-hadoop3.2"
spark: SparkSession = SparkSession \
    .builder \
    .appName("FPM App Driver...") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
data_path = "../../resources/data/instacart"

# Import Data
# source: https://www.kaggle.com/c/instacart-market-basket-analysis/data

aisles = spark.read.csv(f"{data_path}\\aisles.csv", header=True, inferSchema=True)
departments = spark.read.csv(f"{data_path}\\departments.csv", header=True, inferSchema=True)
order_products_prior = spark.read.csv(f"{data_path}\\order_products__prior.csv", header=True, inferSchema=True)
order_products_train = spark.read.csv(f"{data_path}\\order_products__train.csv", header=True, inferSchema=True)
orders = spark.read.csv(f"{data_path}\\orders.csv", header=True, inferSchema=True)
products = spark.read.csv(f"{data_path}\\products.csv", header=True, inferSchema=True)

# orders.show(5)
# products.show(5)
# aisles.show()
# departments.show(5)
# order_products_train.show(5)
# order_products_prior.show(5)


# Create Temporary Tables
aisles.createOrReplaceTempView("aisles")
departments.createOrReplaceTempView("departments")
order_products_prior.createOrReplaceTempView("order_products_prior")
order_products_train.createOrReplaceTempView("order_products_train")
orders.createOrReplaceTempView("orders")
products.createOrReplaceTempView("products")

# Amine - Colval
# Organize the data by shopping basket
rawData = spark.sql("select p.product_name, o.order_id from products p "
                    "inner join order_products_train o "
                    "where o.product_id = p.product_id")
baskets = rawData.groupBy('order_id').agg(F.collect_set('product_name').alias('items'))
baskets.createOrReplaceTempView('baskets')
print("Raw data in basket...")
# rawData.show(5)
baskets.show(10, truncate=False)

# Extract out the items
baskets_df = spark.sql("select items from baskets").toDF("items")
print('data to analyse:')
baskets_df.show(5)
start_time = time.time()

# Use FPGrowth
fpGrowth = FPGrowth(itemsCol="items", minSupport=0.001, minConfidence=0.2)
fpm_model = fpGrowth.fit(baskets_df)
fpm_model.setPredictionCol("newPrediction")

print("fit cost: %s seconds." % (time.time() - start_time))

# Display frequent itemsets.
mostPopularItemInABasket = fpm_model.freqItemsets
mostPopularItemInABasket.show(10, truncate=False)

# Explore the frequent itemsets generated above
mostPopularItemInABasket.createOrReplaceTempView('mostPopularItemInABasket')
query = 'select items, freq ' \
        'from mostPopularItemInABasket ' \
        'where size(items) > 2 ' \
        'order by freq desc ' \
        'limit 5'
patterns_3items = spark.sql(query).toDF("items", "frequency")
print('The frequent >2 itemsets we generated above')
patterns_3items.show()


# Display generated association rules.
assoc_rules = fpm_model.associationRules
# assoc_rules.show(5, False)
assoc_rules.createOrReplaceTempView('assoc_rules')

query = 'select antecedent, consequent, confidence ' \
        'from assoc_rules ' \
        'order by confidence desc ' \
        'limit 5'
rules_top5 = spark.sql(query).toDF('antecedent', 'consequent', 'confidence')
print('Top 5 associations rules...')
rules_top5.show()


# test
test_transaction = [(0, ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'])]
test_DF = spark.createDataFrame(test_transaction, ["id", "items"])
fpm_model.transform(test_DF).show()

print('all ok')
spark.stop()