import time
import os
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list


start_time = time.time()

os.environ['HADOOP_HOME'] = "D:\\Apps\\spark-3.0.0-preview2-bin-hadoop2.7"
spark: SparkSession = SparkSession\
                        .builder\
                        .appName("FPM App Driver...")\
                        .master("local[4]")\
                        .getOrCreate()

# dataset : https://www.kaggle.com/somesh/order-products-train
data_path = "../../resources/data/instacart"

order_products_train = spark.read\
    .option('header', 'true')\
    .csv(f"{data_path}\\order_products__train.csv")
print(f"order_products_train count = {order_products_train.count()}")

order_products_prior = spark.read\
    .option('header', 'true')\
    .csv(f"{data_path}\\order_products__prior.csv")
print(f"order_products_prior count = {order_products_prior.count()}")

# combine two dataframe as both have same format
# (order_id, product_id, add_to_cart_order, reordered)
baskets = order_products_prior.union(order_products_train)
# dispose dataframes since they are not useful.
order_products_prior.unpersist()
order_products_train.unpersist()
print(f"order_products count = {baskets.count()}")

products = spark.read\
    .option('header', 'true')\
    .csv(f"{data_path}\\products.csv")

baskets = baskets.join(products, baskets.product_id == products.product_id)
products.unpersist()

baskets = baskets.groupby('order_id')\
    .agg(collect_list('product_name').alias('products'))

print(f"grouped count = {baskets.count()}")
baskets.show(5, truncate=False)
#   -------------------------------------------
#   using FP-Growth to generate frequent patterns.
#   -------------------------------------------
fp_growth = FPGrowth()\
    .setItemsCol('products')\
    .setMinSupport(0.01)\
    .setMinConfidence(0.125)

fpm_model = fp_growth.fit(baskets)
fpm_model.setPredictionCol('new_prediction')

frequent_itemsets = fpm_model.freqItemsets
frequent_itemsets.createOrReplaceTempView('frequent')
query = 'select * ' \
        'from frequent ' \
        'order by freq desc ' \
        'limit 10'
print(f"Relevant top-10 itemsets...\n")
frequent_itemsets = spark.sql(query)
frequent_itemsets.show(truncate=False)

# compute association rules
association_rules = fpm_model.associationRules
association_rules.show(5, truncate=False)
association_rules.createOrReplaceTempView('association_rules')

query = 'select * ' \
        'from association_rules ' \
        'order by confidence desc ' \
        'limit 10'
top_rules = spark.sql(query)
print(f"Relevant top-10 rules...\n")
top_rules.show(truncate=False)

# evaluate model
test_transaction = [(0, ['Large Lemon', 'Strawberries'])]
test_df = spark.createDataFrame(test_transaction, ["order_id", "products"])
fpm_model.transform(test_df).show(truncate=False)

print(f"Running time is: {time.time()-start_time} seconds.")
spark.stop()
