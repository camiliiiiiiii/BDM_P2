import findspark
findspark.init()

from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("myApp") \
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1') \
    .getOrCreate()

# Clear the Spark catalog caché
spark.catalog.clearCache()

# Unpersist all persistent RDDs to ensure that the memory is not occupied by previously cached RDDs, avoiding any potential memory issues or unnecessary memory usage.
for (id, rdd) in spark._jsc.getPersistentRDDs().items():
    rdd.unpersist()
    print("Unpersisted {} rdd".format(id))

# Read RDDs from MongoDB
income_lookup_ne_RDD = spark.read.format("mongo") \
    .option('uri', 'mongodb://127.0.0.1/BDM_P2.income_lookup_neighborhood') \
    .load().rdd

rent_lookup_ne_RDD = spark.read.format("mongo") \
    .option('uri', 'mongodb://127.0.0.1/BDM_P2.rent_lookup_neighborhood') \
    .load().rdd

income_ne_RDD = spark.read.format("mongo") \
    .option('uri', 'mongodb://127.0.0.1/BDM_P2.income') \
    .load().rdd

rent_extra_ne_RDD = spark.read.format("mongo") \
    .option('uri', 'mongodb://127.0.0.1/BDM_P2.lloguer') \
    .load().rdd




