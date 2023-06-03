import findspark
from pyspark import Row
from pyspark.sql.functions import lit
from functools import reduce
from pyspark.sql.types import FloatType

findspark.init()
import os
import re
import datetime
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

# Look up tables
income_lookup_ne_RDD = spark.read.format("mongo") \
    .option('uri', 'mongodb://10.4.41.45/BDM_P2.income_lookup_neighborhood') \
    .load().rdd

rent_lookup_ne_RDD = spark.read.format("mongo") \
    .option('uri', 'mongodb://10.4.41.45/BDM_P2.rent_lookup_neighborhood') \
    .load().rdd

# Income
income_ne_RDD = spark.read.format("mongo") \
    .option('uri', 'mongodb://10.4.41.45/BDM_P2.income') \
    .load()


# Rent extra dataset
rent_extra_ne_RDD = spark.read.format("mongo") \
    .option('uri', 'mongodb://10.4.41.45/BDM_P2.lloguer') \
     .load().rdd



# Idealista
def get_date(x):
    date = datetime.datetime.strptime(x.replace("_idealista", ""), '%Y_%m_%d').date()
    return date

path = os.getcwd()
path_idealista = os.path.join(path, "P2_data/idealista")
file_list = [f for f in os.listdir(path_idealista) if not f.startswith('.')]  # Exclude hidden files

spark.sparkContext.setLogLevel("WARN")  # Set log level to WARN

df = None
for i in file_list:
    try:
        property = spark.read.parquet(os.path.join(path_idealista, i))
        date = get_date(i)
        property = property.withColumn("date", lit(date))
        if i == file_list[0]:
            df = property
        else:
            df = df.union(property)
    except:
        pass

if df:
    # Rearrange columns with "date" column at the beginning
    columns = ["date"] + df.columns[:-1]
    df = df.select(columns)

    idealista_ne_RDD = df.rdd
else:
    print("No valid files found")



#####

##RENT-OPENDATA
def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

rent_extra_ne_num_rdd = rent_extra_ne_RDD.filter(lambda x: is_numeric(x[7])).map(lambda x: (float(x[7]), x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[8]))

records = rent_extra_ne_num_rdd.take(5)
for record in records:
    print(record)

preuRDD = rent_extra_ne_num_rdd \
    .filter(lambda x: x[7] != '--' and x[7] != None) \
    .filter(lambda x: 2022 == x[1]) \
    .map(lambda x: (x[5], float(x[7]))) \
    .groupByKey() \
    .mapValues(lambda x: round(sum(x) / len(x), 2)) \
    .map(lambda x: (x[0], x[1]))\
    .cache()


print("Count after filter: ", preuRDD.count())  # Add this line

sample_records =preuRDD.take(5)
print("Sample records:")
for record in sample_records:
    print(record)

