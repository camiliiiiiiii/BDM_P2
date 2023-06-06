import findspark
findspark.init()

import pyspark
import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, lit
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *
from pyspark.sql.types import StructField

import pandas as pd
from pymongo import MongoClient

import findspark
from pyspark import Row
from pyspark.sql.functions import lit
from functools import reduce
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer

from pyspark.ml.feature import VectorAssembler
import pandas as pd
from pymongo import MongoClient
findspark.init()
import os
import re
import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum

# Create a SparkSession
spark = SparkSession.builder \
    .appName("myApp") \
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1') \
    .getOrCreate()


final_table = spark.read.format("mongo") \
    .option('uri', 'mongodb://10.4.41.45/Formatted_BDM_P2.Final_results') \
    .load() \
    .rdd \
    .cache()

final_table_df = final_table.toDF()
column_name = "_id"
final_table_df = final_table_df.drop(column_name)


final_table_df.describe().show()


na_counts = final_table_df.agg(*[
    sum(col(column).isNull().cast("int")).alias(column + "_na_count")
    for column in final_table_df.columns
])

final_table_df = final_table_df.dropna(subset=["hasLift"])
final_table_df.show()


final_table_df = final_table_df.withColumn("avg_income", final_table_df["avg_income"].cast(IntegerType()))
final_table_df = final_table_df.withColumn("avg_pop", final_table_df["avg_pop"].cast(IntegerType()))
final_table_df = final_table_df.withColumn("avg_rent", final_table_df["avg_rent"].cast(IntegerType()))
final_table_df = final_table_df.withColumn("bathrooms", final_table_df["bathrooms"].cast(IntegerType()))
final_table_df = final_table_df.withColumn("price", final_table_df["price"].cast(IntegerType()))
final_table_df = final_table_df.withColumn("priceByArea", final_table_df["priceByArea"].cast(IntegerType()))
final_table_df = final_table_df.withColumn("size", final_table_df["size"].cast(IntegerType()))
final_table_df = final_table_df.withColumn("rooms", final_table_df["rooms"].cast(IntegerType()))

final_table_df.printSchema()



# Transform the names of the neighbourhoods into an index
final_table_df_index = StringIndexer(inputCol='neigh_id', outputCol='neigh_idx').fit(final_table_df).transform(final_table_df)


# Create an instance of the one hot encoder
onehot = OneHotEncoder(inputCols=['neigh_idx'], outputCols=['neigh_dummy'])

# Apply the one hot encoder to the neighbourhood data
onehot = onehot.fit(final_table_df_index)
final_table_result = onehot.transform(final_table_df_index)

neigh_dict = final_table_result.select('neigh_id', 'neigh_dummy')
neigh_dict = neigh_dict.dropDuplicates()


vectorAssembler = VectorAssembler(inputCols = ['neigh_dummy', 'price'], outputCol = 'features')
model_df = vectorAssembler.transform(final_table_result)
model_df = model_df.select(['features', 'size'])
model_df.show(3)

###############STREAM##################
from pyspark.sql import SparkSession
from kafka import KafkaConsumer
from json import loads

consumer = KafkaConsumer('bdm_p2', bootstrap_servers=['pidgeot.fib.upc.edu:9092'])

spark = SparkSession \
        .builder \
        .master(f"local[*]") \
        .appName("myApp") \
        .config('spark.jars.packages','org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0')\
        .getOrCreate()

df = spark \
    .readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', 'pidgeot.fib.upc.edu:9092') \
    .option('subscribe', 'bdm_p2') \
    .load()

parsed_message1 = df.withColumn('neigh_id1', split(df['value'], ',').getItem(1)) \
    .withColumn('price', split(df['value'], ',').getItem(2)) \
    .select('neigh_id1', 'price')

parsed_message2 = parsed_message1.join(neigh_dict, parsed_message1.neigh_id1 == neigh_dict.neigh_id, "left")

parsed_message = parsed_message2.withColumn("price", parsed_message2["price"].cast(IntegerType()))

RandomForestClassifier = vectorAssembler(inputCols=['neigh_dummy', 'price'], outputCol='features')
model_df = vectorAssembler.transform(parsed_message)
model_df1 = model_df.select('features')

predictions = classifier.transform(model_df1)

predictions = predictions.join(model_df, predictions.features == model_df.features)

query = predictions.select('neigh_id', 'price', 'prediction') \
    .writeStream \
    .format("console") \
    .trigger(once=True) \
    .start()

query.awaitTermination()


