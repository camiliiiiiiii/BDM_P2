import findspark
from pyspark import Row, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import lit
from functools import reduce
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml.feature import OneHotEncoder, IndexToString
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

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

subset_data = final_table_df.select('neigh_id', 'price', 'rooms','status')
labelIndexer_rooms = StringIndexer(inputCol="rooms", outputCol="indexed_rooms").fit(subset_data).setHandleInvalid("keep")
labelIndexer_status = StringIndexer(inputCol="status", outputCol="indexed_status").fit(subset_data).setHandleInvalid("keep")
indexers = [StringIndexer(inputCol=column, outputCol=column + "-index").fit(subset_data).setHandleInvalid("keep") for column in ['neigh_id']]
ohe_single_col = OneHotEncoder(inputCol="neigh_id-index", outputCol="neigh_id-onehot")
assembler = VectorAssembler(inputCols=['neigh_id-onehot', 'indexed_rooms', 'price'], outputCol="indexed_features")

(training_data, test_data) = final_table_df.randomSplit([0.75, 0.25])

rf = RandomForestClassifier(labelCol="indexed_status",
                            featuresCol="indexed_features",
                            numTrees=4,
                            maxDepth=5,
                            maxBins=32)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer_status.labels)

pipeline = Pipeline(stages=[labelIndexer_status, labelIndexer_rooms] + indexers + [ohe_single_col, assembler, rf])

model = pipeline.fit(training_data)

predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="indexed_status", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))  # 0.614
print("Accuracy = %g" % (accuracy))
rfModel = model.stages[-1]

model.write().overwrite().save('exploitation/model_Random_forest')
