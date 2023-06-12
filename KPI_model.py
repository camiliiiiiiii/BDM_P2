import findspark
from pyspark.ml.feature import OneHotEncoder
findspark.init()
from pyspark.sql.functions import col, sum, round, avg
from pyspark.sql.functions import split, when, col
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F



# Create a SparkSession
spark = SparkSession.builder \
    .appName("myApp") \
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2') \
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
na_counts.show()

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
final_table_df = final_table_df.fillna({'neigh_id': 'missing', 'rooms': 0.0, 'price': 0.0})
final_table_df = final_table_df.na.drop()
final_table_rdd = final_table_df.rdd

#####
#DESCRIPTIVE KPI

print("Saving KPIs: ")
# 1. Average price per neighborhood
avg_price_neighborhood = final_table_df.groupBy("neigh_id", "neigh").avg("price")
avg_price_neighborhood = avg_price_neighborhood.withColumn("avg_price_rounded", round(col("avg(price)"), 2))
KPI1_avg_price_neighborhood = avg_price_neighborhood.drop("avg(price)")
KPI1_avg_price_neighborhood.show()
KPI1_avg_price_neighborhood_pandas = KPI1_avg_price_neighborhood.toPandas()
KPI1_avg_price_neighborhood_pandas.to_csv('exploitation/KPI/KPI1_avg_price_neighborhood_pandas.csv', index=False)
# 2. Correlation between price and family income per neighborhood

avg_rent_income_neighborhood = final_table_df.groupBy("neigh_id", "neigh").agg(F.avg("avg_rent").alias("average_rent"), F.avg("avg_income").alias("average_income"))
KPI2_avg_rent_income_neighborhood = avg_rent_income_neighborhood
KPI2_avg_rent_income_neighborhood.show()
KPI2_avg_rent_income_neighborhood = KPI2_avg_rent_income_neighborhood.withColumn("average_rent", KPI2_avg_rent_income_neighborhood["average_income"].cast(IntegerType()))
KPI2_avg_rent_income_neighborhood_pandas = KPI2_avg_rent_income_neighborhood.toPandas()
KPI2_avg_rent_income_neighborhood_pandas.to_csv('exploitation/KPI/KPI2_avg_rent_income_neighborhood_pandas.csv', index=False)
# 3. Correlation between neighborhood and status
# Get distinct status values
distinct_status_values = final_table_df.select("status").distinct().rdd.flatMap(lambda x: x).collect()

# Calculate count of properties for each status per neighborhood
status_counts_per_neighborhood = final_table_df.groupBy("neigh_id", "neigh").pivot("status").count()

# Fill missing values with 0 for status values that are not present in a neighborhood
for status_value in distinct_status_values:
    status_counts_per_neighborhood = status_counts_per_neighborhood.fillna(0, subset=status_value)

status_counts_per_neighborhood.show()
KPI3_status_counts_per_neighborhood_pandas = status_counts_per_neighborhood.toPandas()
KPI3_status_counts_per_neighborhood_pandas.to_csv('exploitation/KPI/KPI3_status_counts_per_neighborhood_pandas.csv', index=False)

# 4. Average number of rooms and average price by neighborhood
rooms_price_neigh = final_table_df.groupBy("neigh_id", "neigh").agg(avg('rooms').alias('avg_rooms'), avg('price').alias('avg_price'))
# Show the KPI DataFrame
KPI4_rooms_price_neigh = rooms_price_neigh.withColumn("avg_rooms", round("avg_rooms", 2)).withColumn("avg_price", round("avg_price", 2))
KPI4_rooms_price_neigh = KPI4_rooms_price_neigh.withColumn("avg_price", KPI4_rooms_price_neigh["avg_price"].cast(IntegerType()))
KPI4_rooms_price_neigh = KPI4_rooms_price_neigh.withColumn("avg_rooms", KPI4_rooms_price_neigh["avg_rooms"].cast(IntegerType()))
KPI4_rooms_price_neigh.show()
KPI4_rooms_price_neigh_pandas = KPI4_rooms_price_neigh.toPandas()
KPI4_rooms_price_neigh_pandas.to_csv('exploitation/KPI/KPI4_rooms_price_neigh_pandas.csv', index=False)

# Show the rounded KPI DataFrame


#PREDICTIVE KPI
subset_data = final_table_df.select('neigh_id', 'price','status')

labelIndexer_status = StringIndexer(inputCol="status", outputCol="indexed_status").fit(subset_data).setHandleInvalid("keep")
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").fit(subset_data).setHandleInvalid("keep") for column in ['neigh_id']]
ohe_single_col = OneHotEncoder(inputCol="neigh_id_index", outputCol="neigh_id_onehot")
print("Schema: ")
subset_data.printSchema()

assembler = VectorAssembler(inputCols=['neigh_id_onehot', 'price'], outputCol="indexed_features", handleInvalid="keep")

(training_data, test_data) = subset_data.randomSplit([0.75, 0.25])

rf = RandomForestClassifier(labelCol="indexed_status",
                            featuresCol="indexed_features",
                            numTrees=4,
                            maxDepth=5,
                            maxBins=32)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer_status.labels)

pipeline = Pipeline().setStages([labelIndexer_status] + indexers + [ohe_single_col, assembler, rf, labelConverter])

model = pipeline.fit(training_data)

# Apply the model on test_data
predictions = model.transform(test_data).select("neigh_id", "price", "indexed_status","prediction", "predictedLabel")

#evaluator = MulticlassClassificationEvaluator(labelCol="indexed_status", predictionCol="predictedLabel", metricName="accuracy")
#accuracy = evaluator.evaluate(predictions)


#prediction = predictions.select("neigh_id", "indexed_status", "prediction")
print("Predictions: ")
# Show the predictions
predictions.show()

correct_predictions = predictions.filter(col("indexed_status") == col("prediction"))
incorrect_predictions = predictions.filter(col("indexed_status") != col("prediction"))


total_count = predictions.count()
correct_count = correct_predictions.count()
incorrect_count = incorrect_predictions.count()

accuracy = correct_count / total_count
print("Model Results: ")
print("Total Predictions: ", total_count)
print("Correct Predictions: ", correct_count)
print("Incorrect Predictions: ", incorrect_count)
print("Accuracy: ", accuracy*100)

rfModel = model.stages
model.write().overwrite().save('exploitation/model_Random_forest')
