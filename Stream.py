from pyspark.ml import PipelineModel
from pyspark.ml.feature import IndexToString
from pyspark.sql import SparkSession
from pyspark.sql.functions import  split


spark = SparkSession.builder \
    .appName("myApp") \
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2') \
    .getOrCreate()

# Read the data stream from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "pidgeot.fib.upc.edu:9092") \
    .option("subscribe", "bdm_p2") \
    .load() \
    .selectExpr("CAST(value AS STRING)")

model = PipelineModel.load("exploitation/model_Random_forest")

data = df.withColumn('neigh_id', split(df['value'], ',').getItem(1)) \
        .withColumn('price', split(df['value'], ',').getItem(2)) \
        .select('neigh_id', 'price')
data = data.withColumn('price', data['price'].cast('double'))

# Add the original label column to the predictions
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=model.stages[0].labels)
prediction = model.transform(data).select("neigh_id", "price", "prediction", "predictedLabel")

query = prediction \
        .writeStream \
        .format("console") \
        .start()



query.awaitTermination(timeout=30)