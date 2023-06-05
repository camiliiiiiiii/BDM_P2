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
income_RDD = spark.read.format("mongo") \
    .option('uri', 'mongodb://10.4.41.45/BDM_P2.income') \
    .load() \
    .rdd


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

    idealista_RDD = df.rdd
else:
    print("No valid files found")

#count distinct number

sample_records =idealista_RDD.take(1)
for record in sample_records:
    print(record)

distinct_property_codes_count = idealista_RDD.map(lambda row: row.propertyCode).distinct().count()
print("Distinct propertyCode count:", distinct_property_codes_count)
#####

##RENT-OPENDATA
def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

rent_extra_ne_num_rdd = rent_extra_ne_RDD.filter(lambda x: is_numeric(x[7])).map(lambda x: (float(x[7]), x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[8]))
print("Count after filter: ", rent_extra_ne_num_rdd.count())  # Add this line


rent_extra_ne_num_RDD = rent_extra_ne_num_rdd \
    .filter(lambda x: x[7] != '--' and x[7] != None and x[7] != 'NA'and x[7] !='N/A') \
    .filter(lambda x: 2022 == x[1] or 2021 == x[1] or 2020 == x[1]) \
    .map(lambda x: (x[5], float(x[7]))) \
    .groupByKey() \
    .mapValues(lambda x: round(sum(x) / len(x), 2)) \
    .map(lambda x: (x[0], x[1])) \
    .cache()

print("Count after filter: ", rent_extra_ne_num_RDD.count())  # Add this line

sample_records = rent_extra_ne_num_RDD.take(5)
for record in sample_records:
    print(record)

### INCOME


# Income  RDDs
sample_records = income_RDD.take(5)
print(income_RDD.count())

for record in sample_records:
    print('income_RDD:',record)

def avg_pop(rows):
    list_population = []
    max_year = 0
    for row in rows:
        year = row.year
        if year > max_year:
            max_year = year
            list_population = [row.pop]
        elif year == max_year:
            list_population.append(row.pop)
    if len(list_population) > 0:
        return round(sum(list_population) / len(list_population), 2)
    else:
        return 0

def avg_income(rows):
    list_RDF = []
    max_year = 0
    for row in rows:
        year = row.year
        if year > max_year:
            max_year = year
            list_RDF = [row.RFD]
        elif year == max_year:
            list_RDF.append(row.RFD)
    if len(list_RDF) > 0:
        return round(sum(list_RDF) / len(list_RDF), 2)
    else:
        return 0

# Original code
sample_records = income_RDD.take(5)
print(income_RDD.count())
for record in sample_records:
    print(record)

rdd_income_ne = income_RDD \
    .map(lambda x: (x[4], avg_pop(x[3]), avg_income(x[3])))\
    .map(lambda x: (x[0], list(x[1:]))) \
    .cache()

print("Count after filter income: ", rdd_income_ne.count())
sample_records = rdd_income_ne.take(2)
for record in sample_records:
    print(record)



# JOIN INCOME + RENT OPEN DATA

rdd_income_rent_opendata_ne = rent_extra_ne_num_RDD \
    .join(rdd_income_ne) \
    .map(lambda x: (x[0], Row(avg_pop=x[1][1][0], avg_income=x[1][1][1], avg_rent=x[1][0]))) \
    .cache()

sample_records = rdd_income_rent_opendata_ne.take(1)
for record in sample_records:
    print(record)


sample_records = income_lookup_ne_RDD.take(1)
for record in sample_records:
    print("lookup:",record)

lookup_income_rent = income_lookup_ne_RDD.map(lambda x: (x[1], (x[3], x[0]))).cache()
join_income_rent = rdd_income_rent_opendata_ne.join(lookup_income_rent).cache()

sample_records = lookup_income_rent.take(1)
for record in sample_records:
    print("join income-lookup:",record)

#%%
join_p1= join_income_rent.map(lambda x: (x[1][1][1], Row(neigh_name=x[1][1][0],
                                                                 avg_pop=x[1][0][0],
                                                                 avg_income=x[1][0][1],
                                                                 avg_rent=x[1][0][2]))).cache()
join_p1.take(1)



##IDEALISTA

idealista_ne_RDD = idealista_RDD\
    .map(lambda x: (x['propertyCode'], x))\
    .reduceByKey(max)\
    .map(lambda x: (x[1]['date'], x))\
    .map(lambda x: (x[1][1]['neighborhood'], x[1]))\
    .cache()
print(idealista_ne_RDD.count())
idealista_ne_RDD.take(1)

sample_records =idealista_ne_RDD.take(1)
for record in sample_records:
    print(record)

idealista_ne_RDD = idealista_ne_RDD.map(lambda x: (x[0],
    Row(property_id=x[1][0],
        rooms=x[1][1]['rooms'],
        bathrooms=x[1][1]['bathrooms'],
        size=x[1][1]['size'],
        price=x[1][1]['price'],
        status= x[1][1]['status'],
        hasLift =x[1][1]['hasLift'],
        priceByArea=x[1][1]['priceByArea'])
)).cache()

sample_records =idealista_ne_RDD.take(1)
for record in sample_records:
    print("idealista_rdd:",record)


lookup_idealista = rent_lookup_ne_RDD.map(lambda x: (x[1], (x[3], x[0]))).cache()

sample_records =lookup_idealista.take(1)
for record in sample_records:
    print("idealista_look_up:",record)

join_idealista = idealista_ne_RDD.join(lookup_idealista).cache()

sample_records =join_idealista.take(1)
for record in sample_records:
    print("join_idealista:",record)

join_idealista_f = join_idealista.map(lambda x: (x[1][1][1],Row(neigh=x[1][1][0], property_id=x[1][0][0], rooms=x[1][0][1],
                                   bathrooms=x[1][0][2],size=x[1][0][3], price=x[1][0][4],status=x[1][0][5],
                                    hasLift=x[1][0][6],priceByArea=x[1][0][7])))\
                                    .cache()

sample_records= join_idealista_f.take(1)
for record in sample_records:
    print("idealista_f: ",record)

sample_records= join_p1.take(1)
for record in sample_records:
    print("join-income-rent: ",record)

final_result1 = join_idealista_f.join(join_p1).cache()
sample_records= final_result1.take(3)
for record in sample_records:
    print("pre-final: ",record)

final_result_ff = final_result1.map(lambda x: Row(neigh_id=x[0], neigh=x[1][0][0], avg_pop=x[1][1][1], avg_income=x[1][1][2],
                                avg_rent=x[1][1][3],property_id=x[1][0][1], rooms=x[1][0][2],
                                   bathrooms=x[1][0][3],size=x[1][0][4], price=x[1][0][5],status=x[1][0][6],
                                    hasLift=x[1][0][7],priceByArea=x[1][0][8]))
sample_records= final_result_ff.take(3)
for record in sample_records:
    print("final: ",record)

df = final_result_ff.toDF()
type(df)

final_result_ff.toDF().printSchema()
final_result_ff.toDF().show()

df.coalesce(1).write.format('json').save('final_results.json')
df.write.format('com.mongodb.spark.sql.DefaultSource').option( "uri", f"mongodb://10.4.41.45/Formatted_BDM_P2.Final_results").save()

