# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark.sql import SparkSession, functions as fs

session = SparkSession.builder.appName("uk-maker-spaces").master("local[*]").getOrCreate()

makerSpace = session.read.option("header", "true") \
    .csv("data/uk-makerspaces-identifiable-data.csv")

postCode = session.read.option("header", "true").csv("data/uk-postcode.csv") \
    .withColumn("PostCode", fs.concat_ws("", fs.col("PostCode"), fs.lit(" ")))

print("=== Print 20 records of makerspace table ===")
makerSpace.select("Name of makerspace", "Postcode").show()

print("=== Print 20 records of postcode table ===")
postCode.select("PostCode", "Region").show()

joined = makerSpace \
    .join(postCode, makerSpace["Postcode"].startswith(postCode["Postcode"]), "left_outer")

print("=== Group by Region ===")
joined.groupBy("Region").count().show(200)
