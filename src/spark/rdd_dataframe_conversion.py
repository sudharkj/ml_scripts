# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

import re

from pyspark.sql import SparkSession


COMMA_DELIMITER = re.compile(''',(?=(?:[^"]*"[^"]*")*[^"]*$)''')


def map_response_rdd(line: str):
    splits = COMMA_DELIMITER.split(line)
    double1 = None if not splits[6] else float(splits[6])
    double2 = None if not splits[14] else float(splits[14])
    return splits[2], double1, splits[9], double2


def get_col_names(line: str):
    splits = COMMA_DELIMITER.split(line)
    return [splits[2], splits[6], splits[9], splits[14]]


session = SparkSession.builder.appName("stack-over-flow-survey").master("local[*]").getOrCreate()
sc = session.sparkContext

lines = sc.textFile("data/2016-stack-overflow-survey-responses.csv")

responseRDD = lines \
    .filter(lambda line: not COMMA_DELIMITER.split(line)[2] == "country") \
    .map(map_response_rdd)

colNames = lines \
    .filter(lambda line: COMMA_DELIMITER.split(line)[2] == "country") \
    .map(get_col_names)

responseDataFrame = responseRDD.toDF(colNames.collect()[0])

print("=== Print out schema ===")
responseDataFrame.printSchema()

print("=== Print 20 records of responses table ===")
responseDataFrame.show(20)

for response in responseDataFrame.rdd.take(10):
    print(response)
