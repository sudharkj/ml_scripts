# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

import re

from pyspark import SparkConf, SparkContext


COMMA_DELIMITER = re.compile(''',(?=(?:[^"]*"[^"]*")*[^"]*$)''')

conf = SparkConf().setAppName("stack-overflow").setMaster("local[*]")
sc = SparkContext(conf=conf)
total = sc.accumulator(0)
missingSalaryMidPoint = sc.accumulator(0)
processedBytes = sc.accumulator(0)
responseRDD = sc.textFile('data/2016-stack-overflow-survey-responses.csv')


def filter_response_from_canada(response):
    processedBytes.add(len(response.encode('utf-8')))
    splits = COMMA_DELIMITER.split(response)
    total.add(1)
    if not splits[14]:
        missingSalaryMidPoint.add(1)
    return splits[2] == "Canada"


responseFromCanada = responseRDD.filter(filter_response_from_canada)
print("Count of responses from Canada: {}".format(responseFromCanada.count()))
print("Number of bytes processed: {}".format(processedBytes.value))
print("Total count of responses: {}".format(total.value))
print("Count of responses missing salary middle point: {}".format(missingSalaryMidPoint.value))
