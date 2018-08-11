# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

import re

from pyspark import SparkConf, SparkContext


COMMA_DELIMITER = re.compile(''',(?=(?:[^"]*"[^"]*")*[^"]*$)''')

conf = SparkConf().setAppName("airports").setMaster("local[*]")
sc = SparkContext(conf=conf)

lines = sc.textFile('data/airports.text')
countryAndAirportNameAndPair = lines \
    .map(lambda airport: (COMMA_DELIMITER.split(airport)[3], COMMA_DELIMITER.split(airport)[1]))

airportsByCountry = countryAndAirportNameAndPair.groupByKey()

for country, airportName in airportsByCountry.collectAsMap().items():
    print("{}: {}".format(country, list(airportName)))
