# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

import re

from pyspark import SparkConf, SparkContext


COMMA_DELIMITER = re.compile(''',(?=(?:[^"]*"[^"]*")*[^"]*$)''')


def split_comma(line):
    splits = COMMA_DELIMITER.split(line)
    return '{}, {}'.format(splits[1], splits[2])


conf = SparkConf().setAppName("airports").setMaster("local[*]")
sc = SparkContext(conf=conf)

airports = sc.textFile('data/airports.text')
airportsInUSA = airports.filter(lambda line: COMMA_DELIMITER.split(line)[3] == '"United States"')

airportNamesAndCityNames = airportsInUSA.map(split_comma)
airportNamesAndCityNames.saveAsTextFile('data/airports_in_usa.text')
