# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

import re

from pyspark import SparkConf, SparkContext


COMMA_DELIMITER = re.compile(''',(?=(?:[^"]*"[^"]*")*[^"]*$)''')

conf = SparkConf().setAppName("airports").setMaster("local[*]")
sc = SparkContext(conf=conf)

airportsRDD = sc.textFile('data/airports.text')
airportPairRDD = airportsRDD.map(lambda line: (COMMA_DELIMITER.split(line)[1], COMMA_DELIMITER.split(line)[3]))
airportsNotInUSA = airportPairRDD.filter(lambda key_value: key_value[1] != "\"United States\"")

airportsNotInUSA.saveAsTextFile('data/airports_not_in_usa_pair_rdd.text')
