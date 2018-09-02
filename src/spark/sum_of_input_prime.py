# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("prime-numbers").setMaster("local[*]")
sc = SparkContext(conf=conf)

lines = sc.textFile('data/prime_nums.text')
numbers = lines.flatMap(lambda line: line.split())

validNumbers = numbers.filter(lambda number: number)

print("sum: {}".format(validNumbers.reduce(lambda x, y: int(x) + int(y))))
