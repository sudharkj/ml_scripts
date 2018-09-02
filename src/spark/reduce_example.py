# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("reduce").setMaster("local[*]")
sc = SparkContext(conf=conf)

inputIntegers = [1, 2, 3, 4, 5]
integerRdd = sc.parallelize(inputIntegers)

product = integerRdd.reduce(lambda x, y: x * y)
print("product: {}".format(product))
