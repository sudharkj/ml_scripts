# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("take").setMaster("local[*]")
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

inputWords = ["spark", "hadoop", "spark", "hive", "pig", "cassandra", "hadoop"]
wordRdd = sc.parallelize(inputWords)

words = wordRdd.take(3)
for word in words:
    print(word)
