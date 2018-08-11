# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("count").setMaster("local[*]")
sc = SparkContext(conf=conf)

inputWords = ["spark", "hadoop", "spark", "hive", "pig", "cassandra", "hadoop"]
wordRdd = sc.parallelize(inputWords)
print("Count: {}".format(wordRdd.count()))

wordCountByValue = wordRdd.countByValue()
print("Count by value: ")
for word, count in wordCountByValue.items():
    print("{}: {}".format(word, count))
