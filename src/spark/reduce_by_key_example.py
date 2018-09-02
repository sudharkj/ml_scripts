# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("reduce-by-key").setMaster("local[*]")
sc = SparkContext(conf=conf)

lines = sc.textFile('data/word_count.text')
wordRdd = lines.flatMap(lambda line: line.split(' '))
wordPairRDD = wordRdd.map(lambda cur: (cur, 1))

wordCounts = wordPairRDD.reduceByKey(lambda x, y: x + y)
for word, count in wordCounts.collect():
    print("{}: {}".format(word, count))
