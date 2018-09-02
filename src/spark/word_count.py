# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("word-count").setMaster("local[*]")
sc = SparkContext(conf=conf)

lines = sc.textFile('data/word_count.text')
words = lines.flatMap(lambda line: line.split(' '))

word_counts = words.countByValue()
for word, count in word_counts.items():
    print(word, count)
