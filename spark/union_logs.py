# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("union-logs").setMaster("local[*]")
sc = SparkContext(conf=conf)

julyLogs = sc.textFile('data/nasa_19950701.tsv')
augustLogs = sc.textFile('data/nasa_19950701.tsv')

aggregateLogs = julyLogs.union(augustLogs)
cleanLogs = aggregateLogs.filter(lambda line: not (line.startswith("host") and "bytes" in line))
sample = cleanLogs.sample(withReplacement=True, fraction=0.1)

sample.saveAsTextFile('data/sample_nasa_logs.csv')
