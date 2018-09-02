# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("pair-rdd").setMaster("local[*]")
sc = SparkContext(conf=conf)

tuples = [("Lily", 23), ("Jack", 29), ("Mary", 29), ("James", 8)]
pairRDD = sc.parallelize(tuples)

pairRDD.coalesce(1).saveAsTextFile("data/pair_rdd_from_tuple_list")
