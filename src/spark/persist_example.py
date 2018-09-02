# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext, StorageLevel


conf = SparkConf().setAppName("persist").setMaster("local[*]")
sc = SparkContext(conf=conf)

inputIntegers = [1, 2, 3, 4, 5]
integerRdd = sc.parallelize(inputIntegers)

integerRdd.persist(StorageLevel.MEMORY_ONLY)

integerRdd.reduce(lambda x, y: x * y)

integerRdd.count()
