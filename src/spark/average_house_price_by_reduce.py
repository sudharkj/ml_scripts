# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("avg-house-price").setMaster("local[*]")
sc = SparkContext(conf=conf)

lines = sc.textFile('data/RealEstate.csv')
cleanedLines = lines.filter(lambda line: "Bedrooms" not in line)

housePricePairRDD = cleanedLines.map(lambda line: (line.split(",")[3], (1, float(line.split(",")[2]))))
housePriceTotal = housePricePairRDD.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

print("house price total: ")
for bedroom, avgCount in housePriceTotal.collect():
    print("{}: ({}, {})".format(bedroom, avgCount[0], avgCount[1]))

housePriceAvg = housePriceTotal.mapValues(lambda avg_count: avg_count[1] / avg_count[0])
print("house price average: ")
for bedroom, avgCount in housePriceAvg.collect():
    print("{}: {}".format(bedroom, avgCount))
