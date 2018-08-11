# From the course: Apache Spark with Python - Big Data with PySpark and Spark
# Link: https://www.udemy.com/draft/1386444/
# Dataset: https://github.com/jleetutorial/python-spark-tutorial

import re

from pyspark import SparkConf, SparkContext


COMMA_DELIMITER = re.compile(''',(?=(?:[^"]*"[^"]*")*[^"]*$)''')


def load_post_code_map():
    lines = open("data/uk-postcode.csv", "r").read().split("\n")
    splits_for_lines = [COMMA_DELIMITER.split(line) for line in lines if line != ""]
    return {splits[0]: splits[7] for splits in splits_for_lines}


def get_post_prefix(line: str):
    splits = COMMA_DELIMITER.split(line)
    postcode = splits[4]
    return None if not postcode else postcode.split(" ")[0]


conf = SparkConf().setAppName('uk-maker-spaces').setMaster("local[*]")
sc = SparkContext(conf=conf)

postCodeMap = sc.broadcast(load_post_code_map())

makerSpaceRdd = sc.textFile("data/uk-makerspaces-identifiable-data.csv")

regions = makerSpaceRdd \
  .filter(lambda line: COMMA_DELIMITER.split(line)[0] != "Timestamp") \
  .filter(lambda line: get_post_prefix(line) is not None) \
  .map(lambda line: postCodeMap.value[get_post_prefix(line)] \
    if get_post_prefix(line) in postCodeMap.value else "Unknown")

for region, count in regions.countByValue().items():
    print("{} : {}".format(region, count))
