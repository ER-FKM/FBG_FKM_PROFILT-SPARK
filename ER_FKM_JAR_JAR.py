from pyspark.sql import SparkSession
from pyspark import SparkConf

conf = SparkConf() \
        .setAppName("InitalMode") \
        .setMaster("spark://spark-master-c10:7077") \
        .set("spark.dynamicAllocation.enabled","True") \
        .set("spark.executor.memory","6g") \
        .set("spark.executor.instances", "2") \
        .set("spark.executor.memoryOverhead","10g") \
        .set("spark.executor.cores", "2") \
        .set("spark.driver.memory", "32g") \
        .set("spark.driver.maxResultSize", "32g") \
        .set("spark.network.timeout","600s") \
        .set("spark.sql.shuffle.partitions","200") \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .set("spark.eventLog.enabled","True")\
        .set("spark.eventLog.dir","file:/tmp/spark-events")\

spark = SparkSession.builder \
    .config(conf=conf) \
    .getOrCreate()
