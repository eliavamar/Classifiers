import findspark
from pyspark import SparkContext
import numpy as np


def naive_bayes(query):
    sc = SparkContext(master="local[*]", appName="Simple App")
    input_data = sc.parallelize([("hello there", 0), ("hi there", 0), ("go home", 1), ("see you", 1), ("good bye to you", 1)])
    pk = input_data.map(lambda tup: (tup[1], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
    ptot = sum(pk.values())
    pki = input_data \
        .flatMap(lambda tup: list([(tup[1], w) for w in tup[0].split()])) \
        .map(lambda tup: ((tup[0], tup[1]), 1)) \
        .reduceByKey(lambda a, b: a + b).collectAsMap()
    class_probs = [(pk[k]+1) / (float(ptot)+2) * np.prod(np.array([(pki.get((k, i), 0)+1) / (float(pk[k])+2) for i in query.split()])) for k in range(0, 2)]
    print(class_probs)
    y_star = np.argmax(np.array(class_probs))
    print(y_star)

if __name__ == '__main__':
    findspark.init()
    q = "hello hi"
    naive_bayes(q)
