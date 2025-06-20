from pyspark.sql import SparkSession
from pyspark import SparkConf

conf = SparkConf() \
        .setAppName("FKM-ERs") \
        .setMaster("spark://spark-master-c10:7077") \
        .set("spark.dynamicAllocation.enabled","False") \
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

sparkcontxt = spark.sparkContext
sparkcontxt.setLogLevel("WARN")


def BlockingKey_Generator(x,bk0):
  import pandas as pd
  import numpy as np
  import random
  import operator
  import os
  import re
  import jellyfish

  codif = list()
  codif.append("SOUNDEX")
  codif.append("METAPHONE")
  codif.append("NYSIIS")
  codif.append("All")
  codif.append("Firts Caracters")
  codif.append("Firts numbers of adresse")

  thekey = str
  keytemp = str
  thekey = ''
  keytem = ''
  MyWord = ''
  TheBK = list()

  for t in range(len(bk0)):
      bk = bk0[t]
      keytem = ''
      keytem1 = ''
      thekey = ''
      for z in range(len(bk)):
          keytem1 = ''
          keytem = ''
          text=''
          text = str(x[bk[z][0]])
          text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
          text = re.sub(r'\s+', ' ', text)
          Myword = text.lower()
          if (bk[z][1]==0):
              keytem1=str(jellyfish.soundex(Myword))
              keytem = keytem1
          elif (bk[z][1]==1):
              keytem1=str(jellyfish.metaphone(str.strip(Myword)))
              keytem = keytem1
          elif (bk[z][1]==2):
              keytem1=str(jellyfish.nysiis(str.strip(Myword)))
              keytem = keytem1
          elif (bk[z][1]==3):
              keytem1=Myword #str(x[n])
              keytem = keytem1
          elif (bk[z][1]==4):
              firstcar = bk[z][2]
              keytem1=Myword #str(x[n])
              keytem = keytem1
              keyte1  = keytem[0,firstcar]
              keytem = keytem1
          elif (bk[z][1]==5):
              keytem1=Myword #str(x[n])
              keytem = keytem1
              keytem1 = ""
              for h in range(len(keytem)) :
                  if (keytem[h].isdigit): keytem1 = keytem1 + keytem[h]
              keytem = keytem1
          elif (bk[z][1]==6):
              keytem = Myword
              indice = int(min(150,len(keytem)))
              keytem  = keytem[0:indice]
          elif (bk[z][1]==7):
              keytem = Myword
              indice = int(min(120,len(keytem)))
              keytem  = keytem[0:indice]
          elif (bk[z][1]==8):
              keytem = Myword[-15:]
          thekey = thekey +  str.strip(keytem)
      if (t==0): TheBK = thekey
      elif (t>0): TheBK = TheBK +','+ thekey
  return TheBK

import Levenshtein as lev
import jaro as jaro
import numpy as np
def sentence_similarity(sentence1, sentence2):
    # Tokenize sentences into words
    words1 = sentence1.split()
    words2 = sentence2.split()
    if (len(words1)>=len(words2)):
      words3 = words1
      words1 = words2
      words2 = words3

    similarity_matrix = np.zeros((len(words1), len(words2)))

    for i, word1 in enumerate(words1):
        for j, word2 in enumerate(words2):
            similarity_matrix[i][j] = jaro.jaro_winkler_metric(word1,word2)   #lev.ratio(word1, word2)
    max_similarities = np.max(similarity_matrix, axis=1)
    filtered_matrix = max_similarities[max_similarities != 0]
    #print(filtered_matrix)
    sentence_sim = np.mean(filtered_matrix)
    #print(sentence_similarity)
    return sentence_sim

def jaccard_similarity(pair1, pair2):
    set1 = set(pair1)
    set2 = set(pair2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def CalculateDissimilaritiesrdd(df_full0,clustersNumber,nColumn,cpt):
  import Levenshtein as lev
  import pandas as pd
  import csv
  import jaro as jaro
  membership_temp = list()
  clusterdiss = list()
  clusterdisstemp = list()
  membershiptemp = list()
  distance_p1 = list()
  distance_temp = list()
  attributes = list()
  karim = 1
  df_full = df_full0[0]
  cc = df_full0[1]
  if (cpt >0):
    sorted_data = []
    sorted_data = sorted(cc, key=lambda x: x[0])
    clustercenter = [list(item[1]) for item in sorted_data]
  else:
    clustercenter = cc

  alpha = 1
  beta = 1
  attributes = list()
  for k in range(nColumn):
      attributes.append(df_full[k])
  for y in range(clustersNumber):
      dissmilarities = 0
      cctemp = clustercenter[y]
      distance_p_temp = 0.0
      for z in range(1,nColumn):
          distance_p_temp = distance_p_temp + (1-jaro.jaro_winkler_metric(str(df_full[z]).replace(" ",""),str(cctemp[z-1]).replace(" ","")))
      distance_temp.append(distance_p_temp)
  distance_p1.append([attributes+distance_temp])
  return attributes+distance_temp

import numpy as np

def updateMembership(distance_dp1,k,nColumn):
    distance_dp = distance_dp1[nColumn:(k + nColumn)]  
    distances = np.array(distance_dp)
    C = len(distances)
    memberships = np.zeros(C)

    for j in range(C):
        valid_distances = distances[distances == 0]
        if len(valid_distances) == 0:
            valid_distances = distances[distances != 0]
            denominator = np.sum((distances[j] / valid_distances) ** 2)
        elif len(valid_distances) > 0:
            index = np.where(distances == 0)[0]
            memberships = np.zeros(C)
            denominator=1
            j = index
            memberships[j] = 1
            break
        memberships[j] = 1 / denominator if denominator != 0 else 0
    t = [i for i, _ in sorted(enumerate(list(memberships)), key=lambda x: x[1], reverse=True)]
    return distance_dp1[:nColumn]+list(memberships)+t

clustersNumber = 100 
nColumn = 3

############### Data Loading + New Data (Bloking Key) ###########################
print("**************************************************************")
print("--------->  Data Loading + New Data (Bloking Key)  ")
print("**************************************************************")
from datetime import datetime
nowload1 = datetime.now()
nowall1 = datetime.now()
print("Start time :", nowload1.strftime("%Y-%m-%d %H:%M:%S"))
print("**************************************************************")
print("")
####### 1- Generate The Blocking Key ########
bk0 = list()
BK0 = [[0,3]]
BK1 = [[1,3]]
BK2 = [[2,3]]
bk0.append(BK0)
bk0.append(BK1)
bk0.append(BK2)
data1=sparkcontxt.textFile('dataset.csv')
data2 = data1.map(lambda y: y.split(","))
DataBK = data2.map(lambda x: BlockingKey_Generator(x,bk0))
pairs = DataBK.map(lambda y: y.split(","))
sizeData =DataBK.count()
nowim1 = datetime.now()
"""
print("**************************************************************")
print("- Frequency ")
print("**************************************************************")

import Levenshtein as lev
import difflib
from fuzzywuzzy import fuzz
import jaro as jaro
alpha = 1
beta = 1
#clustersNumber = 220

textRDD = DataBK
pairs = DataBK.map(lambda y: y.split(","))
frenqRdd = pairs.map(lambda x: (tuple(x), 1)).reduceByKey(lambda a, b: a + b)
ma_liste = frenqRdd.collect()

selected = []
result = []
all = []
l = []
a = 0
i = 0
j = 0
b = True
nn = 0
cptt = 0
while i <= len(ma_liste0) - 2:
  if b:
    #print(ma_liste0[i][0][0])
    nn = ma_liste0[i][0][0]
    #if ((nn == '429')or(nn == '818')):
      #print('ma_liste0')
      #print(ma_liste0)
    selected = []
    sousliste = ma_liste0[i]
    freq = ma_liste0[i][1]
    word = ma_liste0[i][0]
    selected.append(ma_liste0[i][0][0])
    ma_liste0.pop(i)
    b = False
  j = i #+1

  while(((str(word[1][0]) == str(ma_liste0[j][0][1][0]))or(jaro.jaro_winkler_metric(str(word[1]),str(ma_liste0[j][0][1])) >= 0.4))and(j<=len(ma_liste0) - 3)):
      t = 1
      soo = True
      prob = 0
      while ((t<=nColumn-1)and(soo)):
        #if ((nn == '90')or(nn == '536')):
        #  print(jaro.jaro_winkler_metric(str(word[t]),str(ma_liste0[j][0][t])))
        #if ((jaro.jaro_winkler_metric(str(word[t]),str(ma_liste0[j][0][t])) < 0.75)) :
        if ((jaro.jaro_winkler_metric(str(word[t]),str(ma_liste0[j][0][t])) < 0.55)) :
        #if ((sentence_similarity(str(word[t]),str(ma_liste0[j][0][t])) < 0.75)) :
          soo = False
          break
        prob = prob +  jaro.jaro_winkler_metric(str(word[t]),str(ma_liste0[j][0][t]))
        t += 1
      if(prob/(nColumn-1)<0.8):
        soo = False

      if (soo) :
        #if (nn == '58'):
        #  print('word 2')
        #  print(ma_liste0[j][0][0])
        #i=i+1
        freq = freq + ma_liste0[j][1]
        selected.append(ma_liste0[j][0][0])
        ma_liste0.pop(j)
      else:
        j += 1
    else:
      j += 1
  if (i <= len(ma_liste0) - 2):
  #else:
    f = 2
    ma_liste1 = ma_liste0
    while ((f<=nColumn-1)):
      new = False
      ma_liste1 = sorted(ma_liste1, key=lambda x: x[0][f])
      #if ((nn == '429')or(nn == '818')):
        #print('ma_liste5')
        #print(ma_liste1)
      y = 0
      while ((y <= len(ma_liste1) - 2)and(word[f] <= ma_liste1[y][0][f])and(jaro.jaro_winkler_metric(str(word[f]),str(ma_liste0[j][0][f])) >= 0.55)):
        soosoo = True
        ff=0
        while ((ff<=nColumn-1)and(soosoo)):
          if ((jaro.jaro_winkler_metric(str(word[ff]),str(ma_liste1[y][0][ff])) < 0.75)):
            soosoo = False
            break
          ff += 1
        if (soosoo):
            freq = freq + ma_liste1[y][1]
            selected.append(ma_liste1[y][0][0])
            ma_liste1.pop(y)
        #if ((jaro.jaro_winkler_metric(str(word[f]),str(ma_liste1[y][0][f])) > 0.6)):
        else:
            y += 1
      f += 1
    l.append((word,freq))
    all.append(selected)
    i = 0
    b = True
    ma_liste1 = sorted(ma_liste1, key=lambda x: x[0][1])
    ma_liste0 = ma_liste1


filtered_data = [entry for entry in l if '' not in entry[0]]
data_by_tuple = [((item[0][1], item[0][2]), item[1]) for item in filtered_data]
sorted_data_by_tuple = sorted(data_by_tuple , key=lambda x: x[1], reverse=True)
intial_mode= sorted_data_by_tuple[0:clustersNumber]
"""
import random
initial_mode0 = []
mydata = pairs.collect()
cpt = clustersNumber
r_numbers = [random.randint(0, sizeData-1) for _ in range(cpt)]
for i in range(len(random_numbers)):
  initial_mode0.append(((mydata[r_numbers[i]][1],mydata[r_numbers[i]][2]),cpt))
  cpt=cpt-1
intial_mode = initial_mode0
nowim2 = datetime.now()
clustersNumber=len(intial_mode)
############### Fuzzy K-Modes Algorithm ###########################
print("************************************************************************")
print("-->  Fuzzy K-Mode Algorithm - Update Mode : Old  -   ")
print("************************************************************************")
from datetime import datetime
nowfkm1 = datetime.now()
print("Start time :", nowfkm1.strftime("%Y-%m-%d %H:%M:%S"))
print("************************************************************************")
print("")
from pyspark.sql import SparkSession

def are_equal1(rdd1, rdd2):
    if(rdd1.isEmpty() or rdd2.isEmpty()):
        return False
    else:
        intersection_count = rdd1.intersection(rdd2).count()
        union_count = rdd1.union(rdd2).distinct().count()
        return intersection_count == rdd1.count() and intersection_count == rdd2.count() and intersection_count == union_count

def are_equal(rdd_a, rdd_b):
    return (rdd_a.subtract(rdd_b).isEmpty() and
            rdd_b.subtract(rdd_a).isEmpty())

import Levenshtein as lev
import jaro as jaro
import difflib
from fuzzywuzzy import fuzz
import pandas as pd
import csv

alpha = 1
beta = 1
endit = 5
cpt = 0
stop = True
data5 = []
clustercenter = []
t = []
####### 3- Iterate the steps of the Fuzzy K-Modes (Update The Membership Matrix, Calculate the New modes of the clusters) ########
modes_iterations = []
m = []
m1 = []
m0 = []
m01 = []
subset_to_check_set = []
bigger_subset_sets = []
cc = []
cc = intial_mode
sorted_data = sorted(cc, key=lambda x: x[1])
clustercenter = [list(item[0]) for item in sorted_data]
textRdd = DataBK
rdd_m = sparkcontxt.emptyRDD()
pairs = textRdd.map(lambda y: y.split(",")).persist()
ccrdd = sparkcontxt.parallelize(clustercenter)
pairs103 = pairs.map(lambda x: (15051982,x)).persist()
U = 0
while (cpt<=endit):
  data222 = []
  def f_lev(s1,s2):
    m = lev.ratio(s1,s2)
    return m
  pairs101 = ccrdd.map(lambda x: ((15051982,(x))))

  pairs102 = pairs101.groupByKey().mapValues(list).repartition(120)

  pairs104 = pairs103.join(pairs102).repartition(120)


  pairs105 = pairs104.map(lambda x: x[1])

  pairs2 = pairs105.map(lambda x: CalculateDissimilaritiesrdd(x,clustersNumber,nColumn,cpt))

  MembershipsMatrix = []
  pairs3 = pairs2.map(lambda x: updateMembership(x,clustersNumber,nColumn))

  data1 = []
  print("Nombre de partition : ",pairs3.getNumPartitions())
  liste1 = []
  liste2 = []

  if (cpt<=endit): 
    pairsk = pairs3.flatMap(lambda x: [((x[(nColumn+(clustersNumber*1))],i),(x[i],x[(nColumn + int(x[(nColumn+(clustersNumber*1))]))])) for i in range(1,nColumn)])#.sortBy(lambda x: x[0],ascending=True)

    rdd_grouped = pairsk.groupByKey().mapValues(list)


    def aggregate_similar_probs(records):
        results = []
        for key, values in records:
            grouped = {}
            for (third_value, prob) in values:
                if third_value not in grouped:
                    grouped[third_value] = prob
                else:
                    grouped[third_value] += prob

            aggregated = {}
            for val1, prob1 in grouped.items():
                merged = False
                for val2 in aggregated.keys():
                    #if (sentence_similarity(val1, val2) >= 0.92):
                    i#if (sentence_similarity(val1, val2) >= 0.65):
                    #if are_similar(val1, val2):
                    if (lev.ratio(val1.replace(" ",""), val2.replace(" ","")) >= 0.85):
                    #if (jaro.jaro_winkler_metric(val1, val2) >= 0.8):
                    #if (jaccard_similarity(val1, val2) >= 0.7): #cosine with 0.85
                        aggregated[val2] += prob1
                        merged = True
                        break
                if not merged:
                    aggregated[val1] = prob1

            for third_value, total_prob in aggregated.items():
                results.append((key[0], key[1], third_value, total_prob))
        return results

    aggregated_results = rdd_grouped.map(lambda x: aggregate_similar_probs([x])).flatMap(lambda x: x)#.collect()
    pairs7 = aggregated_results.map(lambda x: ((x[0], x[1]),(x[2], x[3])))

    pairs77 = pairs7.groupByKey().mapValues(list)#.sortBy(lambda x: x[0],ascending=True)

    pairs8 = pairs77.map(lambda a: (a[0], max(a[1], key=lambda x: x[1])))

    pairs88 = pairs8.map(lambda x: ((x[0][0], x[0][1]),(x[1][0])))

    pairs99 = pairs88.map(lambda x: ((x[0][0],( x[0][1],x[1])))).groupByKey().mapValues(list)

    def transform(group):
        key, values = group
        
        sorted_values = sorted(values, key=lambda x: x[0])
        
        extracted_values = [v[1] for v in sorted_values]
        return (key, extracted_values)

    pairs100 = pairs99.map(transform)
    epsilon = 1e-5
    #U_new = pairs3.map(lambda row: sum(row[nColumn:nColumn+3])).sum()
    #delta_U = np.sum(np.abs(U_new - U))  # Change in membership matrix
    #print("Some U_new, U, delta_U")
    #print(U)
    #print(U_new)
    #print(delta_U)
    #if delta_U < epsilon:
    #    print(f"Converged after {cpt} iterations.")
    #    break
    #else:
    #  U = U_new
    #  U_new = 0

    ccrdd = sparkcontxt.emptyRDD()
    ccrdd = pairs100
    clustersNumber = pairs100.count()
    data1 = []
    cpt=cpt+1
  else : cpt = 30

pairs3.persist()

nowfkm2 = datetime.now()
print(nowfkm2-nowfkm1)
print ('Compteur:')
print(cpt)

print("")
print("******* End of the phase : Fuzzy K-Modes Algorithms *********")
print("Finish time :", nowfkm2.strftime("%Y-%m-%d %H:%M:%S"))

print(" ******************************************************")
print("                      Thresholds T1  and T2                  ")
print(" ******************************************************")
from datetime import datetime
print("Start time :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("**************************************************************")
print("")
nowt1 = datetime.now()
nowt11 = datetime.now()
import numpy as np
import math as mt

data11 = []

def poids(p1, q1, p2, q2,nColumn):
    assert len(p1) == len(q1) == len(p2) == len(q2), "Distributions must have the same length"

    equal_indices = np.where(p2 == q2)[0]
    poids_array = np.zeros(len(equal_indices))
    poids = 0.00
    for i, idx in enumerate(equal_indices):
        poids_array[i] = p1[idx] * q1[idx]
        poids = poids + ((p1[idx]*(100-idx)) * (q1[idx] * (100-idx)) )
    return sum(poids_array)

def poids1(p1, q1, p2, q2,nColumn):

    equal_indices = np.where(p2 == q2)[0]
    poids_array = np.zeros(len(equal_indices))
    poids = 0.00
    for i, idx in enumerate(equal_indices):
        poids_array[i] = p1[idx] * q1[idx]
    return sum(poids_array) * len(equal_indices), len(equal_indices)

import numpy as np
from scipy.stats import norm

m01 = m1

RddNew = pairs3.map(lambda x: [int(x[0])] + x[1:])
infomatrix = []
PairsToMatch = sparkcontxt.emptyRDD()
cartesian_rdd1 = sparkcontxt.emptyRDD()
cartesian_rdd1_pairs = sparkcontxt.emptyRDD()
cartesian_rdd0_pairs = sparkcontxt.emptyRDD()
temp = sparkcontxt.emptyRDD()

print(" ******************************************************")
print(" ******************************************************")
print(" -------------------> Calculating...                   ")
print(" ******************************************************")
print(" ******************************************************")
from datetime import datetime
now = datetime.now()
print("Start time :", now.strftime("%Y-%m-%d %H:%M:%S"))
print("**************************************************************")
print("")

def outliers_modified_z_score(ys):
    outliers = []
    t1 = 0
    tt1 = 0
    threshold = 3.5
    ys = list(set(ys))
    ys = list(filter(lambda x: x != 1, ys))
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    outliers = [y for y, z in zip(ys, modified_z_scores) if abs(z) > threshold]
    if (len(outliers) > 0):
      t1 = min(outliers)
    else:
      t1 = 0.5
      if (len(ys)>0):
        tt1 = max(ys)
        #quartile_1, quartile_3 = np.percentile(ys, [25, 75])
        #tt1 = quartile_3
        if (tt1>0):
          t1 = tt1
        else:
          t1 = 0.5
      else:
        t1 = 0.5
    t1 = truncate_to_n_decimals(t1, 3)
    return t1 #np.where(np.abs(modified_z_scores) > threshold)

def outliers_modified_z_score2(ys):
    outliers = []
    t1 = 0
    tt1 = 0
    threshold = 5.5
    ys = list(set(ys))
    ys = list(filter(lambda x: x != 1, ys))
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    outliers = [y for y, z in zip(ys, modified_z_scores) if abs(z) > threshold]
    if (len(outliers) > 0):
      t1 = min(outliers)
    else:
      t1 = max(ys)
    t1 = truncate_to_n_decimals(t1, 3)
    return t1 #np.where(np.abs(modified_z_scores) > threshold)

import numpy as np

def compute_entropy(probabilities):
    p = probabilities[probabilities > 0]
    return -np.sum(p * np.log2(p))

def max_entropy_threshold(prob_array):

    candidate_thresholds = sorted(set(prob_array))
    best_thresh = None
    best_entropy = -np.inf

    for t in candidate_thresholds:
        filtered = prob_array[prob_array >= t]
        if filtered.size == 0:
            continue
        normalized = filtered / filtered.sum()
        entropy = compute_entropy(normalized)
        if entropy > best_entropy:
            best_thresh = t
            best_entropy = entropy
    return best_t2
    
rdd_transformed = RddNew.map(lambda x: (x[nColumn+clustersNumber+0], round(x[nColumn+0+x[nColumn+0+(clustersNumber*1)]],4)))####.sortByKey()
rdd_grouped = rdd_transformed.groupByKey().repartition(120)
rdd_thresholds = rdd_grouped.mapValues(outliers_modified_z_score)###.sortByKey()
rdd1_kv = rdd_thresholds.map(lambda x: (x[0], x[1]))
rdd2_kv = RddNew.map(lambda x: (x[nColumn+clustersNumber+0], x+[x[nColumn+0+x[nColumn+clustersNumber+0]]]))
joined_rdd = rdd2_kv.leftOuterJoin(rdd1_kv).repartition(120)
filtered_rdd = joined_rdd.filter(lambda x: x[1][0][len(x[1][0])-1] >= x[1][1])####.sortByKey()
grouped_rdd = filtered_rdd.map(lambda x: (x[0], x[1][0])).groupByKey().repartition(120)####.sortByKey()
grouped_rdd =filtered_rdd.map(lambda x: (x[0], x[1][0][0])).groupByKey().mapValues(list).repartition(120)


record_rdd22 = joined_rdd.filter(lambda x: x[1][0][len(x[1][0])-1] < x[1][1])
grouped_rdd22 = record_rdd22.map(lambda x: round((x[1][0][0],4))
rdd_thresholds22 = grouped_rdd22.mapValues(max_entropy_threshold)
rdd22_kv = rdd_thresholds22.map(lambda x: (x[0], x[1]))
joined_rdd22 = rdd22_kv.leftOuterJoin(rdd2_kv).repartition(120)
Filtred_record_rdd22  = joined_rdd22.filter(lambda x: x[1][0][len(x[1][0])-1] >= x[1][1])####.sortByKey()

from itertools import product
def cartesian_product0(lst):
    return list(product(lst, repeat=2))
def cartesian_product(record):
    key, values = record
    return [(key, (x, y)) for x in values for y in values if x < y]

cartesian_rdd = grouped_rdd.flatMap(cartesian_product)
result_rdd = cartesian_rdd#.filter(lambda x: x[1][0] < x[1][1])
pairstomatch = result_rdd.map(lambda x: x[1])

nowthresholdt1_1 = datetime.now()
nowthresholdt1_2 = datetime.now()

print("*******************************************************")
print("      End of the phase")
print("*******************************************************")
from datetime import datetime
nowt12 = datetime.now()
print("Finish time :", nowt12.strftime("%Y-%m-%d %H:%M:%S"))
print("**************************************************************")
print("")

print(" ******************************************************")
print(" ******************************************************")
print("                      ProbFilt                    ")
print(" ******************************************************")
print(" ******************************************************")
from datetime import datetime
nowt31 = datetime.now()
print("Start time :", nowt31.strftime("%Y-%m-%d %H:%M:%S"))
print("**************************************************************")
print("")

filtered_rdd_t3 = joined_rdd.filter(lambda x: x[1][0][len(x[1][0])-1] < x[1][1])####.sortByKey()
filtered_rdd_t3 = filtered_rdd_t3.filter(lambda x: x[1][0][len(x[1][0])-1] >0.1)####.sortByKey()

grouped_rdd_t3 = filtered_rdd_t3.map(lambda x: (x[0], x[1][0][:-1]))
def extract_key_values(record):
    group, values = record
    #item = (values[0], values[1:4], values[4:])
    item = (values[0], values[nColumn+0:nColumn+clustersNumber+0], values[nColumn+clustersNumber+0:])
    return (group, item)

transformed_rdd = grouped_rdd_t3.map(extract_key_values)
grouped_rdd = transformed_rdd.groupByKey().repartition(120)

def cartesian_products(group_items):
    from itertools import product
    group, items = group_items

    items_list = list(items)
    results = []

    for (item1, vals1, attrs1), (item2, vals2, attrs2) in product(items_list, repeat=2):
#        if  item1 < item2:
#            product_sum = sum(v1 * v2 for v1, v2 in zip(vals1, vals2))
#            results.append(((item1, item2), product_sum))
      if item1 < item2:
            # Calculer la somme des produits des probabilités uniquement si les attributs sont égaux
            product_sum = 0
            cpt = 0
            for i in range(clustersNumber):
                if attrs1[i] == attrs2[i]:
                    product_sum += vals1[i] * vals2[i]
                    cpt = cpt + 1

            if product_sum > 0:  # Ajouter au résultat seulement si le produit est non nul
                results.append((item1, item2, product_sum*cpt))
    return (group, results)

result_rdd = grouped_rdd.map(cartesian_products).repartition(120)
from datetime import datetime
now = datetime.now()
print("Finish time :", now.strftime("%Y-%m-%d %H:%M:%S"))
print(" ******************************************************")
from datetime import datetime
now = datetime.now()
rdd_transformed_t3 = result_rdd.flatMap(lambda x: [(x[0], round(y[2],4)) for y in x[1]])
rdd_grouped = rdd_transformed_t3.groupByKey().repartition(120)
## Evaluate Outlier M Zscore 
#rdd_thresholds = rdd_grouped.mapValues(max_entropy_threshold)
rdd_thresholds = rdd_grouped.mapValues(outliers_modified_z_score2)


rdd_transformed_t4 = result_rdd.flatMap(lambda x: [(x[0], [y[0],y[1],round(y[2],4)]) for y in x[1]])
joined_rdd = rdd_transformed_t4.leftOuterJoin(rdd_thresholds).repartition(120)
filtered_rdd = joined_rdd.filter(lambda x: x[1][0][len(x[1][0])-1] >= x[1][1]).repartition(120)####.sortByKey()
pairs = filtered_rdd.map(lambda x: (x[1][0][0], x[1][0][1]))
print(" ******************************************************")
print("       End of the phase                ")
print(" ******************************************************")
from datetime import datetime
nowt32 = datetime.now()
print("Finish time :", nowt32.strftime("%Y-%m-%d %H:%M:%S"))
print("**************************************************************")
print("")

print(" ******************************************************")
print(" ******************************************************")
print("                      Result                    ")
print(" ******************************************************")
print(" ******************************************************")
pairstomatch10 = pairstomatch_confirmed.union(pairs)


def normalize_pair(pair):
    return tuple(sorted(pair))

matched_pairs0=sparkcontxt.textFile('matched_pairs.csv')
matched_pairs1 = matched_pairs0.map(lambda y: y.split(",")).map(lambda x: (int(x[0]), int(x[1])))
rdd1 = pairstomatch10.map(lambda x: normalize_pair(x))
rdd2 = matched_pairs1.map(lambda x: normalize_pair(x))

duplicate_pairs = rdd1.intersection(rdd2)
print("Number of real matched pairs-Bechmark: ")
print(matched_pairs1.count())
print("Number of records pairs generated: ")
print(pairstomatch10.count())
print("Number of duplicate pairs detected : ")
print(duplicate_pairs.count())
size_matched_pairs = matched_pairs1.count()
size_generated = pairstomatch10.count()
size_PC_dataset = (sizeData*(sizeData-1))/2
nbr_detected = matched_pairs1.count()

print("PC: ")
pc = nbr_detected/size_matched_pairs
print(pc)
print("RR: ")
rr=size_generated/size_PC_dataset
print(rr)
print("FRR : ")
frr=(2*rr*pc)/(rr+pc)
print("PQ : ")
pq=nbr_detected/size_generated
print(pq)
results = [
    ["Metric", "Value"],
    ["PC", pc],
    ["RR", rr],
    ["FRR", frr],
    ["PQ", pq]
]
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

def normalize_pair(pair):
    return tuple(sorted(pair))

spark.stop()

