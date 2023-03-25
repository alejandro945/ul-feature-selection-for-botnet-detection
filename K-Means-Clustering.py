"""
The following script takes in a csv file and a file name containing column names to keep and filter out the rest from the dataset.
The dataset name and the filter list file names are passed as command line arguments.
Usage:
>>> python K-Means-Clustering.py predictors_score_file K predictors_start_ID number_of_columns
 --> predictors_score_file - The file containing feature scores for feature selection such as chi-sqaure score.
 --> K - number of clusters for K Means clustering algorithm.
 --> predictors_start_ID - Column ID for the starting column with score.
 --> number_of_columns - number of columns for K Means clustering to use.
 --> output filename for predictors identified based either on minimum or maximum score.
"""

import numpy as np
import pandas as pd
import sys

from sklearn.cluster import KMeans

data = pd.read_csv(sys.argv[1]) #'Values_to_Keep_Drop_v3.csv'
k = int(sys.argv[2]) # Assign K for K-Means clustering algorithm
predictor_column_ID = int(sys.argv[3]) # Assign Initial predictor column ID
column_predictors = int(sys.argv[4]) # Assign number of columns containing predictor scores (predictors)

x = data.iloc[:,predictor_column_ID:predictor_column_ID+column_predictors] # Extract Feature estimator score, this could be multiple rows (for example data.iloc[:,2:4] assuming there are two columns predictors: column 2 and 3)

kmeans = KMeans(k)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
PREDICTOR_SCORE_COLUMN = 2
PREDICTOR_NAME_COLUMN = 0
i = 0

clusters = [[] for j in range(k)]
min_value = data.iloc[i,PREDICTOR_SCORE_COLUMN]
min_cluster = 0
for item in identified_clusters:
    clusters[item].append(data.iloc[i,PREDICTOR_NAME_COLUMN])
    if(data.iloc[i,PREDICTOR_SCORE_COLUMN]<min_value):
        min_cluster = item
        min_value = data.iloc[i,PREDICTOR_SCORE_COLUMN]
    i += 1

#Print the best cluster predictors identified.    
print('The column names identified are:')
#print(clusters)
out_file = open(sys.argv[5], 'w')
for ln in clusters[min_cluster]:
	print(ln)
	out_file.write(ln+'\n')

out_file.close()