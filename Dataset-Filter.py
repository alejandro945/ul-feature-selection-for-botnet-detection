"""
The following script takes in a csv file and a file name containing column names to keep and filter out the rest from the dataset.
The dataset name and the filter list file names are passed as command line arguments.
Usage:
>>> python Dataset-Filter.py dataset_filename column_names_filename output_filename
 --> dataset_filename - The name of the dataset csv file to be reduced in dimension.
 --> column_names_filename - The file containing list of columns to filter our from dataset.
 --> output_filename - The output file name for resulting dataset.

"""

import sys
import pandas as pd
import numpy as np

def compute_columns_to_drop(df, column_names):
    total_instances = len(df)
    column_names_to_drop = []
    column_names.append(df.columns[-1]) # Add target label column to predictors to be retained.
    for col_name in df.columns:
        if(col_name not in column_names):
            column_names_to_drop.append(col_name)

    return column_names_to_drop

column_names = []

df = pd.read_csv(sys.argv[1])
print("\nOriginal data to be cleaned ...")
print(df.head(3))

with open(sys.argv[2]) as fname:
    for ln in fname:
    	column_names.append(ln.rstrip('\n'))

print(column_names)

col_drop_list = compute_columns_to_drop(df, column_names)
df = df.drop(col_drop_list,axis=1)

print("\nCleaned network data ...")
print(df.head(3))

df.to_csv(sys.argv[3], sep=',', index=False)