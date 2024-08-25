############################################################################################################
###        Data pre-processing of Waddle VR from Open Game Data                                          ### 
###                                                                                                      ###
###    -Pre-processing for being fed into a 1D CNN model                                                 ###
###         - Find the dataset at: https://opengamedata.fielddaylab.wisc.edu/gamedata.php?game=PENGUIN   ###
###    -Extracts user rotations from each session, aggregates and                                        ###
####    calculates dot product                                                                           ###
####   -Creates labels for the processed dataset based on timestamp                                      ###
############################################################################################################

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

## Load the raw data from multiple months into separate DataFrames ##
df_aug= pd.read_csv("C:\\Users\\Patron\\Downloads\\PENGUINS_20230801_to_20230831_418a972_all-events\\PENGUINS_20230801_to_20230831\\PENGUINS_20230801_to_20230831_418a972_all-events.tsv", sep="\t")
print(len(df_aug))
print("unique number of sessions in August: ", df_aug['session_id'].nunique())

df_sept = pd.read_csv("C:\\Users\\Patron\\Downloads\\sept_data_duplicate\\PENGUINS_20230901_to_20230930\\PENGUINS_20230901_to_20230930_5cb9496_events.tsv", sep = "\t")
print(len(df_sept))
print("unqiue number of sessions in September: ", df_sept['session_id'].nunique())

df_nov = pd.read_csv("C:\\Users\\Patron\\Downloads\\OpenGameDataOctober\\PENGUINS_20231101_to_20231130\\PENGUINS_20231101_to_20231130_481f8ea_events.tsv", sep = "\t")
print(len(df_nov))
print("unqiue number of sessions in November: ", df_nov['session_id'].nunique())

df_oct = pd.read_csv("C:\\Users\\Patron\\Downloads\\OpenGameDataNovember\\PENGUINS_20231001_to_20231031\\PENGUINS_20231001_to_20231031_30abaa2_events.tsv", sep = "\t")
print(len(df_oct))
print("unqiue number of sessions in October: ", df_oct['session_id'].nunique())

# Combine all the monthly DataFrames into a single DataFrame ##
df = df_aug._append(df_sept, ignore_index=True)
df = df._append(df_oct, ignore_index=True)
df = df._append(df_nov, ignore_index=True)
print(len(df))
print("unique number of sessions in df: ", df['session_id'].nunique())
pd.set_option('display.max_rows', None)
# print('Head of df:\n', df[0:5])

## Dropping unnecessary columns from the DataFrame ##
df = df.drop(["app_id", "event_source", "app_version", "app_branch", "log_version",	"offset", "user_id", "user_data", "game_state",	"index"], axis = 1)
# print('Head of df after dropping columns:\n', df[0:5])

## Filter the DataFrame to only include rows where 'event_name' is ##
## 'viewport_data' and then drop the 'event_name' column           ##
condition = ~(df["event_name"] == "viewport_data")
df.drop(df[condition].index, inplace=True)
df = df.reset_index(drop = True)
df = df.drop(["event_name"], axis = 1)
# print('df before grouping:\n', df[0:5])

### Function to extract rotation data from the 'event_data' JSON field. ###
def transform_event_data(entry):
    gazeDataPackage = []
    entryDict = json.loads(entry)
    gazeData = entryDict.get("gaze_data_package")
    pairs = json.loads(gazeData)
    for posNrot in pairs:
        rotation = posNrot["rot"]
        gazeDataPackage.append(np.array(rotation))
    return gazeDataPackage

### Function to calculate dot products between consecutive rotations. ###
count = -1
def dot_product_rotations(entry):
    global count
    count += 1
    if len(entry) < 2:
        return None
    
    dotProducts = []
    for i, _ in enumerate(entry[:-1]):
        try:
            if len(entry[i]) != len(entry[i + 1]):
                return None
            dotProducts.append(np.dot(entry[i], entry[i + 1]))
        except ValueError as e:
            print("Value error at index " + str(count))
    
    
    return dotProducts

## Convert 'timestamp' to datetime, calculate time differences between ##
## consecutive entries within each session                             ##
df['timestamp'] = pd.to_datetime(df['timestamp'], errors = 'coerce')
df = df.dropna(subset = ['timestamp'])
df['time_duration'] = df.groupby('session_id')['timestamp'].diff().dt.total_seconds()
df['time_duration'] = df['time_duration'].fillna(0)

## Apply the transformation function to extract rotation data from the 'event_data' field. ##
df['event_data'] = df['event_data'].apply(transform_event_data)

## Aggregate rotation data and total session duration for each session_id. ##
df['event_data'] = df['event_data'].apply(lambda x: [np.array(rotation) for rotation in x])
df = df.groupby('session_id').agg({ 'event_data': lambda x: sum(x, []), 'time_duration': 'sum' }).reset_index()

## Calculate dot products of consecutive rotations for each session.            ##
## Dot product of all rotations would give us a resultant rotation which can be ##
## relativized with the rotations of other data entries                         ##
df['event_data'] = df['event_data'].apply(dot_product_rotations)

## Filter sessions based on the duration and ensure each session has a minimum of x rotations. ##
df = df[df['time_duration'] >= 42].reset_index() 
lengths = df.loc[:len(df)-1, 'event_data'].apply(len)
df['event_data'] = df['event_data'].apply(lambda arr: arr[:1500]) # Truncate to first n dot products
df = df[df['event_data'].apply(len) >= 1500]

train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 1)
train_df = train_df.reset_index(drop = True)
test_df = test_df.reset_index(drop = True)

## Split the data into training and testing sets, and create labels based on session duration. ##
train_df['label'] = train_df['time_duration'].apply(lambda x: 'quit' if x < 60 else 'continue') 
test_df['label'] = test_df['time_duration'].apply(lambda x: 'quit' if x < 60 else 'continue')

## Save the processed training and testing datasets to CSV files. ##
train_df.to_csv('C:\\Users\\Patron\\Documents\\open_game_data_training.csv', index = False)
test_df.to_csv('C:\\Users\\Patron\\Documents\\open_game_data_testing.csv', index = False)
