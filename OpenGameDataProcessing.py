#############################################################################
###        Data pre-processing of Waddle VR from Open Game Data           ### 
###                                                                       ###
###    -Pre-processing for being fed into a 1D CNN model
###    -Extracts user rotations from each session, aggregates and         ###
####    calculates dot product                                            ###
####   -Creates labels for the processed dataset based on timestamp       ###
#############################################################################

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_aug= pd.read_csv("C:\\Users\\Patron\\Downloads\\PENGUINS_20230801_to_20230831_418a972_all-events\\PENGUINS_20230801_to_20230831\\PENGUINS_20230801_to_20230831_418a972_all-events.tsv", sep="\t")
print(len(df_aug))
print("unique number of sessions in df_aug: ", df_aug['session_id'].nunique())
df_sept = pd.read_csv("C:\\Users\\Patron\\Downloads\\sept_data_duplicate\\PENGUINS_20230901_to_20230930\\PENGUINS_20230901_to_20230930_5cb9496_events.tsv", sep = "\t")
print(len(df_sept))
print("unqiue number of sessions in df_sept: ", df_sept['session_id'].nunique())
df_nov = pd.read_csv("C:\\Users\\Patron\\Downloads\\OpenGameDataOctober\\PENGUINS_20231101_to_20231130\\PENGUINS_20231101_to_20231130_481f8ea_events.tsv", sep = "\t")
print(len(df_nov))
print("unqiue number of sessions in df_nov: ", df_nov['session_id'].nunique())
df_oct = pd.read_csv("C:\\Users\\Patron\\Downloads\\OpenGameDataNovember\\PENGUINS_20231001_to_20231031\\PENGUINS_20231001_to_20231031_30abaa2_events.tsv", sep = "\t")
print(len(df_oct))
print("unqiue number of sessions in df_oct: ", df_oct['session_id'].nunique())

df = df_aug._append(df_sept, ignore_index=True)
df = df._append(df_oct, ignore_index=True)
df = df._append(df_nov, ignore_index=True)
print(len(df))
print("unique number of sessions in df: ", df['session_id'].nunique())
pd.set_option('display.max_rows', None)
print('Head of df:\n', df[0:5])

## Dropping cloumns that aren't needed ##
df = df.drop(["app_id", "event_source", "app_version", "app_branch", "log_version",	"offset", "user_id", "user_data", "game_state",	"index"], axis = 1)
# print('Head of df after dropping columns:\n', df[0:5])

## droping rows where the event_name column does not have viewport data ##
condition = ~(df["event_name"] == "viewport_data")
df.drop(df[condition].index, inplace=True)
df = df.reset_index(drop = True)
df = df.drop(["event_name"], axis = 1)
# print('df before grouping:\n', df[0:5])

### Extracts each rotation quaternion from position-rotation pair in gaze data and appends them into an array ###
def transform_event_data(entry):
    gazeDataPackage = []
    entryDict = json.loads(entry)
    gazeData = entryDict.get("gaze_data_package")
    pairs = json.loads(gazeData)
    for posNrot in pairs:
        rotation = posNrot["rot"]
        gazeDataPackage.append(np.array(rotation))
    #gazeDataPackageFlattened = np.array(gazeDataPackage).flatten().astype(float)
    # dotProducts = []
    # for i, _ in enumerate(gazeDataPackage[:-1]):
    #     dotProducts.append(np.dot(gazeDataPackage[i], gazeDataPackage[i + 1]))
    return gazeDataPackage

### Calculates dot products of each rotation with the next one in the aggregated rotations array ###
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

## Calculates time difference between each row of same session_id ##

df['timestamp'] = pd.to_datetime(df['timestamp'], errors = 'coerce')
# print(len(df['timestamp']))
df = df.dropna(subset = ['timestamp'])
# print(len(df['timestamp']))
df['time_duration'] = df.groupby('session_id')['timestamp'].diff().dt.total_seconds()
# print(len(df['timestamp']))
# print(len(df))
# print(df['time_duration'])
df['time_duration'] = df['time_duration'].fillna(0)

df['event_data'] = df['event_data'].apply(transform_event_data)
# print(df.head())

## Aggregates data resulting in 1 row for each session_id with all rotations of the session appended in order ##

df['event_data'] = df['event_data'].apply(lambda x: [np.array(rotation) for rotation in x])
df = df.groupby('session_id').agg({ 'event_data': lambda x: sum(x, []), 'time_duration': 'sum' }).reset_index()
print(df['time_duration'])

print(df.head())
lengths = df.loc[:453, 'event_data'].apply(len)
# print(lengths)

df['event_data'] = df['event_data'].apply(dot_product_rotations)

## Debug statements ##

print(df.head())
print(len(df))
print(df['session_id'].nunique())
# lengths = df.loc[:453, 'event_data'].apply(len)
# print(lengths)
# print(df.shape[1])
# print(df.dtypes)

## Filters and modifies length of event_data of each session according to input requirment for CNN model ##

df = df[df['time_duration'] >= 42].reset_index() # Filter out sessions that are less than n minutes long
# print(df['time_duration'])
lengths = df.loc[:len(df)-1, 'event_data'].apply(len)
# print(lengths)
df['event_data'] = df['event_data'].apply(lambda arr: arr[:1500]) # Slice each session's event_data to have only the first n dot products
print(len(df))
df = df[df['event_data'].apply(len) >= 1500] 
print('after filtering out entries with less than 1500 rotations: ', len(df))
lengths = df.loc[:len(df)-1, 'event_data'].apply(len)
# print(lengths)

train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 1)
train_df = train_df.reset_index(drop = True)
test_df = test_df.reset_index(drop = True)
print('train_df length: ', len(train_df))
train_df['label'] = train_df['time_duration'].apply(lambda x: 'quit' if x < 60 else 'continue') 
test_df['label'] = test_df['time_duration'].apply(lambda x: 'quit' if x < 60 else 'continue')
print(train_df['label'])
print(f'test')
print(test_df['label'])
### Save to csv file ###
#df.to_csv('open_game_data_rotations.csv', index = False)
train_df.to_csv('C:\\Users\\Patron\\Documents\\open_game_data_training.csv', index = False)
test_df.to_csv('C:\\Users\\Patron\\Documents\\open_game_data_testing.csv', index = False)