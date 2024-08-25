#############################################################################
###  1D CNN model to learn patterns in user head rotations and quitting   ### 
###                                                                       ###
###    - Supervised model for predicting quit or continue                 ###
###    - Unsupervised model can be used for learnt pattern extraction     ###      
#############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import ast
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

## Load the preprocessed training and testing datasets ##
df_train = pd.read_csv('C:\\Users\\Patron\\Documents\\open_game_data_training.csv')
df_test = pd.read_csv('C:\\Users\\Patron\\Documents\\open_game_data_testing.csv')
print('number of quits in test_df: ', (df_test['label'] == 'quit').sum())
print('total entries in df_test: ', len(df_test))

device = torch.device('cpu')

# Hyperparameters
random_seed = 1
# num_epochs = 10
input_size = len(df_train)
batch_size = 60
sequence_length = 40

# Convert event data from string representation of lists to actual lists
def convert_to_float_list(row):
  return ast.literal_eval(row)

df_train['event_data'] = df_train['event_data'].apply(convert_to_float_list)
df_test['event_data'] = df_test['event_data'].apply(convert_to_float_list)
print(df_train.head())

### DataLoader class for unsupervised model ###
class rotationsDataLoader(Dataset):       # For unsupervised
   def __init__(self, df, sequence_length):
      self.df = torch.tensor(df, dtype = torch.float32)
      self.sequence_length = sequence_length

   def __len__(self):
        return self.df.__len__() - (self.sequence_length - 1)
    
   def __getitem__(self, index):
        features = self.df[index:index + self.sequence_length].T
        return features

### DataLoader class for supervised model ###
class SupervisedRotationsDataLoader(Dataset): # For supervised
   def __init__(self, df, sequence_length, labels):
      self.df = torch.tensor(df, dtype = torch.float32)
      self.sequence_length = sequence_length
      self.labels = labels

   def __len__(self):
        return self.df.__len__() - (self.sequence_length - 1)
    
   def __getitem__(self, index):
        features = self.df[index:index + self.sequence_length].T
        labels = self.labels[index]
        return features, labels
   
# dataset = rotationsDataLoader(df['event_data'], sequence_length = sequence_length)
# loader = DataLoader(dataset, shuffle = True, batch_size = batch_size)

## One-hot encode the labels for the supervised learning task ##
label_encoder = LabelEncoder()
df_train['label_encoded'] = label_encoder.fit_transform(df_train['label'])
labels_onehot = torch.nn.functional.one_hot(torch.tensor(df_train['label_encoded'], dtype=torch.long), num_classes = 2)

df_test['label_encoded'] = label_encoder.fit_transform(df_test['label'])
labels_onehot_test = torch.nn.functional.one_hot(torch.tensor(df_test['label_encoded'], dtype=torch.long), num_classes = 2)

# Create DataLoaders for supervised learning
supervised_dataset = SupervisedRotationsDataLoader(df_train['event_data'], labels = labels_onehot, sequence_length = sequence_length)
supervised_loader = DataLoader(supervised_dataset, shuffle=True, batch_size = batch_size)

testing_dataset = SupervisedRotationsDataLoader(df_test['event_data'], labels = labels_onehot_test, sequence_length = sequence_length)
testing_loader = DataLoader(testing_dataset, shuffle=True, batch_size = batch_size)

# Count the total number of samples in the DataLoader
count = 0
for batch in supervised_loader:
    inputs, labels = batch
    count += len(inputs)
print(count)

### Unsupervised CNN Architecture ###
# class unsupervised1DCNN(nn.Module):
#    def __init__(self):
#       super(unsupervised1DCNN, self).__init__()
#       self.conv1 = nn.Conv1d(in_channels = 600, out_channels = 6, kernel_size = 2)
#       # self.conv2 = nn.Conv1d(in_channels = 4, out_channels = 4, kernel_size = 2)
#       # self.dropout = nn.Dropout(0.5)
#       self.pool = nn.MaxPool1d(kernel_size = 2)
#       # self.flatten = nn.Flatten()
#       # self.fc1 = nn.Linear(64, 100)
    
#    def forward(self, x):
#       #forward pass through all the layers
#       x = F.relu(self.conv1(x))
#       # x = F.relu(self.conv2(x))
#       # x = self.dropout(x)
#       x = self.pool(x)
#       # x = self.flatten(x)
#       # print(x.shape)
#       # x = F.relu(self.fc1(x))
#       return x
   

### Supervised CNN Architecture ###
class Supervised1DCNN(nn.Module):
   def __init__(self):
      super(Supervised1DCNN, self).__init__()
      self.conv1 = nn.Conv1d(in_channels = 1500, out_channels = 16, kernel_size = 3)
      self.pool = nn.MaxPool1d(kernel_size = 2)
      self.conv2 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 2)
      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(18, 300)
      self.fc2 = nn.Linear(300, 2)
    
   def forward(self, x):
      # Forward pass through all the layers
      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x = F.relu(self.conv2(x))
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x

### Training the Supervised Model ###
model = Supervised1DCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
num_epochs = 50
print_freq = 5
num_batches = len(df_train) // batch_size

for epoch in range(num_epochs):
   running_loss = 0.0
   for i, data in enumerate(supervised_loader, 0):
      inputs, labels = data
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % print_freq == print_freq - 1:
         print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{num_batches}], Loss: {running_loss / print_freq:.4f}')
         running_loss = 0.0
   print('Finished Training')

### Testing the Supervised Model ###
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testing_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Get predictions
        print(outputs.shape)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (torch.argmax(predicted, 1) == labels.argmax(dim=1)).sum().item()

# Compute and display accuracy
accuracy = 100 * correct / total
print(f"Test Loss: {test_loss / len(testing_loader):.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")

## For unsupervised model

# torch.manual_seed(random_seed)
# model = unsupervised1DCNN()
# model = model.to(device)

# model.eval()

# count = 0
# for data in loader:
#    inputs = data
#    inputs = inputs.to(device)

#    with torch.no_grad():
#       features = model(inputs)
#       count = count + 1
#       # print(features)
#       print(features.shape)
# print(count)

# flattened_features = [tensor.view(-1).cpu().numpy() for tensor in features]

# print(flattened_features[0])
# print(len(flattened_features))

## Using PCA to visualize the learnt features from unsupervised model ##

# pca = PCA(n_components = 3)
# components = pca.fit_transform(flattened_features)

# data = {
#     'PC1': components[:, 0],
#     'PC2': components[:, 1],
#     'PC3': components[:, 2]
# }

# fig = px.scatter_3d(data, x = 'PC1', y = 'PC2', z = 'PC3')
# fig.show()

# kmeans = KMeans(n_clusters=5)
# kmeans.fit(flattened_features)
# clusters = kmeans.predict(data)

# data2 = {
#    'Cluster': clusters
# }
# fig2 = px.scatter(data2, x='Component 1', y='Component 2')


