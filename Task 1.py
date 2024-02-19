#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")


# # Exploratory Data Analysis (EDA)

# In[2]:


data = pd.read_csv("C:\Siddhant\CodeAlpha ML 3 Months\Task 1\data\data.csv")
genre_data = pd.read_csv("C:\Siddhant\CodeAlpha ML 3 Months\Task 1\data\data_by_genres.csv")
year_data = pd.read_csv("C:\Siddhant\CodeAlpha ML 3 Months\Task 1\data\data_by_year.csv")


# In[3]:


data.info()


# In[4]:


genre_data.info()


# In[5]:


year_data.info()


# In[6]:


# Check for Missing Values
print(data.isnull().sum())


# In[7]:


# Selecting only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Correlation Matrix
correlation_matrix = numeric_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")

plt.title('Correlation Matrix')
plt.show()


# # Data Visualization

# 1) Occurences of Each Decade

# In[8]:


# Define a function to get the decade
def get_decade(year):
    return f"{int(year)//10*10}s"

# Create a new column 'decade' based on the 'year' column
data['decade'] = data['year'].apply(get_decade)

# Set the plot size
sns.set(rc={'figure.figsize':(11 ,6)})

# Plot the count of occurrences for each decade
sns.countplot(data=data, x='decade')

# Set plot title and labels
plt.title('Occurences of Each Decade')
plt.xlabel('Decade')
plt.ylabel('Count')

# Show plot
plt.show()


# In[9]:


sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=sound_features)
fig.show()


# 2) Characteristics of Different Genres

# In[10]:


top10_genres = genre_data.nlargest(10, 'popularity')

fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
fig.show()


# # K-Means Clustering of Genres

# In[11]:


cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('kmeans', KMeans(n_clusters=10))
])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)

# Get the cluster labels
cluster_labels = cluster_pipeline.named_steps['kmeans'].labels_

# Visualizing the Clusters with t-SNE
tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)

# Create a DataFrame for the t-SNE projection
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = cluster_labels  # Use the obtained cluster labels

# Plot the scatter plot with Plotly
fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['genres', 'cluster'],
    labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2', 'cluster': 'Cluster'},
    title='t-SNE Visualization of Clusters'
)

fig.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode='markers'))
fig.show()


# # K-Means Clustering of Songs

# In[12]:


song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('kmeans', KMeans(n_clusters=20))
])
X = data.select_dtypes(np.number)
song_cluster_pipeline.fit(X)

# Get the cluster labels
song_cluster_labels = song_cluster_pipeline.predict(X)

# Assign cluster labels to the original data
data['cluster_label'] = song_cluster_labels

# Visualizing the Clusters with PCA
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)

# Create a DataFrame for the PCA projection
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

# Plot the scatter plot with Plotly
fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['title', 'cluster'],
    labels={'x': 'PCA Dimension 1', 'y': 'PCA Dimension 2', 'cluster': 'Cluster'},
    title='PCA Visualization of Song Clusters'
)

fig.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode='markers'))
fig.show()


# # Recommendation System

# In[13]:


# Function to recommend songs
def recommend_songs(song_cluster_label, num_recommendations=5):
    similar_songs = data[data['cluster_label'] == song_cluster_label].sample(num_recommendations)
    return similar_songs

# Function to recommend genres based on popularity
def recommend_genres(num_recommendations=5):
    sampled_genres = genre_data.sample(num_recommendations)
    return sampled_genres

# Example usage:
song_cluster_label = 0
genre_cluster_label = 1

# Recommended Songs
recommended_songs = recommend_songs(song_cluster_label)
print("Recommended Songs:")
print(recommended_songs['name'])

# Visualization for Recommended Songs
plt.figure(figsize=(10, 6))
plt.barh(recommended_songs['name'], recommended_songs['popularity'], color='skyblue')
plt.xlabel('Popularity')
plt.ylabel('Songs')
plt.title('Recommended Songs')
plt.gca().invert_yaxis()
plt.show()

# Recommended Genres
recommended_genres = recommend_genres(5)
print("\nRecommended Genres:")
print(recommended_genres['genres'])

# Visualization for Recommended Genres
plt.figure(figsize=(10, 6))
sns.barplot(x='popularity', y='genres', data=recommended_genres, palette='viridis')
plt.xlabel('Popularity')
plt.ylabel('Genres')
plt.title('Recommended Genres')
plt.show()


# In[ ]:




