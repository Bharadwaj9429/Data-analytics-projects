# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 21:22:12 2021

@author: BHARADWAJ
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D:/tinku/outstanding/netflix_titles.csv")

df.head(10)

print("shape:",df.shape)
print(df.columns)

dict = {}
for i in list (df.columns):
    dict[i]=df[i].value_counts().shape[0]
print(pd.DataFrame(dict,index = ["unique count"]).transpose())  

# Identify the unique values and missing values 
dict = {}
for i in list(df.columns):
    dict[i] = df[i].value_counts().shape[0]
    
print(pd.DataFrame(dict,index = ["unique count"]).transpose())

print('Table of missing values: ')
print(df.isnull().sum())

# top 10 country count 
Netflix_top_country = df['country'].value_counts().head(10)

df2 = pd.DataFrame(Netflix_top_country, columns= ['country'])

print(df2)

#last ten years of netflix

Last_ten_years = df[df['release_year']>2010]
Last_ten_years. head()

#look at the count of type, rating and country

fig = plt.figure(figsize = (20,20))
gs = fig.add_gridspec(2,2)
gs.update(wspace=0.3, hspace=0.3)

sns.set(style="darkgrid")
ax0=fig.add_subplot(gs[0,0])
ax1=fig.add_subplot(gs[0,1])
ax2=fig.add_subplot(gs[1,0])
ax3=fig.add_subplot(gs[1,1])

#set titles and lables
ax0.set_title("Tv_shows vs Movies")
ax1.set_title("Distribution of ratings")
ax2.set_title("Distribution of country")
ax3.set_title("Distribution of release year")

ax1.set_xticklabels(labels=[], rotation= 90)
ax2.set_xticklabels(labels=[], rotation= 90)

#construction subplot

sns.countplot(ax = ax0, x ="type", data = df, palette="Set2")
sns.countplot(ax = ax1, x = "rating" , hue = "type", data= df)
sns.countplot(ax = ax2, x = "country" , hue = "type", data= df, order= df.country.value_counts().iloc[:10].index)
sns.countplot(ax = ax3, x = "release_year" , hue = "type", data = Last_ten_years)
plt.show()

df['description'].head()

#TFidfvectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#define TFidfVectorizer object. remove all stop words english such as 'the', 'a'

Tfidf= TfidfVectorizer(stop_words = 'english')

#remove NAN with empty string
df['description']=df['description'].fillna('')

#construct the TFIDF matrix by fitting and trasnforming the data
Tfidf_matrix = Tfidf.fit_transform(df['description'])

#output the matrix

Tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

#compute the cosine similarity matrix 

cosine_sim= linear_kernel(Tfidf_matrix,Tfidf_matrix)

#Construct the reverse map of indices and movie titles

indices = pd.Series(df.index,index=df['title']).drop_duplicates()
indices.head()

def get_recommendations(title, cosine_sim = cosine_sim):
    #get index of the matching title
    idx=indices[title]
    #get the similarity score of the similar titles
    sim_scores=list(enumerate(cosine_sim[idx]))
    #sort the movies based on the similarity score
    sim_scores=sorted(sim_scores, key=lambda x:x[1], reverse=True)
    #get the similarity score of top 10 movies
    sim_scores=sim_scores[1:11]
    #get the indices 
    movie_indices = [i[0] for i in sim_scores]
    #return the top indices
    return df['title'].iloc[movie_indices]

get_recommendations('Supernatural')

######### Then we can Say that we have these many predictions on Netflix_titles over analysis
