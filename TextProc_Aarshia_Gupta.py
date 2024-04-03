#!/usr/bin/env python
# coding: utf-8

# # Assignment 02
# 
# This is the Text Processing project.
# 
# See Canvas for its deadline. 

# In[ ]:


# import packages
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import re
from urllib.parse import urlparse
import urllib.robotparser
from bs4 import BeautifulSoup

# This code checks the robots.txt file
def canFetch(url):

    parsed_uri = urlparse(url)
    domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(domain + "/robots.txt")
    try:
        rp.read()
        canFetchBool = rp.can_fetch("*", url)
    except:
        canFetchBool = None
    
    return canFetchBool


# ## Assignment 02 examples
# 
# #### Example 0 of project for assignment 02: Text processing
# 
# 1) perform the analysis of data/ira.csv (similarly to what done in processing_text.ipynb, including most frequent words etc.)
# 
# 2) perform sentiment analysis on this dataset
# 
# 3) detail comments, explain step by step what is happening, and try to write down a paragraph or two at the end discussing what you figured out
# 
# ************

# # Decoding Voter Sentiments
# 
# 
# In anticipation of the upcoming presidential election scheduled for November of this year, this project aims to conduct sentiment analysis on a comprehensive dataset comprising tweets from American citizens. With a specific focus on evaluating public sentiment towards Donald Trump, who held office for the past four years before Joe Biden, the analysis provides valuable insights into the current political landscape. The dataset utilized in this project has been sourced from the Internet Research Agency, offering a rich and diverse collection of opinions and perspectives

# ## The Main Question
# 
# This project seeks to understand the prevailing sentiments of American voters, particularly towards Donald Trump.
# Are these sentiments predominantly positive or negative?

# ## First Glance: Exploring the Data

# In[ ]:


# Taking a look at all the data by sorting through it to attain a list of lists
ira_tweets = [x.strip() for x in open("data/ira.csv", encoding='utf8').readlines()]
ira_tweets


# In[68]:


# Finding out how many tweets in total have been uncovered by the Internet Research Agency
def count_ira(x):
    count = 0
    for i in x:
        count+=1
    return count

count_ira(ira_tweets)


# #### There are 90,000 tweets collected by the IRA that were analysed to make a prediction about Trump's standing in the elections.

# In[69]:


# Computing the length of the shortest tweet found by IRA
def shortest_ira_tweet(lst):
    least_len = float('inf')
    for i in lst:
        if len(i) < least_len:
            least_len = len(i)
    return least_len 

shortest_ira_tweet(ira_tweets)


# #### The shortest  tweet is 43 characters. 

# In[70]:


# Computing the length of the longest tweet found by IRA
def longest_ira_tweet(lst):
    most_len = 0
    for i in lst:
        if len(i) > most_len:
            most_len = len(i)
    return most_len 

longest_ira_tweet(ira_tweets)


# #### The longest tweet is 305 characters. 

# ## Cleaning & Filtering the Data

# Every tweet in ira_tweets is preceded by a number code, account name and date+time stamp.
# This is irrelevant in our analysis of solely the text content in the tweets itself.

# In[6]:


#Sorting through data to attain a list of lists, where each item related to a tweet is in the corresponding sublist.
tweets_all = [x.strip().split(',') for x in open("data/ira.csv").readlines()]
tweets_all


# In[7]:


# Filtering the data to get a list of only tweets, thereby eliminating the unrequired information.
tweets_only = [x[3] for x in tweets_all]
tweets_only


# In[8]:


# Filtering the IRA tweets further to attain the ones that talk about Trump, former President of the United States.
def in_text(y):
    return 'Trump' in y
trump_tweets = list(filter(in_text, tweets_only))
trump_tweets


# In[9]:


len(trump_tweets)


# #### 6276 of the original 90,000 tweets uncovered by the IRA directly reference Trump and are related to him. 

# In[10]:


# Creating a DataFrame with just the filtered tweets.
trump_df = pd.DataFrame().assign(Tweets=trump_tweets)
trump_df


# # Counting Words

# Finding the most frequently used words, using tokenizing
# 

# In[11]:


# Made a single long string with the tweet text, and split the tweets into a list of only words.
all_tweets_text = " ".join(trump_tweets)
words_list = all_tweets_text.split()
print("First 20 words:", words_list[:20])


# In[12]:


# Checking total number of words, and distinct words used in the tweets. 
total_words = len(words_list)
distinct_words = set(words_list)
num_distinct_words = len(distinct_words)

print("Total words:", total_words)
print("Number of distinct words:", num_distinct_words)


# #### There is a total of 87407 words, out of which 24,249 are unique.

# In[13]:


# Removing 'stop' words like 'a', 'the', 'in' that are not helpful in our analysis.

# Removed short words (less than three characters)
filtered_words = [word for word in words_list if len(word) >= 3]

# Calculated the total number of words after filtering
total_filtered_words = len(filtered_words)

# Calculated the number of distinct words after filtering using a set
distinct_filtered_words = set(filtered_words)
num_distinct_filtered_words = len(distinct_filtered_words)

# Print the results
print("Total words after removing short words:", total_filtered_words)
print("Number of distinct words after removing short words:", num_distinct_filtered_words)


# #### After eliminating 'stop' words that hindered the sentiment analysis, we were left with 72,702 words, out of which 23,694 are unique.

# # Counting Word Frequency

# In[14]:


# Created a categorical distribution using dictionary.

categorical_distribution = {}
for word in words_list:
    if word in categorical_distribution:
        categorical_distribution[word] += 1
    else:
        categorical_distribution[word] = 1

# Printed the categorical distribution
print(categorical_distribution)


# # Tokenizing again (using NLTK)

# In[55]:


from nltk import tokenize
import nltk
nltk.download('punkt')


# In[56]:


allText = all_tweets_text # pass in a string consisting of all tweets

wordList = tokenize.word_tokenize(allText)
len(wordList)


# # Counting again

# In[57]:


# Removed short words
filtered_words = [word for word in wordList if len(word) >= 3]

# Created a categorical distribution using a dictionary for filtered words
categorical_distribution_filtered = {}
for word in filtered_words:
    if word in categorical_distribution_filtered:
        categorical_distribution_filtered[word] += 1
    else:
        categorical_distribution_filtered[word] = 1

# Printed the categorical distribution for filtered words
print(categorical_distribution_filtered)


# # Sentiment with NLTK

# In[58]:


nltk.download('vader_lexicon')


# In[59]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[60]:


sid = SentimentIntensityAnalyzer()
sid.polarity_scores("Good test!")


# In[61]:


tweetSentiments = []

for tweet in trump_tweets:
    tweetSentiment = sid.polarity_scores(tweet)
    tweetSentiment['text'] = tweet
    tweetSentiments.append(tweetSentiment)
tweetSentiments  


# In[62]:


tweetSentimentDf = pd.DataFrame(tweetSentiments)


# In[63]:


tweetSentimentDf.sort_values('compound')


# The dataframe above represents the negative, positive and neutral quotient for each of the filtered tweets.

# In[64]:


# Calculating total proportion of negative words from the filtered tweets.
neg_sum = tweetSentimentDf['neg'].sum()
neg_prop = neg_sum/tweetSentimentDf.shape[0]

print("Negativity quotient:", neg_prop)


# In[65]:


# Calculating total proportion of positive words from the filtered tweets.
pos_sum = tweetSentimentDf['pos'].sum()
pos_prop = pos_sum/tweetSentimentDf.shape[0]

print("Positivity quotient:", pos_prop)


# In[66]:


# Calculating total proportion of neutral words from the filtered tweets.
neu_sum = tweetSentimentDf['neu'].sum()
neu_prop = neu_sum/tweetSentimentDf.shape[0]

print("Neutral quotient:", neu_prop)


# ## Conclusion
# 
# The outcome of the sentiment analysis yielded a nuanced perspective, revealing a somewhat inconclusive sentiment distribution. Approximately 83.65 percent of the analyzed tweets exhibited a neutral stance towards Donald Trump. A discernible polarization was observed among the remaining 16.35 percent, with 7 percent expressing a negative sentiment, reflecting disapproval or discontent with Trump, while 8 percent conveyed a positive sentiment, indicating support or admiration for the former president. This breakdown illustrates the complex landscape of public sentiment, suggesting a considerable prevalence of neutrality alongside discernible expressions of both criticism and endorsement.
