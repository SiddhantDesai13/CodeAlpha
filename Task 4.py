#!/usr/bin/env python
# coding: utf-8

# # Data Collection

# In[ ]:


import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt
import re

# Add your Twitter API credentials
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Function to fetch tweets
def get_tweets(query, count=100):
    tweets = []
    try:
        fetched_tweets = tweepy.Cursor(api.search, q=query, tweet_mode='extended').items(count)
        for tweet in fetched_tweets:
            parsed_tweet = {
                'text': tweet.full_text,
                'created_at': tweet.created_at,
                'username': tweet.user.screen_name
            }
            tweets.append(parsed_tweet)
        return tweets
    except tweepy.TweepError as e:
        print("Error : " + str(e))


# # Data Preprocessing

# In[ ]:


# Clean tweet text
def clean_tweet(tweet):
    cleaned_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    return cleaned_tweet


# # Sentiment Analysis

# In[ ]:


def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Fetch and process tweets
def fetch_and_process_tweets(query, count=100):
    tweets = get_tweets(query, count)
    if tweets:
        for tweet in tweets:
            tweet['cleaned_text'] = clean_tweet(tweet['text'])
            tweet['sentiment'] = analyze_sentiment(tweet['cleaned_text'])
        return tweets
    else:
        return []


# # Data Visualization

# In[ ]:


def visualize_sentiment(tweets):
    sentiments = [tweet['sentiment'] for tweet in tweets]
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}

    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['gold', 'lightcoral', 'lightskyblue']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Sentiment Analysis of Tweets')
    plt.show()


# In[ ]:


# Example usage
query = "Python programming"
tweets = fetch_and_process_tweets(query, count=100)
if tweets:
    visualize_sentiment(tweets)
else:
    print("No tweets found.")


# In[ ]:




