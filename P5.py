import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from numpy import double
import pandas as pd
import os.path 
from os import path
import datetime

nltk.download('twitter_samples')
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("Wow, NLTK is really powerful!"))
#tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]
from random import shuffle

def is_positive(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0
def is_neutral(tweet: str) -> bool:
    return sia.polarity_scores(tweet)["neu"] > .5
    
#shuffle(tweets)
#for tweet in tweets[:30]:
    #print(">", is_positive(tweet), tweet)

if(path.exists('Btweets_clean.csv')):
    Tweet_df = pd.read_csv('Btweets_clean.csv')
    print("Tweets loaded")
    print(Tweet_df.head())
else:
    Tweet_df = pd.read_csv('Bitcoin_tweets.csv',low_memory=False)
    print(Tweet_df.head())
    del Tweet_df["is_retweet"]
    del Tweet_df["source"]
    del Tweet_df["hashtags"]
    del Tweet_df["user_favourites"]
    del Tweet_df["user_friends"]
    #TODO: Loop through all rows and delete tweets that appear within
    #        2 days of an account being created
    del Tweet_df["user_created"]
    del Tweet_df["user_description"]
    del Tweet_df["user_name"]
    del Tweet_df["user_location"]
    print("\n",Tweet_df.head(),"\n")
    for col in Tweet_df.columns:
        print(col)
    Tweet_df.to_csv('Btweets_clean.csv',index=False)


