from operator import index
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from numpy import double
import pandas as pd
import os.path 
from os import path
import datetime
import tweepy
import re
from sklearn.utils import shuffle
import math
import emoji

nltk.download('twitter_samples')
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("Wow, NLTK is really powerful!"))
#tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]
from random import shuffle
def cleanText(line):
    line = re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", line)
    line = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", line).split())
    #line = re.sub(r"@")
    line = line.lower()
    line = line.strip()
    #TODO: replace contradictions
    for letters in line:
        if letters in """[]!.,"-!—@;$ç€:#$%^&*()+/?\n1234567890…\\""":
            line = line.replace(letters, " ")
    #TODO: put e's around the emojis so they don't get mixed up with the user words
    line = emoji.demojize(line)
    line = line.replace(":"," ")
    line = ' '.join(line.split())
    return line

def is_positive(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0
def is_neutral(tweet: str) -> bool:
    return sia.polarity_scores(tweet)["neu"] > .75
def how_neutral(tweet: str) -> double:
    return sia.polarity_scores(tweet)["neu"]
def countwords(words, is_spam, counted):
    for each_word in words:
        if each_word in counted:
            if is_spam == 1:
                counted[each_word][1]=counted[each_word][1] + 1
            else:
                counted[each_word][0]=counted[each_word][0] + 1
        else:
            if is_spam == 1:
                counted[each_word] = [0,1]
            else:
                counted[each_word] = [1,0]
    return counted

def calculateSpam(subject,vocab,numSpam, numHam):
    spam = 0.0
    not_spam = 0.0
    total = float(numHam) + float(numHam)
    for key in vocab:
        if key in subject:
            not_spam = not_spam + math.log(vocab[key][0])
            spam = spam + math.log(vocab[key][1])
        else:
            not_spam = not_spam + math.log(1-(vocab[key][0]))
            spam = spam + math.log(1-(vocab[key][1]))
    not_spam = math.exp(not_spam)
    spam = math.exp(spam)
    x1 = float(not_spam) * (float(numHam)/total)
    x2 = float(spam) * (float(numSpam)/total)
    result = math.log(x1) - math.log(x2)
    sig = 1 / (1 + math.exp(result))  
    sig = round(sig)
    return sig

def make_percent_list(k, theCount, spams, hams):
    for each_key in theCount:
        theCount[each_key][0] = (theCount[each_key][0] + k)/(2*k+hams)
        theCount[each_key][1] = (theCount[each_key][1] + k)/(2*k+spams)
    return theCount



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
Tweet_df = Tweet_df.sample(frac = 1)
print("Shuffled")
print(Tweet_df.shape)
count = 0
file = input("What file would you like to write to?")
f = open(file, "a")
if(not path.exists('Btweets_clean_small.csv')):
    newdf = pd.read_csv('Btweets_clean.csv', nrows = 450000, low_memory=False)
    newdf.to_csv('Btweets_clean_small.csv', index=False)
#f2 = open("verifiedTweets.txt", "w")
"""for i in Tweet_df.index:
    cleanT = Tweet_df["text"][i]
    if(Tweet_df["user_verified"][i]):
        print(cleanT)
        string = cleanText(cleanT)
        string = "0 " + string + "\n"
        f2.write(string)
f2.close()
"""
for i in Tweet_df.index:
    cleanT = Tweet_df["text"][i]
    print("-------------------------------\n",cleanT, "\nFollowers: ", Tweet_df["user_followers"][i], "\n")
    spam = input("Is this spam? ")
    if(spam == "done"):
        break
    print("\n")
    string = cleanText(cleanT)
    string = spam + " " + string + "\n"
    f.write(string)
    count = count + 1
f.close()

spam = 0
notSpam = 0
counted = dict()
infile = input("Enter the name of the spam-ham file: ")
data = open(infile, "r")
line = data.readline()
while(line != ""):
    is_spam = int(line[:1])
    if is_spam:
        spam = spam + 1
    else:
        notSpam = notSpam + 1
    cleanLine = cleanText(line)
    words = cleanLine.split()
    words = set(words)
    counted = countwords(words, is_spam, counted)
    line = data.readline()
stopFile = input("Enter the name of the stop words file: ")
file2 = open(stopFile, "r")
line2 = file2.readline()
while(line2!= ""):
    line2clean = cleanText(line2)
    if line2clean in counted:
        del counted[line2clean]
        print(line2 + "deleted")
    line2 = file2.readline()
print(counted)
vocab = (make_percent_list(1, counted, spam, notSpam))
print(vocab)
testName = input("Enter the name of the test set file: ")
file3 = open(testName, "r")
line3 = file3.readline()
spam_actual=0
spam_predict=0
not_spam_predict = 0
not_spam_actual = 0
TP = 0
TN = 0
FP = 0
FN = 0
count = 0
spam_prediction = 0
while(line3!= ""):
    is_spam = int(line3[:1])
    if is_spam:
        spam_actual = spam_actual + 1
    else:
        not_spam_actual = not_spam_actual+1
    cleanLine = cleanText(line3)
    words = cleanLine.split()
    spam_prediction = calculateSpam(words,vocab,spam,notSpam)
    if spam_prediction:
        spam_predict = spam_predict +1
        if spam_prediction == is_spam:
            TP = TP + 1
        else:
            FP = FP + 1
    else:
        not_spam_predict = not_spam_predict +1
        if spam_prediction == is_spam:
            TN = TN + 1
        else:
            FN = FN + 1
    count = count + 1
    line3 = file3.readline()
total = spam_actual + not_spam_actual
accuracy = (float(TP + TN)/float(total))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2*(1/((1/precision)+(1/recall)))
print("TN: ", TN)
print("TP: ", TP)
print("FN: ", FN)
print("FP: ", FP)
print("Accuracy: ", accuracy*100, "%")
print("Precision: ", precision*100, "%")
print("Recall: ", recall*100, "%")
print("F1: ", f1*100, "%")
file2.close()
file3.close()
data.close()

#This is all my old code from project 4 




"""
Tweet_df.insert(Tweet_df.shape[1],"positive",int, False)
for tweet in Tweet_df["text"]:
    if(is_positive(tweet)):
        Tweet_df["positive"][count] = 1
    else:
        Tweet_df["positive"][count] = 0
    if((count % 1000) == 0):
        print("1,000 tweets processed")
    count = count + 1
"""

