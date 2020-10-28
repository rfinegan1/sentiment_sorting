#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:54:48 2020

@author: ryanfinegan
"""

#libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

#making a dataframe of all the dataframes
def merge(l=['mine.csv', 'companies.csv','extracted_companies.csv','requests_output.csv','output.csv']):
    #merging all the csv files into one dataframe
    df = pd.concat(map(pd.read_csv, l),ignore_index=True)
    #return the dataframe to be used in to complete the problem 
    return df

#sentiment analysis on all the business ideas to see which are the best and worst
def sentiment(df):
    #calling vader sentiment 
    analyzer = SentimentIntensityAnalyzer()
    #creating an empty list to dump all the results into 
    results = []
    #retrieving all the company purposes for each company 
    for sentence in df['Purpose']:
        #scores (neg,neu,pos,compound) of each company purpose
        vs = analyzer.polarity_scores(sentence)
        #adding all these scores to the results list 
        results.append(vs)
    #dataframe of all the results
    dataframe = pd.DataFrame(results)
    #combining the company name with the sentiment scores of the company purpose
    df = pd.concat([df[['Name']],dataframe],axis=1).set_index('Name')[['compound']]
    #getting the maximum and minimum sentiment scores of each company purpose 
    #did it this was because there were multiple mimimums and I wanted to show dataframes for it
    maximum,minimum = df[df['compound']==df['compound'].max()],df[df['compound']==df['compound'].min()]
    #returning the dataframes of the best and worst business ideas 
    return maximum,minimum

#function is used for identifying the 10 most common words in the description of companies
def common(df):
    #empty list to dump the company purposes into
    data = []
    #looping through each company purpose
    for sentence in df['Purpose']:
        #appending that info into an empty list
        data.append(sentence)
    #adding that info into one dataframe
    df = pd.DataFrame(data) 
    #making the company info into one string 
    df = df[0].str.cat(sep=' ')
    #finding the ten most common words in each of the company purposes
    common_words = Counter(df.split()).most_common()[0:10]
    return common_words

sent = sentiment(merge())
common_words = common(merge())

def main():
    return sent, common_words

if __name__ == '__main__':
    main()
