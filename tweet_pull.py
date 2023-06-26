#https://developer.twitter.com/en/docs/tutorials/step-by-step-guide-to-making-your-first-request-to-the-twitter-api-v2

#curl --request GET 'https://api.twitter.com/2/tweets/search/recent?query=from:twitterdev' --header 'Authorization: Bearer $BEARER_TOKEN'
import requests
import tweepy
import configparser
tweeter_handle = '@SamSammeli'


config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['twitter']['api_key']
api_secret = config['twitter']['api_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

import json
public_tweets =api.user_timeline(screen_name=tweeter_handle, count=100, exclude_replies=False)
selected_tweets=[] 
for pt in public_tweets:    
    if pt.lang == 'en':
        st = {'id': pt.id_str, 'reply_to' :pt.in_reply_to_status_id_str, 'text':pt.text}         
        selected_tweets.append(st)
with open('./data/tweetfiles/tweets.txt', 'a') as f:
    f.write(json.dumps(selected_tweets))
