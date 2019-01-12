import json
from twython import Twython

# read authentication credentials
with open('credentials.json') as f:
  credentials = json.load(f)
  accessToken = credentials['access_token']
  accessSecret = credentials['access_token_secret']
  consumerKey = credentials['consumer_key']
  consumerSecret = credentials['consumer_key_secret']

twitter = Twython(accessToken, accessSecret, oauth_version=2)
authentication_token = twitter.obtain_access_token()

print(authentication_token)