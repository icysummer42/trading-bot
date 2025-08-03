import snscrape.modules.twitter as sntwitter

for tweet in sntwitter.TwitterSearchScraper('TSLA').get_items():
    print(tweet)
    break
