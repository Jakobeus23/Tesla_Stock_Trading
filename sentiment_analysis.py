import os
import requests
import json
from datetime import datetime, timedelta


API_KEY = "MY_API_KEY_YOU_CANT_SEE"  
BASE_URL = "https://gnews.io/api/v4/search"
QUERY = "Tesla"
START_DATE = "2010-06-29"
END_DATE = "2022-03-24"
OUTPUT_FOLDER = "tweets_articles"
LANG = "en"
MAX_RESULTS = 100  

# Create the output folder if needed
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helper function to fetch news articles
def fetch_news(query, from_date, to_date, api_key, lang="en", max_results=100):
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "token": api_key,
        "lang": lang,
        "max": max_results
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Break the date range into chunks (e.g., one month at a time)
def generate_date_ranges(start_date, end_date, chunk_size_days=30):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while start < end:
        chunk_end = min(start + timedelta(days=chunk_size_days), end)
        yield start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        start = chunk_end

# Fetch articles and save them to files
for from_date, to_date in generate_date_ranges(START_DATE, END_DATE):
    print(f"Fetching articles from {from_date} to {to_date}...")
    articles = fetch_news(QUERY, from_date, to_date, API_KEY, lang=LANG, max_results=MAX_RESULTS)
    
    if articles and "articles" in articles:
        for article in articles["articles"]:
            # Save each article as a JSON file
            timestamp = article["publishedAt"].replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
            filename = f"{timestamp}.json"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(article, file, ensure_ascii=False, indent=4)
        print(f"Saved {len(articles['articles'])} articles from {from_date} to {to_date}.")
    else:
        print(f"No articles found for {from_date} to {to_date}.")
