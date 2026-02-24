import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any
from textblob import TextBlob
import time


def _parse_news_item(item: dict) -> dict | None:
    content = item.get('content', item)

    title = content.get('title', '') or item.get('title', '')
    if not title:
        return None

    pub_date_str = content.get('pubDate') or content.get('displayTime', '')
    pub_ts = item.get('providerPublishTime', 0)

    if pub_date_str:
        try:
            pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00')).replace(tzinfo=None)
        except Exception:
            pub_date = datetime.now()
    elif pub_ts:
        pub_date = datetime.fromtimestamp(pub_ts)
    else:
        pub_date = datetime.now()

    provider = content.get('provider', {})
    source = provider.get('displayName', '') if isinstance(provider, dict) else ''
    if not source:
        source = item.get('publisher', 'Yahoo Finance')

    click_url = content.get('clickThroughUrl', {})
    url = click_url.get('url', '') if isinstance(click_url, dict) else ''
    if not url:
        canonical = content.get('canonicalUrl', {})
        url = canonical.get('url', '') if isinstance(canonical, dict) else ''
    if not url:
        url = item.get('link', '')

    blob = TextBlob(title)
    sentiment_score = blob.sentiment.polarity

    return {
        'headline': title,
        'source': source,
        'published_date': pub_date.isoformat(),
        'url': url,
        'sentiment_score': sentiment_score,
        'relevance_score': 0.85,
        '_pub_date': pub_date,
    }


def fetch_stock_news(symbol: str, days_back: int = 2) -> List[Dict[str, Any]]:
    cutoff = datetime.now() - timedelta(days=days_back)
    articles = []

    try:
        ticker = yf.Ticker(symbol)
        raw_news = ticker.news or []
    except Exception:
        raw_news = []

    if not raw_news:
        try:
            search = yf.Search(symbol, news_count=10)
            raw_news = search.news or []
        except Exception:
            raw_news = []

    for item in raw_news:
        parsed = _parse_news_item(item)
        if not parsed:
            continue

        pub_date = parsed.pop('_pub_date')
        if pub_date < cutoff:
            continue

        recency = 1.0 - (datetime.now() - pub_date).total_seconds() / (days_back * 86400)
        parsed['relevance_score'] = min(1.0, 0.7 + 0.3 * max(0, recency))

        articles.append(parsed)

    articles.sort(key=lambda x: x['published_date'], reverse=True)
    return articles


def fetch_news_batch(symbols: List[str], days_back: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    results = {}
    for symbol in symbols:
        results[symbol] = fetch_stock_news(symbol, days_back)
        time.sleep(0.2)
    return results
