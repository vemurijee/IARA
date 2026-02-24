import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from textblob import TextBlob
import re

from utils.news_fetcher import fetch_stock_news


class SentimentAnalysisEngine:
    """
    Stage 3: Sentiment Analysis Engine
    Analyzes real news sentiment for RED-flagged assets only.
    Fetches live news from Yahoo Finance, limited to last 2 days.
    """
    
    def __init__(self):
        self.sentiment_threshold_negative = -0.3
        self.sentiment_threshold_positive = 0.3
    
    def analyze_sentiment(self, red_flagged_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sentiment_results = []
        
        for asset in red_flagged_assets:
            news_articles = fetch_stock_news(asset['symbol'], days_back=2)
            asset_sentiment = self.analyze_asset_sentiment(asset, news_articles)
            sentiment_results.append(asset_sentiment)
        
        return sentiment_results
    
    def analyze_asset_sentiment(self, asset: Dict[str, Any], news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not news_articles:
            return {
                'symbol': asset['symbol'],
                'sector': asset.get('sector', ''),
                'risk_rating': asset.get('risk_rating', 'RED'),
                'news_count': 0,
                'sentiment_score': 0.0,
                'sentiment_label': 'NEUTRAL',
                'confidence': 0.0,
                'recent_news': [],
                'sentiment_trend': 'STABLE',
                'key_themes': [],
                'recent_significant_news': [],
                'recent_news_count': 0,
                'negative_news_ratio': 0.0,
                'recent_sentiment': 0.0,
                'older_sentiment': 0.0,
                'sentiment_volatility': 0.0,
                'all_articles': [],
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        sentiment_scores = [article['sentiment_score'] for article in news_articles]
        relevance_scores = [article['relevance_score'] for article in news_articles]
        
        weighted_sentiment = np.average(sentiment_scores, weights=relevance_scores)
        
        one_day_ago = datetime.now() - timedelta(days=1)
        recent_articles = []
        older_articles = []
        for a in news_articles:
            try:
                pub = datetime.fromisoformat(a['published_date'].replace('Z', '+00:00')).replace(tzinfo=None)
            except Exception:
                pub = datetime.now()
            if pub > one_day_ago:
                recent_articles.append(a)
            else:
                older_articles.append(a)
        
        recent_sentiment = np.mean([a['sentiment_score'] for a in recent_articles]) if recent_articles else 0
        older_sentiment = np.mean([a['sentiment_score'] for a in older_articles]) if older_articles else 0
        
        if recent_sentiment > older_sentiment + 0.1:
            sentiment_trend = 'IMPROVING'
        elif recent_sentiment < older_sentiment - 0.1:
            sentiment_trend = 'DETERIORATING'
        else:
            sentiment_trend = 'STABLE'
        
        if weighted_sentiment <= self.sentiment_threshold_negative:
            sentiment_label = 'NEGATIVE'
        elif weighted_sentiment >= self.sentiment_threshold_positive:
            sentiment_label = 'POSITIVE'
        else:
            sentiment_label = 'NEUTRAL'
        
        sentiment_std = np.std(sentiment_scores)
        confidence = min(1.0, len(news_articles) / 10.0) * (1.0 - min(1.0, sentiment_std))
        
        key_themes = self.extract_key_themes(news_articles)
        
        recent_significant = sorted(
            [a for a in news_articles if abs(a['sentiment_score']) > 0.1],
            key=lambda x: abs(x['sentiment_score']),
            reverse=True
        )[:5]
        
        neg_count = len([a for a in news_articles if a['sentiment_score'] < -0.1])
        
        return {
            'symbol': asset['symbol'],
            'sector': asset.get('sector', ''),
            'risk_rating': asset.get('risk_rating', 'RED'),
            'sentiment_score': weighted_sentiment,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'sentiment_trend': sentiment_trend,
            'news_count': len(news_articles),
            'recent_news_count': len(recent_articles),
            'negative_news_ratio': neg_count / len(news_articles) if news_articles else 0,
            'key_themes': key_themes,
            'recent_significant_news': recent_significant,
            'recent_sentiment': recent_sentiment,
            'older_sentiment': older_sentiment,
            'sentiment_volatility': sentiment_std,
            'all_articles': news_articles,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def extract_key_themes(self, news_articles: List[Dict[str, Any]]) -> List[str]:
        theme_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'guidance', 'quarter'],
            'regulatory': ['investigation', 'regulatory', 'compliance', 'lawsuit', 'settlement', 'sec', 'ftc'],
            'management': ['ceo', 'resign', 'leadership', 'board', 'management', 'executive'],
            'market_share': ['competitor', 'market share', 'competitive', 'rival'],
            'financial_health': ['debt', 'credit', 'rating', 'downgrade', 'bankruptcy', 'bond'],
            'operations': ['restructuring', 'layoffs', 'production', 'supply chain', 'manufacturing'],
            'growth': ['expansion', 'partnership', 'acquisition', 'buyback', 'investment', 'deal'],
            'technology': ['ai', 'artificial intelligence', 'technology', 'innovation', 'product launch'],
            'macro': ['inflation', 'interest rate', 'fed', 'recession', 'tariff', 'trade'],
        }
        
        theme_counts = {theme: 0 for theme in theme_keywords}
        
        for article in news_articles:
            headline_lower = article['headline'].lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in headline_lower for keyword in keywords):
                    theme_counts[theme] += 1
        
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes if count > 0][:5]
    
    def calculate_sentiment_impact_score(self, sentiment_result: Dict[str, Any], 
                                       risk_metrics: Dict[str, Any]) -> float:
        sentiment_impact = abs(sentiment_result['sentiment_score']) * sentiment_result['confidence']
        
        if sentiment_result['sentiment_trend'] == 'DETERIORATING':
            sentiment_impact *= 1.2
        elif sentiment_result['sentiment_trend'] == 'IMPROVING':
            sentiment_impact *= 0.8
        
        risk_score = risk_metrics.get('risk_score', 0) / 7.0
        volatility_score = min(1.0, risk_metrics.get('volatility', 0) / 0.5)
        
        combined_score = (
            0.4 * sentiment_impact +
            0.3 * risk_score +
            0.3 * volatility_score
        )
        
        return min(1.0, combined_score)
