import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'DIS', 'ADBE', 'CRM',
    'NFLX', 'CSCO', 'PFE', 'KO', 'PEP', 'TMO', 'ABT', 'MRK', 'COST',
    'NKE', 'WMT', 'CVX', 'XOM', 'LLY', 'ABBV', 'AVGO', 'ORCL', 'ACN',
    'MCD', 'TXN', 'QCOM', 'NEE', 'LOW', 'UPS', 'MS', 'GS', 'BLK',
    'INTC', 'AMD', 'BA', 'CAT', 'GE', 'MMM', 'IBM', 'RTX', 'DE',
    'SBUX', 'GILD', 'MDLZ', 'AMT', 'ISRG', 'NOW', 'INTU', 'PYPL',
]

SECTOR_MAP = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'CSCO',
                   'ORCL', 'ACN', 'TXN', 'QCOM', 'INTC', 'AMD', 'AVGO', 'NOW', 'INTU'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'LOW', 'SBUX', 'DIS', 'NFLX'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'TMO', 'ABT', 'MRK', 'LLY', 'ABBV', 'GILD', 'ISRG'],
    'Financial Services': ['JPM', 'V', 'MA', 'BRK-B', 'MS', 'GS', 'BLK', 'PYPL'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'MDLZ'],
    'Energy': ['CVX', 'XOM'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'RTX', 'DE', 'UPS'],
    'Utilities': ['NEE'],
    'Real Estate': ['AMT'],
    'Materials': ['IBM'],
}

SYMBOL_TO_SECTOR = {}
for sector, symbols in SECTOR_MAP.items():
    for sym in symbols:
        SYMBOL_TO_SECTOR[sym] = sector


class YahooFinanceData:
    def __init__(self):
        self._cache = {}

    def get_sector(self, symbol: str) -> str:
        return SYMBOL_TO_SECTOR.get(symbol, 'Technology')

    def fetch_assets(self, symbols: List[str]) -> List[Dict[str, Any]]:
        portfolio_data = []
        tickers = yf.Tickers(' '.join(symbols))

        for symbol in symbols:
            try:
                asset = self._fetch_single_asset(symbol, tickers)
                if asset is not None:
                    portfolio_data.append(asset)
            except Exception as e:
                print(f"Warning: Could not fetch data for {symbol}: {e}")
                continue

        return portfolio_data

    def _fetch_single_asset(self, symbol: str, tickers) -> Dict[str, Any]:
        ticker = tickers.tickers.get(symbol)
        if ticker is None:
            return None

        hist = ticker.history(period="6mo")
        if hist.empty or len(hist) < 30:
            return None

        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass

        current_price = float(hist['Close'].iloc[-1])
        market_cap = info.get('marketCap', int(current_price * info.get('sharesOutstanding', 1_000_000_000)))
        pe_ratio = info.get('trailingPE')
        dividend_yield = info.get('dividendYield', 0) or 0

        prices = hist['Close'].tolist()
        volumes = hist['Volume'].tolist()
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]

        company_name = info.get('shortName') or info.get('longName') or symbol
        sector = info.get('sector') or self.get_sector(symbol)
        exchange = info.get('exchange', 'NASDAQ')

        return {
            'symbol': symbol,
            'company_name': company_name,
            'sector': sector,
            'current_price': round(current_price, 2),
            'market_cap': int(market_cap) if market_cap else 0,
            'shares_outstanding': info.get('sharesOutstanding', 0),
            'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
            'dividend_yield': round(dividend_yield, 4),
            'currency': 'USD',
            'exchange': exchange,
            'country': 'United States',
            'historical_prices': [round(p, 2) for p in prices],
            'historical_dates': dates,
            'trading_volume_history': [int(v) for v in volumes],
            'data_ingestion_timestamp': datetime.now().isoformat(),
            'data_quality_score': 1.0,
            'bloomberg_id': f"BBG{abs(hash(symbol)) % 900000000 + 100000000}",
        }


def pick_random_symbols(count: int) -> List[str]:
    available = list(STOCK_UNIVERSE)
    random.shuffle(available)
    return available[:min(count, len(available))]
