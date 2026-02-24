import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random
import time


STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE',
                   'INTC', 'CSCO', 'IBM', 'TXN', 'QCOM', 'NOW', 'INTU', 'AMAT', 'MU', 'LRCX'],
    'Healthcare': ['JNJ', 'UNH', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
                   'AMGN', 'GILD', 'MDT', 'CVS', 'CI', 'SYK', 'ZTS', 'REGN', 'VRTX', 'BDX'],
    'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'AXP', 'USB',
                           'PNC', 'TFC', 'COF', 'BK', 'CME', 'ICE', 'MCO', 'SPGI', 'MMC', 'AON'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
                                'MAR', 'GM', 'F', 'ROST', 'DHI', 'YUM', 'ORLY', 'AZO', 'EBAY', 'ETSY'],
    'Industrial': ['CAT', 'HON', 'UPS', 'BA', 'RTX', 'DE', 'LMT', 'GE', 'MMM', 'UNP',
                   'FDX', 'EMR', 'ITW', 'ETN', 'WM', 'NSC', 'CSX', 'JCI', 'GD', 'PH'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES',
               'WMB', 'KMI', 'HAL', 'DVN', 'BKR', 'FANG', 'OKE', 'TRGP', 'PXD', 'CTRA'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'GIS',
                         'KHC', 'HSY', 'K', 'SJM', 'CAG', 'STZ', 'KDP', 'TAP', 'TSN', 'HRL'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'WEC',
                  'PCG', 'ES', 'AWK', 'ATO', 'CMS', 'DTE', 'PPL', 'FE', 'AES', 'CEG'],
    'Real Estate': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'DLR', 'WELL', 'AVB',
                    'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'ESS', 'HST', 'KIM', 'REG', 'BXP'],
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'VMC',
                  'MLM', 'PPG', 'ALB', 'CF', 'MOS', 'IFF', 'CE', 'EMN', 'PKG', 'WRK']
}


def select_random_tickers(portfolio_size: int = 25) -> List[str]:
    all_tickers = []
    for sector, tickers in STOCK_UNIVERSE.items():
        all_tickers.extend(tickers)

    random.shuffle(all_tickers)
    return all_tickers[:portfolio_size]


def select_diversified_tickers(portfolio_size: int = 25) -> List[str]:
    selected = []
    sectors = list(STOCK_UNIVERSE.keys())
    per_sector = max(1, portfolio_size // len(sectors))
    remainder = portfolio_size - per_sector * len(sectors)

    for sector in sectors:
        tickers = STOCK_UNIVERSE[sector].copy()
        random.shuffle(tickers)
        selected.extend(tickers[:per_sector])

    if remainder > 0:
        all_remaining = []
        for sector in sectors:
            tickers = STOCK_UNIVERSE[sector].copy()
            already = [t for t in selected if t in tickers]
            remaining = [t for t in tickers if t not in already]
            all_remaining.extend(remaining)
        random.shuffle(all_remaining)
        selected.extend(all_remaining[:remainder])

    return selected[:portfolio_size]


SECTOR_MAP = {}
for sector, tickers in STOCK_UNIVERSE.items():
    for ticker in tickers:
        SECTOR_MAP[ticker] = sector


def fetch_asset_data(symbol: str, period: str = "1y") -> Optional[Dict[str, Any]]:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty or len(hist) < 60:
            return None

        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass

        current_price = float(hist['Close'].iloc[-1])
        market_cap = info.get('marketCap', None)
        if market_cap is None:
            shares = info.get('sharesOutstanding', None)
            if shares:
                market_cap = int(current_price * shares)
            else:
                market_cap = int(current_price * 1_000_000_000)

        prices = [round(float(p), 2) for p in hist['Close'].tolist()]
        volumes = [int(v) for v in hist['Volume'].tolist()]
        dates = [d.strftime('%Y-%m-%d') for d in hist.index.tolist()]

        company_name = info.get('shortName') or info.get('longName') or symbol
        sector = info.get('sector') or SECTOR_MAP.get(symbol, 'Other')
        exchange = info.get('exchange', 'UNKNOWN')

        exchange_map = {
            'NMS': 'NASDAQ', 'NGM': 'NASDAQ', 'NCM': 'NASDAQ', 'NAS': 'NASDAQ',
            'NYQ': 'NYSE', 'NYS': 'NYSE', 'PCX': 'NYSE ARCA', 'BTS': 'BATS'
        }
        exchange = exchange_map.get(exchange, exchange)

        pe_ratio = info.get('trailingPE')
        if pe_ratio is not None:
            pe_ratio = round(float(pe_ratio), 2)

        dividend_yield = info.get('dividendYield')
        if dividend_yield is not None:
            dividend_yield = round(float(dividend_yield), 4)
        else:
            dividend_yield = 0.0

        shares_outstanding = info.get('sharesOutstanding', int(market_cap / current_price))

        returns = np.diff(prices) / np.array(prices[:-1])
        volatility_base = float(np.std(returns) * np.sqrt(252))

        return {
            'symbol': symbol,
            'company_name': company_name,
            'sector': sector,
            'current_price': round(current_price, 2),
            'market_cap': int(market_cap),
            'shares_outstanding': int(shares_outstanding),
            'pe_ratio': pe_ratio,
            'dividend_yield': dividend_yield,
            'volatility_base': round(volatility_base, 4),
            'currency': info.get('currency', 'USD'),
            'exchange': exchange,
            'country': info.get('country', 'United States'),
            'historical_prices': prices,
            'historical_dates': dates,
            'trading_volume_history': volumes,
            'data_quality_score': 1.0,
            'data_source': 'Yahoo Finance'
        }

    except Exception as e:
        print(f"  Warning: Failed to fetch data for {symbol}: {e}")
        return None


def fetch_portfolio_data(portfolio_size: int = 25, progress_callback=None) -> List[Dict[str, Any]]:
    tickers = select_diversified_tickers(min(portfolio_size + 10, 60))

    portfolio_data = []
    attempted = 0

    for i, symbol in enumerate(tickers):
        if len(portfolio_data) >= portfolio_size:
            break

        attempted += 1
        if progress_callback:
            progress_callback(symbol, len(portfolio_data), portfolio_size)

        asset_data = fetch_asset_data(symbol)
        if asset_data:
            asset_data['data_ingestion_timestamp'] = datetime.now().isoformat()
            asset_data['bloomberg_id'] = f"BBG-YF-{symbol}"
            portfolio_data.append(asset_data)

        if i < len(tickers) - 1:
            time.sleep(0.15)

    return portfolio_data
