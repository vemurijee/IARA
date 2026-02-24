import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
import random

from utils.stock_universe import STOCK_UNIVERSE, SECTOR_MAP, SYMBOL_TO_SECTOR


class YahooFinanceData:
    def __init__(self):
        self._cache = {}
        self._use_db_cache = True
        self._fetch_stats = {'full_fetches': 0, 'delta_fetches': 0, 'cache_only': 0}

    def get_sector(self, symbol: str) -> str:
        return SYMBOL_TO_SECTOR.get(symbol, 'Technology')

    def fetch_assets(self, symbols: List[str], progress_callback=None) -> List[Dict[str, Any]]:
        self._fetch_stats = {'full_fetches': 0, 'delta_fetches': 0, 'cache_only': 0}
        portfolio_data = []
        db_available = True

        try:
            from pipeline.stock_cache import get_latest_dates, store_price_data, store_metadata, load_cached_prices, load_cached_metadata
        except Exception:
            db_available = False

        latest_dates = {}
        if db_available:
            try:
                latest_dates = get_latest_dates(symbols)
            except Exception as e:
                print(f"Warning: Could not check cache: {e}")
                db_available = False

        if not db_available:
            if progress_callback:
                progress_callback("Cache unavailable, fetching all data from Yahoo Finance...")
            return self._fetch_all_fresh(symbols, progress_callback)

        full_fetch_symbols = []
        delta_fetch_symbols = []
        cache_only_symbols = []

        today = date.today()
        for sym in symbols:
            cached_date = latest_dates.get(sym)
            if cached_date is None:
                full_fetch_symbols.append(sym)
            elif cached_date >= today - timedelta(days=2):
                cache_only_symbols.append(sym)
            else:
                delta_fetch_symbols.append(sym)

        if progress_callback:
            progress_callback(f"Cache: {len(cache_only_symbols)} up-to-date, {len(delta_fetch_symbols)} need delta, {len(full_fetch_symbols)} need full fetch")

        need_fetch = full_fetch_symbols + delta_fetch_symbols
        fetched_hist = {}
        fetched_info = {}

        if need_fetch:
            tickers = yf.Tickers(' '.join(need_fetch))
            for symbol in need_fetch:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if ticker is None:
                        continue

                    cached_date = latest_dates.get(symbol)
                    if cached_date is not None:
                        start_date = cached_date + timedelta(days=1)
                        hist = ticker.history(start=start_date.strftime('%Y-%m-%d'))
                        self._fetch_stats['delta_fetches'] += 1
                    else:
                        hist = ticker.history(period="6mo")
                        self._fetch_stats['full_fetches'] += 1

                    fetched_hist[symbol] = hist

                    if not hist.empty:
                        try:
                            store_price_data(symbol, hist)
                        except Exception as e:
                            print(f"Warning: Could not cache prices for {symbol}: {e}")

                    info = {}
                    try:
                        info = ticker.info or {}
                    except Exception:
                        pass
                    fetched_info[symbol] = info

                    if info:
                        try:
                            store_metadata(symbol, info, self.get_sector(symbol))
                        except Exception as e:
                            print(f"Warning: Could not cache metadata for {symbol}: {e}")
                except Exception as e:
                    print(f"Warning: Could not fetch data for {symbol}: {e}")
                    continue

        all_symbols = cache_only_symbols + [s for s in need_fetch if s in fetched_info or s in fetched_hist]
        for symbol in all_symbols:
            try:
                cached_prices = None
                cached_meta = None
                try:
                    cached_prices = load_cached_prices(symbol)
                    cached_meta = load_cached_metadata(symbol)
                except Exception:
                    pass

                if cached_prices is not None and len(cached_prices) >= 30:
                    if symbol in cache_only_symbols:
                        self._fetch_stats['cache_only'] += 1

                    info = fetched_info.get(symbol, {})
                    asset = self._build_asset_from_cache(symbol, cached_prices, cached_meta, info,
                                                         'cache' if symbol in cache_only_symbols else 'delta' if symbol in delta_fetch_symbols else 'full')
                    portfolio_data.append(asset)
                else:
                    asset = self._build_asset_from_fresh_fetch(symbol, fetched_info.get(symbol, {}))
                    if asset:
                        portfolio_data.append(asset)
            except Exception as e:
                print(f"Warning: Could not process {symbol}: {e}")
                continue

        return portfolio_data

    def _fetch_all_fresh(self, symbols: List[str], progress_callback=None) -> List[Dict[str, Any]]:
        portfolio_data = []
        tickers = yf.Tickers(' '.join(symbols))
        for symbol in symbols:
            try:
                asset = self._fetch_single_asset_direct(symbol, tickers)
                if asset is not None:
                    self._fetch_stats['full_fetches'] += 1
                    portfolio_data.append(asset)
            except Exception as e:
                print(f"Warning: Could not fetch data for {symbol}: {e}")
        return portfolio_data

    def _fetch_single_asset_direct(self, symbol: str, tickers) -> Optional[Dict[str, Any]]:
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
        return {
            'symbol': symbol,
            'company_name': info.get('shortName') or info.get('longName') or symbol,
            'sector': info.get('sector') or self.get_sector(symbol),
            'current_price': round(current_price, 2),
            'market_cap': int(market_cap) if market_cap else 0,
            'shares_outstanding': info.get('sharesOutstanding', 0),
            'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
            'dividend_yield': round(dividend_yield, 4),
            'currency': 'USD',
            'exchange': info.get('exchange', 'NASDAQ'),
            'country': 'United States',
            'historical_prices': [round(p, 2) for p in prices],
            'historical_dates': dates,
            'trading_volume_history': [int(v) for v in volumes],
            'data_ingestion_timestamp': datetime.now().isoformat(),
            'data_quality_score': 1.0,
            'bloomberg_id': f"BBG{abs(hash(symbol)) % 900000000 + 100000000}",
            'data_source': 'full',
        }

    def _build_asset_from_cache(self, symbol: str, cached_prices, cached_meta: Optional[Dict], info: Dict, source: str) -> Dict[str, Any]:
        company_name = info.get('shortName') or info.get('longName')
        if not company_name and cached_meta:
            company_name = cached_meta['company_name']
        if not company_name:
            company_name = symbol

        sector = info.get('sector') or (cached_meta['sector'] if cached_meta else None) or self.get_sector(symbol)
        exchange = info.get('exchange') or (cached_meta['exchange'] if cached_meta else 'NASDAQ')

        current_price = float(cached_prices['Close'].iloc[-1])
        market_cap = info.get('marketCap')
        if not market_cap and cached_meta and cached_meta.get('market_cap'):
            market_cap = cached_meta['market_cap']
        if not market_cap:
            market_cap = int(current_price * info.get('sharesOutstanding', 1_000_000_000))

        pe_ratio = info.get('trailingPE')
        if pe_ratio is None and cached_meta:
            pe_ratio = cached_meta.get('pe_ratio')

        dividend_yield = info.get('dividendYield', 0) or 0
        if not dividend_yield and cached_meta:
            dividend_yield = cached_meta.get('dividend_yield', 0) or 0

        shares = info.get('sharesOutstanding', 0)
        if not shares and cached_meta:
            shares = cached_meta.get('shares_outstanding', 0) or 0

        prices = cached_prices['Close'].tolist()
        volumes = cached_prices['Volume'].tolist()
        dates = [d.strftime('%Y-%m-%d') for d in cached_prices.index]

        return {
            'symbol': symbol,
            'company_name': company_name,
            'sector': sector,
            'current_price': round(current_price, 2),
            'market_cap': int(market_cap) if market_cap else 0,
            'shares_outstanding': int(shares) if shares else 0,
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
            'data_source': source,
        }

    def _build_asset_from_fresh_fetch(self, symbol: str, info: Dict) -> Optional[Dict[str, Any]]:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            if hist.empty or len(hist) < 30:
                return None

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
                'data_source': 'full',
            }
        except Exception:
            return None

    def get_fetch_stats(self) -> Dict[str, int]:
        return self._fetch_stats.copy()


def pick_random_symbols(count: int) -> List[str]:
    available = list(STOCK_UNIVERSE)
    random.shuffle(available)
    return available[:min(count, len(available))]
