import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
from utils.real_data import fetch_portfolio_data


class DataIngestionEngine:
    """
    Stage 1: Data Ingestion Engine
    Fetches real market data from Yahoo Finance for portfolio analysis
    """
    
    def __init__(self):
        self.connection_status = "Connected"
    
    def ingest_portfolio_data(self, portfolio_size: int = 25, progress_callback=None) -> List[Dict[str, Any]]:
        """
        Ingest portfolio data from Yahoo Finance (real market data)
        
        Args:
            portfolio_size: Number of assets in portfolio
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of dictionaries containing asset data
        """
        print(f"Connecting to Yahoo Finance API...")
        print(f"Fetching real market data for {portfolio_size} assets...")
        
        portfolio_data = fetch_portfolio_data(portfolio_size, progress_callback)
        
        print(f"Successfully ingested data for {len(portfolio_data)} assets")
        return portfolio_data
    
    def validate_data_integrity(self, portfolio_data: List[Dict]) -> Dict[str, Any]:
        """
        Validate the integrity of ingested data
        
        Args:
            portfolio_data: List of asset data dictionaries
            
        Returns:
            Validation results
        """
        validation_results = {
            'total_assets': len(portfolio_data),
            'complete_records': 0,
            'missing_data_assets': [],
            'data_quality_issues': [],
            'average_data_quality': 0.0
        }
        
        quality_scores = []
        
        for asset in portfolio_data:
            required_fields = ['symbol', 'current_price', 'historical_prices', 'market_cap']
            missing_fields = [field for field in required_fields if field not in asset or asset[field] is None]
            
            if not missing_fields:
                validation_results['complete_records'] += 1
            else:
                validation_results['missing_data_assets'].append({
                    'symbol': asset.get('symbol', 'Unknown'),
                    'missing_fields': missing_fields
                })
            
            if 'data_quality_score' in asset:
                quality_scores.append(asset['data_quality_score'])
                
                if asset['data_quality_score'] < 0.9:
                    validation_results['data_quality_issues'].append({
                        'symbol': asset['symbol'],
                        'quality_score': asset['data_quality_score']
                    })
        
        if quality_scores:
            validation_results['average_data_quality'] = np.mean(quality_scores)
        
        return validation_results
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch real-time data for specific symbols
        
        Args:
            symbols: List of asset symbols
            
        Returns:
            Dictionary with real-time data for each symbol
        """
        import yfinance as yf
        
        real_time_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info or {}
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    last_price = float(hist['Close'].iloc[-1])
                    prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else last_price
                    change_pct = ((last_price - prev_price) / prev_price) * 100
                    
                    real_time_data[symbol] = {
                        'last_price': round(last_price, 2),
                        'bid': round(info.get('bid', last_price), 2),
                        'ask': round(info.get('ask', last_price), 2),
                        'volume': int(hist['Volume'].iloc[-1]),
                        'timestamp': datetime.now().isoformat(),
                        'change_percent': round(change_pct, 2)
                    }
            except Exception as e:
                real_time_data[symbol] = {
                    'last_price': 0,
                    'bid': 0,
                    'ask': 0,
                    'volume': 0,
                    'timestamp': datetime.now().isoformat(),
                    'change_percent': 0,
                    'error': str(e)
                }
        
        return real_time_data
    
    def check_connection_status(self) -> Dict[str, Any]:
        """
        Check Yahoo Finance API connection status
        
        Returns:
            Connection status information
        """
        import yfinance as yf
        
        try:
            test = yf.Ticker("AAPL")
            hist = test.history(period="1d")
            if not hist.empty:
                status = "Connected"
                latency = "OK"
            else:
                status = "Limited"
                latency = "Slow"
        except Exception:
            status = "Disconnected"
            latency = "N/A"
        
        return {
            'status': status,
            'data_source': 'Yahoo Finance (Free)',
            'last_check': datetime.now().isoformat(),
            'latency': latency,
            'rate_limit': 'No hard limit (respectful throttling applied)'
        }
