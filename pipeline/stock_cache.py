import os
import psycopg2
import psycopg2.extras
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def get_latest_dates(symbols: List[str]) -> Dict[str, Optional[date]]:
    if not symbols:
        return {}
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT symbol, MAX(trade_date) as latest_date
            FROM stock_price_cache
            WHERE symbol = ANY(%s)
            GROUP BY symbol
            """,
            (symbols,),
        )
        result = {sym: None for sym in symbols}
        for row in cur.fetchall():
            result[row[0]] = row[1]
        return result
    finally:
        conn.close()


def store_price_data(symbol: str, hist_df: pd.DataFrame):
    if hist_df.empty:
        return
    conn = get_connection()
    try:
        cur = conn.cursor()
        rows = []
        for idx, row in hist_df.iterrows():
            trade_date = idx.date() if hasattr(idx, 'date') else idx
            rows.append((
                symbol,
                trade_date,
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume']),
            ))
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO stock_price_cache (symbol, trade_date, open_price, high_price, low_price, close_price, volume)
            VALUES %s
            ON CONFLICT (symbol, trade_date) DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                volume = EXCLUDED.volume,
                created_at = NOW()
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def store_metadata(symbol: str, info: Dict[str, Any], sector_fallback: str = "Technology"):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO stock_metadata_cache (symbol, company_name, sector, exchange, market_cap, shares_outstanding, pe_ratio, dividend_yield, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (symbol) DO UPDATE SET
                company_name = EXCLUDED.company_name,
                sector = EXCLUDED.sector,
                exchange = EXCLUDED.exchange,
                market_cap = EXCLUDED.market_cap,
                shares_outstanding = EXCLUDED.shares_outstanding,
                pe_ratio = EXCLUDED.pe_ratio,
                dividend_yield = EXCLUDED.dividend_yield,
                last_updated = NOW()
            """,
            (
                symbol,
                info.get('shortName') or info.get('longName') or symbol,
                info.get('sector') or sector_fallback,
                info.get('exchange', 'NASDAQ'),
                info.get('marketCap'),
                info.get('sharesOutstanding'),
                info.get('trailingPE'),
                info.get('dividendYield', 0) or 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_cached_prices(symbol: str, since_months: int = 6) -> Optional[pd.DataFrame]:
    cutoff = date.today() - timedelta(days=since_months * 30)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT trade_date, open_price, high_price, low_price, close_price, volume
            FROM stock_price_cache
            WHERE symbol = %s AND trade_date >= %s
            ORDER BY trade_date ASC
            """,
            (symbol, cutoff),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return df
    finally:
        conn.close()


def load_cached_metadata(symbol: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT company_name, sector, exchange, market_cap, shares_outstanding, pe_ratio, dividend_yield, last_updated
            FROM stock_metadata_cache
            WHERE symbol = %s
            """,
            (symbol,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            'company_name': row[0],
            'sector': row[1],
            'exchange': row[2],
            'market_cap': row[3],
            'shares_outstanding': row[4],
            'pe_ratio': row[5],
            'dividend_yield': row[6],
            'last_updated': row[7],
        }
    finally:
        conn.close()


def get_cache_stats() -> Dict[str, Any]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(trade_date), MAX(trade_date) FROM stock_price_cache")
        row = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM stock_metadata_cache")
        meta_count = cur.fetchone()[0]
        return {
            'cached_symbols': row[0],
            'total_price_rows': row[1],
            'oldest_date': row[2].isoformat() if row[2] else None,
            'newest_date': row[3].isoformat() if row[3] else None,
            'metadata_entries': meta_count,
        }
    finally:
        conn.close()


def clear_cache():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM stock_price_cache")
        cur.execute("DELETE FROM stock_metadata_cache")
        conn.commit()
    finally:
        conn.close()
