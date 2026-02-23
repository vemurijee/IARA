import os
import json
import psycopg2
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def save_pipeline_run(
    run_name: str,
    portfolio_size: int,
    risk_thresholds: Dict,
    portfolio_data: List[Dict],
    analysis_results: List[Dict],
    ml_results: Dict,
    sentiment_results: List[Dict],
    execution_time: float,
) -> int:
    red_count = len([a for a in analysis_results if a["risk_rating"] == "RED"])
    yellow_count = len([a for a in analysis_results if a["risk_rating"] == "YELLOW"])
    green_count = len([a for a in analysis_results if a["risk_rating"] == "GREEN"])

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO pipeline_runs 
                (run_name, portfolio_size, risk_thresholds, portfolio_data, 
                 analysis_results, ml_results, sentiment_results, execution_time,
                 red_count, yellow_count, green_count, total_assets)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                run_name,
                portfolio_size,
                json.dumps(risk_thresholds, cls=NumpyEncoder),
                json.dumps(portfolio_data, cls=NumpyEncoder),
                json.dumps(analysis_results, cls=NumpyEncoder),
                json.dumps(ml_results, cls=NumpyEncoder),
                json.dumps(sentiment_results, cls=NumpyEncoder),
                execution_time,
                red_count,
                yellow_count,
                green_count,
                len(analysis_results),
            ),
        )
        run_id = cur.fetchone()[0]
        conn.commit()
        return run_id
    finally:
        conn.close()


def list_pipeline_runs() -> List[Dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, run_name, run_timestamp, portfolio_size, execution_time,
                   red_count, yellow_count, green_count, total_assets
            FROM pipeline_runs
            ORDER BY run_timestamp DESC
            LIMIT 50
            """
        )
        rows = cur.fetchall()
        columns = [
            "id", "run_name", "run_timestamp", "portfolio_size", "execution_time",
            "red_count", "yellow_count", "green_count", "total_assets",
        ]
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def load_pipeline_run(run_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT portfolio_data, analysis_results, ml_results, sentiment_results,
                   execution_time, risk_thresholds, run_name, run_timestamp
            FROM pipeline_runs
            WHERE id = %s
            """,
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "portfolio_data": row[0] if isinstance(row[0], list) else json.loads(row[0]),
            "analysis_results": row[1] if isinstance(row[1], list) else json.loads(row[1]),
            "ml_results": row[2] if isinstance(row[2], dict) else json.loads(row[2]),
            "sentiment_results": row[3] if isinstance(row[3], list) else json.loads(row[3]),
            "execution_time": row[4],
            "risk_thresholds": row[5] if isinstance(row[5], dict) else json.loads(row[5]),
            "run_name": row[6],
            "run_timestamp": row[7],
        }
    finally:
        conn.close()


def delete_pipeline_run(run_id: int) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM pipeline_runs WHERE id = %s", (run_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
