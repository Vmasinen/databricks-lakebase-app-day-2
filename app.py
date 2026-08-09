"""
Databricks App boilerplate:
- Serves a small Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls data from the Massive API via massive_client.py and syncs it into Lakebase

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os
import re

import requests
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from flask import Flask, jsonify, render_template, request

import lakebase
from massive_client import MassiveClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("massive-app")

app = Flask(__name__)
_w = WorkspaceClient()

TABLE_NAME = os.environ.get("MASSIVE_TABLE_NAME", "massive_records")
WATCHLIST_TABLE_NAME = os.environ.get("WATCHLIST_TABLE_NAME", "watchlist")
NEWS_TABLE_NAME = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")

# Tickers to fetch news for by default (comma-separated), e.g. "AAPL,MSFT,GOOGL"
DEFAULT_NEWS_TICKERS = [
    t.strip().upper()
    for t in os.environ.get("NEWS_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
    if t.strip()
]

# Basic stock ticker shape check: 1-10 uppercase letters, with an optional
# ".X" or ".XX" share-class suffix (e.g. "BRK.B"). This rejects obviously
# malformed input before we even call the Massive API.
_TICKER_RE = re.compile(r"^[A-Z]{1,10}(\.[A-Z]{1,2})?$")


def ensure_table():
    """Create the destination table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def ensure_watchlist_table():
    """Create the watchlist table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WATCHLIST_TABLE_NAME} (
            symbol TEXT NOT NULL,
            email TEXT NOT NULL,
            latest_price NUMERIC,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, email)
        )
        """
    )


def ensure_news_table():
    """
    Create the raw ticker-news documents table in Lakebase if it doesn't
    exist yet. This is the RAW document store the Spark notebook
    (notebooks/ingest_ticker_news_embeddings.py) reads from to compute
    vector embeddings into a separate `<NEWS_TABLE_NAME>_embeddings` table.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {NEWS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            author TEXT,
            article_url TEXT,
            publisher_name TEXT,
            keywords JSONB,
            sentiment TEXT,
            sentiment_reasoning TEXT,
            published_utc TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{NEWS_TABLE_NAME}_ticker "
        f"ON {NEWS_TABLE_NAME} (ticker)"
    )


def _current_user_email() -> str:
    """
    Resolve the current user's email so the watchlist can be personalized.

    Databricks Apps inject the logged-in user's identity via the
    X-Forwarded-Email header on every request. Fall back to the Databricks
    SDK's current_user API for local development where that header isn't set.
    """
    header_email = request.headers.get("X-Forwarded-Email")
    if header_email:
        return header_email
    return _w.current_user.me().user_name


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Simple UI to submit a list of stock symbols to sync from Massive."""
    return render_template("index.html")


@app.route("/records")
def list_records():
    """Read records already synced into Lakebase."""
    limit = int(request.args.get("limit", 100))
    rows = lakebase.run_query(
        f"SELECT id, payload, synced_at FROM {TABLE_NAME} ORDER BY synced_at DESC LIMIT %s",
        (limit,),
    )
    return jsonify(rows)


@app.route("/sync", methods=["POST"])
def sync_from_massive():
    """
    Pull data from the Massive API (paginated, potentially huge dataset) and
    upsert it into Lakebase in batches.
    """
    ensure_table()
    client = MassiveClient()

    path = request.json.get("path", "/records") if request.is_json else "/records"
    batch_size = int(request.args.get("batch_size", 500))

    batch = []
    total = 0
    for item in client.paginated_get(path):
        batch.append(item)
        if len(batch) >= batch_size:
            total += _upsert_batch(batch)
            batch = []

    if batch:
        total += _upsert_batch(batch)

    return jsonify({"synced": total})


@app.route("/news/sync", methods=["POST"])
def sync_news_from_massive():
    """
    Pull recent news articles for a set of tickers from Massive (ONE API
    call per ticker, via MassiveClient.get_news) and upsert them into the
    ticker_news_documents table in Lakebase.

    Body (optional JSON): {"tickers": ["AAPL", "MSFT"], "limit": 50}
    Defaults to DEFAULT_NEWS_TICKERS when no tickers are supplied.
    """
    ensure_news_table()
    client = MassiveClient()

    body = request.json if request.is_json else {}
    tickers = body.get("tickers") or DEFAULT_NEWS_TICKERS
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    limit = int(body.get("limit", 50))

    total = 0
    for ticker in tickers:
        if not _TICKER_RE.match(ticker):
            continue
        articles = client.get_news(ticker, limit=limit)
        total += _upsert_news_batch(ticker, articles)

    return jsonify({"synced": total, "tickers": tickers})


@app.route("/watchlist", methods=["GET"])
def get_watchlist():
    """Return the current user's watchlist symbols, with their last known price."""
    ensure_watchlist_table()
    email = _current_user_email()
    rows = lakebase.run_query(
        f"SELECT symbol, email, latest_price, updated_at FROM {WATCHLIST_TABLE_NAME} "
        f"WHERE email = %s ORDER BY symbol ASC",
        (email,),
    )
    return jsonify(rows)


@app.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    """
    Fetch the latest price for a single stock symbol from Massive using
    exactly ONE API call (see MassiveClient.get_latest_price), then add/
    update that symbol on the watchlist in Lakebase.
    """
    ensure_watchlist_table()

    if request.is_json:
        symbol = request.json.get("symbol", "")
    else:
        symbol = request.form.get("symbol", "")

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""

    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    client = MassiveClient()
    try:
        data = client.get_latest_price(symbol)  # <-- single API call, latest price only
    except requests.HTTPError:
        # Massive returns a 404/4xx for tickers it doesn't recognize.
        return jsonify({"error": f"Unknown ticker symbol: {symbol}"}), 400

    price = _extract_latest_price(data)
    if price is None:
        # No usable price in the response (e.g. delisted/invalid ticker
        # that still 200s with an empty result set) - don't add it.
        return jsonify({"error": f"No price data available for ticker: {symbol}"}), 400

    email = _current_user_email()

    lakebase.run_write(
        f"""
        INSERT INTO {WATCHLIST_TABLE_NAME} (symbol, email, latest_price, updated_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (symbol, email) DO UPDATE
            SET latest_price = EXCLUDED.latest_price,
                updated_at = EXCLUDED.updated_at
        """,
        (symbol, email, price),
    )

    return jsonify({"symbol": symbol, "email": email, "latest_price": price})


@app.route("/watchlist/<symbol>", methods=["DELETE"])
def delete_from_watchlist(symbol: str):
    """Remove a single symbol from the current user's watchlist."""
    ensure_watchlist_table()

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    email = _current_user_email()
    deleted = lakebase.run_write(
        f"DELETE FROM {WATCHLIST_TABLE_NAME} WHERE symbol = %s AND email = %s",
        (symbol, email),
    )

    if not deleted:
        return jsonify({"error": f"{symbol} is not on your watchlist"}), 404

    return jsonify({"symbol": symbol, "email": email, "deleted": True})


@app.route("/stock/<symbol>/details", methods=["GET"])
def get_stock_details(symbol):
    """
    Fetch detailed information and recent news for a specific stock symbol.
    """
    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    
    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400
    
    client = MassiveClient()
    details = {}
    news = []
    
    try:
        # Fetch ticker details
        details_data = client.get_ticker_details(symbol)
        details = details_data.get("results", {})
    except requests.HTTPError as e:
        logger.warning(f"Failed to fetch ticker details for {symbol}: {e}")
        # Continue even if details fail - we might still get news
    except Exception as e:
        logger.error(f"Unexpected error fetching ticker details for {symbol}: {e}")
    
    try:
        # Fetch recent news
        news_data = client.get_ticker_news(symbol, limit=5)
        news = news_data.get("results", [])
    except requests.HTTPError as e:
        logger.warning(f"Failed to fetch news for {symbol}: {e}")
        # Continue even if news fails
    except Exception as e:
        logger.error(f"Unexpected error fetching news for {symbol}: {e}")
    
    # Return what we have, even if partial
    if not details and not news:
        return jsonify({"error": f"No data available for {symbol}. This ticker may not be available in the API or may require a different subscription tier."}), 400
    
    return jsonify({
        "symbol": symbol,
        "details": details,
        "news": news
    })


@app.route("/vector-search", methods=["POST"])
def vector_search():
    """
    Perform semantic search against a Databricks Vector Search endpoint.
    
    Expects JSON body with:
    - query: The search query string
    - num_results: Number of results to return (optional, default 5)
    - endpoint_name: Vector search endpoint name (optional, from env)
    - index_name: Vector search index name (optional, from env)
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    num_results = int(request.json.get("num_results", 5))
    endpoint_name = request.json.get("endpoint_name") or os.environ.get("VECTOR_SEARCH_ENDPOINT")
    index_name = request.json.get("index_name") or os.environ.get("VECTOR_SEARCH_INDEX")
    
    if not endpoint_name:
        return jsonify({"error": "Vector search endpoint name not configured. Set VECTOR_SEARCH_ENDPOINT environment variable."}), 400
    
    if not index_name:
        return jsonify({"error": "Vector search index name not configured. Set VECTOR_SEARCH_INDEX environment variable."}), 400
    
    email = _current_user_email()
    logger.info(f"Vector search query from {email}: {query!r}")
    
    try:
        vsc = VectorSearchClient(workspace_client=_w)
        
        # Get the index
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        
        # Perform similarity search
        results = index.similarity_search(
            query_text=query,
            columns=["id", "text", "metadata"],
            num_results=num_results
        )
        
        # Extract results from the response
        result_data = results.get("result", {}).get("data_array", [])
        
        # Format the response
        formatted_results = []
        for item in result_data:
            formatted_results.append({
                "id": item[0] if len(item) > 0 else None,
                "text": item[1] if len(item) > 1 else None,
                "metadata": item[2] if len(item) > 2 else None,
                "score": item[3] if len(item) > 3 else None
            })
        
        return jsonify({
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results,
            "endpoint": endpoint_name,
            "index": index_name
        })
        
    except Exception as e:
        logger.exception(f"Vector search failed for query {query!r}")
        return jsonify({"error": f"Vector search failed: {str(e)}"}), 500


def _extract_latest_price(data: dict) -> float | None:
    """Pull the trade price out of the Massive 'previous close' response shape.

    The /v2/aggs/ticker/{symbol}/prev endpoint returns "results" as a LIST
    containing a single aggregate bar (not a dict), e.g.:
        {"status": "OK", "resultsCount": 1, "results": [{"c": 148.845, ...}]}
    Previously this code treated "results" as a dict, so isinstance(results, dict)
    was always False for this endpoint's real shape and the price silently
    resolved to None. Unwrap the list here, and check "status"/"resultsCount"
    so invalid tickers (empty results) are detected instead of "succeeding"
    with a null price.

    Adjust the key lookup here if the real Massive API returns a different
    field name for the traded/close price.
    """
    if not isinstance(data, dict):
        return None
    if data.get("status") not in (None, "OK") or data.get("resultsCount") == 0:
        return None
    results = data.get("results", data)
    if isinstance(results, list):
        results = results[0] if results else None
    if isinstance(results, dict):
        for key in ("c", "p", "price", "last_price", "vw"):
            if key in results:
                return results[key]
    return None


def _upsert_batch(items: list[dict]) -> int:
    """Upsert a batch of Massive API items into Lakebase, one statement per row.

    For very large batches, consider psycopg2.extras.execute_values for
    higher throughput instead of per-row execute calls.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (id, payload, synced_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (str(item.get("id")), _json.dumps(item)),
                )
                count += 1
            conn.commit()
    return count


def _upsert_news_batch(ticker: str, articles: list[dict]) -> int:
    """Upsert news articles for a single ticker into the news documents table.

    Flattens the top-level "insights" sentiment entry that matches this
    ticker (if present) into its own columns so the Spark notebook can read
    plain text columns instead of parsing JSONB for the common case.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for article in articles:
                sentiment = None
                sentiment_reasoning = None
                for insight in article.get("insights", []) or []:
                    if insight.get("ticker") == ticker:
                        sentiment = insight.get("sentiment")
                        sentiment_reasoning = insight.get("sentiment_reasoning")
                        break

                publisher = article.get("publisher") or {}
                cur.execute(
                    f"""
                    INSERT INTO {NEWS_TABLE_NAME} (
                        id, ticker, title, description, author, article_url,
                        publisher_name, keywords, sentiment, sentiment_reasoning,
                        published_utc, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET ticker = EXCLUDED.ticker,
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            author = EXCLUDED.author,
                            article_url = EXCLUDED.article_url,
                            publisher_name = EXCLUDED.publisher_name,
                            keywords = EXCLUDED.keywords,
                            sentiment = EXCLUDED.sentiment,
                            sentiment_reasoning = EXCLUDED.sentiment_reasoning,
                            published_utc = EXCLUDED.published_utc,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        str(article.get("id")),
                        ticker,
                        article.get("title", ""),
                        article.get("description"),
                        article.get("author"),
                        article.get("article_url"),
                        publisher.get("name"),
                        _json.dumps(article.get("keywords", [])),
                        sentiment,
                        sentiment_reasoning,
                        article.get("published_utc"),
                        _json.dumps(article),
                    ),
                )
                count += 1
            conn.commit()
    return count


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")