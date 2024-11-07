import yfinance as yf
import os

tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "ITC.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ASIANPAINT.NS", "MARUTI.NS", "HDFCBANK.NS",
    "ADANIGREEN.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS", "POWERGRID.NS"
]

os.makedirs("data/stocks", exist_ok=True)

def fetch_multiple_stocks_data(tickers, period='5y'):
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df['Ticker'] = ticker
        df.to_csv(f"data/stocks/{ticker}.csv")

if __name__ == "__main__":
    fetch_multiple_stocks_data(tickers)
