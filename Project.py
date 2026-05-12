# References:
# yfinance library: https://pypi.org/project/yfinance/
# GeeksforGeeks - Moving Averages: https://www.geeksforgeeks.org/pandas/how-to-calculate-moving-average-in-a-pandas-dataframe/
# QuantInsti - Stock Analysis in Python: https://www.quantinsti.com/articles/stock-market-data-analysis-python/
# DEV - Python for Stock Market Analysis Working with Moving Averages: https://dev.to/qviper/python-for-stock-market-analysis-working-with-moving-averages-1mmg
# Isolation Forest for Anomaly Detection: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

import yfinance as yf
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import IsolationForest
import sqlite3


# ── DATABASE SETUP ─────────────────────────────────────────────────────────────
# Source: Python SQLite3 Documentation: https://docs.python.org/3/library/sqlite3.html
# Source: Assisted by ChatGPT

def create_database():
    # Connect to or create a local database file called stock_history.db
    conn = sqlite3.connect("stock_history.db")
    cursor = conn.cursor()

    # Create a table to store search history if it doesn't already exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            latest_price REAL,
            avg_price REAL,
            high_price REAL,
            low_price REAL,
            trend TEXT,
            total_return REAL,
            search_date TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_history(stats):
    # Save the most recent search result into the database with today's date
    conn = sqlite3.connect("stock_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO search_history (ticker, latest_price, avg_price, high_price, low_price, trend, total_return, search_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, DATE('now'))
    """, (
        stats['ticker'],
        stats['latest'],
        stats['avg'],
        stats['high'],
        stats['low'],
        stats['trend'],
        stats['total_return']
    ))
    conn.commit()
    conn.close()

def view_history():
    # Read all past searches from the database and display them in a new window
    conn = sqlite3.connect("stock_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticker, latest_price, avg_price, high_price, low_price, trend, total_return, search_date
        FROM search_history
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    # Create a new popup window to show the history
    history_window = tk.Toplevel(root)
    history_window.title("Search History")
    history_window.geometry("780x400")

    tk.Label(history_window, text="Past Searches", font=("Arial", 13, "bold")).pack(pady=(10, 4))

    # Create a frame to hold the table
    table_frame = tk.Frame(history_window)
    table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

    # Add a scrollbar in case there are many rows
    scrollbar = tk.Scrollbar(table_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Use a Listbox to display each row of history as a formatted line
    listbox = tk.Listbox(table_frame, font=("Courier New", 9),
                         yscrollcommand=scrollbar.set, width=110)
    listbox.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    # Add a header row so the user knows what each column means
    header = f"{'Date':<12} {'Ticker':<8} {'Price':>8} {'Avg':>8} {'High':>8} {'Low':>8} {'Trend':<12} {'Return':>8}"
    listbox.insert(tk.END, header)
    listbox.insert(tk.END, "-" * 90)

    # If no searches have been saved yet, show a message
    if not rows:
        listbox.insert(tk.END, "No search history yet.")
    else:
        # Loop through each saved row and format it into a readable line
        for row in rows:
            ticker, latest, avg, high, low, trend, total_return, date = row
            line = f"{date:<12} {ticker:<8} ${latest:>7} ${avg:>7} ${high:>7} ${low:>7} {trend:<12} {total_return:>7}%"
            listbox.insert(tk.END, line)


# ── ANALYTICS ─────────────────────────────────────────────────────────────────

# Saul Hernandez - Analytics and Data Processing
def run_analysis(ticker):
    # This function takes a stock symbol and returns price data and key numbers.

    # --- DATA DOWNLOAD ---
    # Go to Yahoo Finance and grab the last 6 months of daily prices.
    # auto_adjust=True fixes the prices if the stock ever split.
    # progress=False stops a loading bar from showing up.
    # Source: https://pypi.org/project/yfinance/
    df = yf.download(ticker, period="6mo", interval="1d",
                     auto_adjust=True, progress=False)

    # Only keep the closing price column and remove any rows with missing data
    df = df[['Close']].dropna()
    df.columns = ['Close']

    # --- MOVING AVERAGES ---
    # SMA_30 averages the last 30 days of prices.
    # SMA_60 averages the last 60 days of prices.
    # EMA_20 averages 20 days but cares more about recent prices.
    # Source: https://www.geeksforgeeks.org/pandas/how-to-calculate-moving-average-in-a-pandas-dataframe/
    df['SMA_30'] = df['Close'].rolling(30).mean()
    df['SMA_60'] = df['Close'].rolling(60).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()

    # --- GROWTH RATES ---
    # Daily_Return_% shows how much the price changed each day as a percentage.
    # Total_Return_% shows the total gain or loss from the very first day.
    # Source: https://www.quantinsti.com/articles/stock-market-data-analysis-python/
    df['Daily_Return_%'] = df['Close'].pct_change() * 100
    df['Total_Return_%'] = ((1 + df['Close'].pct_change()).cumprod() - 1) * 100

    # --- ANOMALY DETECTION ---
    # Use Isolation Forest to detect unusual price movements in the stock.
    # It looks at the closing price and daily return together to find outliers.
    # contamination=0.05 means we expect about 5% of days to be unusual.
    # Source: Scikit-learn IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    # Adapted from AnomalyDetection.py (previous group project)
    df_clean = df.dropna().copy()
    features = df_clean[['Close', 'Daily_Return_%']]
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(features)

    # Mark each day as normal (1) or anomalous (-1)
    df_clean['anomaly'] = model.predict(features)

    # Pull out only the rows flagged as anomalies for highlighting on the chart
    anomalies = df_clean[df_clean['anomaly'] == -1]

    # --- TREND SIGNAL ---
    # Grab the most recent row of data after removing any remaining nulls.
    # Source: https://dev.to/qviper/python-for-stock-market-analysis-working-with-moving-averages-1mmg
    # Source: Help from ChatGPT
    p = df_clean.iloc[-1]

    # If the 30-day average is above the 60-day average, stock is in an uptrend.
    trend = "Uptrend" if p['SMA_30'] > p['SMA_60'] else ("Downtrend" if p['SMA_30'] < p['SMA_60'] else "Neutral")

    # --- STATS DICTIONARY ---
    # Package all key numbers into one dictionary for easy access in the GUI.
    # Source: ChatGPT
    stats = {
        "ticker"      : ticker,
        "latest"      : round(float(p['Close']), 2),
        "avg"         : round(float(df['Close'].mean()), 2),
        "high"        : round(float(df['Close'].max()), 2),
        "low"         : round(float(df['Close'].min()), 2),
        "sma_30"      : round(float(p['SMA_30']), 2),
        "sma_60"      : round(float(p['SMA_60']), 2),
        "ema_20"      : round(float(p['EMA_20']), 2),
        "daily_return": round(float(p['Daily_Return_%']), 2),
        "total_return": round(float(p['Total_Return_%']), 2),
        "trend"       : trend
    }

    return df_clean, stats, anomalies


# ── CHART ─────────────────────────────────────────────────────────────────────
# Source: Matplotlib Documentation: https://matplotlib.org/stable/api/pyplot_api.html
# Source: FigureCanvasTkAgg Embedding: https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html
# Source: Assisted by ChatGPT

def draw_chart(df, stats, anomalies):
    # Clear the previous chart before drawing a new one
    ax.clear()

    # Plot the closing price as a solid blue line
    ax.plot(df.index, df['Close'], label='Close Price', color='steelblue', linewidth=1.5)

    # Plot the three moving averages as thinner dashed lines
    ax.plot(df.index, df['SMA_30'], label='SMA 30', color='orange', linewidth=1, linestyle='--')
    ax.plot(df.index, df['SMA_60'], label='SMA 60', color='green', linewidth=1, linestyle='--')
    ax.plot(df.index, df['EMA_20'], label='EMA 20', color='purple', linewidth=1, linestyle='--')

    # Highlight anomaly days as red dots directly on the price line
    if not anomalies.empty:
        ax.scatter(anomalies.index, anomalies['Close'],
                   color='red', zorder=5, label='Anomaly', s=40)

    # Add chart labels and formatting
    ax.set_title(f"{stats['ticker']} — Last 6 Months", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc='upper left', fontsize=7)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    # Refresh the canvas so the new chart appears in the window
    canvas.draw()


# ── GUI LOGIC ─────────────────────────────────────────────────────────────────

def analyze():
    # Get the ticker the user typed in and convert to uppercase
    ticker = entry.get().strip().upper()

    if not ticker:
        messagebox.showwarning("Missing Ticker", "Please enter a stock ticker.")
        return

    # Update the status label while data is loading
    status_label.config(text="Fetching data, please wait...")
    root.update()

    try:
        # Run the full analysis and get back the data, stats, and anomalies
        df, stats, anomalies = run_analysis(ticker)

        # Save this search to the local database for history tracking
        save_to_history(stats)

        # Update the text display with all the key numbers
        result.set(
            f"Ticker:        {stats['ticker']}\n"
            f"Latest Price:  ${stats['latest']}\n"
            f"6-Mo Average:  ${stats['avg']}\n"
            f"6-Mo High:     ${stats['high']}\n"
            f"6-Mo Low:      ${stats['low']}\n"
            f"\n"
            f"SMA 30:        ${stats['sma_30']}\n"
            f"SMA 60:        ${stats['sma_60']}\n"
            f"EMA 20:        ${stats['ema_20']}\n"
            f"\n"
            f"Trend:         {stats['trend']}\n"
            f"Daily Return:  {stats['daily_return']}%\n"
            f"Total Return:  {stats['total_return']}%\n"
            f"\n"
            f"Anomalies:\n"
            f"  {len(anomalies)} unusual\n"
            f"  day(s) detected"
        )

        # Draw the chart with moving averages and anomaly highlights
        draw_chart(df, stats, anomalies)
        status_label.config(text=f"Showing results for {stats['ticker']}")

    except Exception as e:
        messagebox.showerror("Error", f"Could not fetch data:\n{e}")
        status_label.config(text="Error fetching data.")

def clear_all():
    # Clear the text entry box
    entry.delete(0, tk.END)

    # Reset the summary stats back to default text
    result.set("Results will appear here.")

    # Clear the chart area
    ax.clear()
    canvas.draw()

    # Reset the status label
    status_label.config(text="Enter a ticker symbol to begin.")

def clear_history():
    # Ask the user to confirm before permanently deleting all history
    confirm = messagebox.askyesno("Clear History", "Are you sure you want to delete all search history?")
    if not confirm:
        return

    # Connect to the database and delete every row from the search history table
    conn = sqlite3.connect("stock_history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM search_history")
    conn.commit()
    conn.close()

    # Let the user know the history was cleared
    status_label.config(text="Search history cleared.")


# ── WINDOW SETUP ──────────────────────────────────────────────────────────────

# Create the database file when the app starts
create_database()

# Build the main application window
root = tk.Tk()
root.title("Stock Analytics Dashboard")
root.geometry("1000x700")

# Title label at the top
tk.Label(root, text="Stock Analytics Dashboard", font=("Arial", 16, "bold")).pack(pady=(12, 2))

# Input row with label, text box, and button
input_frame = tk.Frame(root)
input_frame.pack(pady=4)

tk.Label(input_frame, text="Enter a ticker symbol:").pack(side=tk.LEFT, padx=5)
entry = tk.Entry(input_frame, font=("Arial", 12), width=14, justify="center")
entry.pack(side=tk.LEFT, padx=5)
entry.bind("<Return>", lambda _: analyze())
tk.Button(input_frame, text="Analyze", command=analyze, width=12).pack(side=tk.LEFT, padx=5)

# View History button opens a popup showing all past searches from the database
tk.Button(input_frame, text="View History", command=view_history, width=12).pack(side=tk.LEFT, padx=5)

# Clear button resets the screen back to its default state
tk.Button(input_frame, text="Clear", command=clear_all, width=8).pack(side=tk.LEFT, padx=5)

# Clear History button permanently deletes all saved searches from the database
tk.Button(input_frame, text="Clear History", command=clear_history, width=12).pack(side=tk.LEFT, padx=5)

# Status message shown below the input row
status_label = tk.Label(root, text="Enter a ticker symbol to begin.", fg="gray")
status_label.pack()

# Main content area split into left (stats) and right (chart)
content_frame = tk.Frame(root)
content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

# Left side — text stats display
left_frame = tk.Frame(content_frame, width=220)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
left_frame.pack_propagate(False)

tk.Label(left_frame, text="Summary", font=("Arial", 11, "bold")).pack(pady=(8, 4))
result = tk.StringVar(value="Results will appear here.")
tk.Label(left_frame, textvariable=result, font=("Courier New", 10),
         justify="left", padx=10, pady=6, anchor="w").pack(fill=tk.X)

# Right side — chart embedded inside the window
right_frame = tk.Frame(content_frame)
right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(figsize=(8, 4.5))
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Start the application
root.mainloop()
