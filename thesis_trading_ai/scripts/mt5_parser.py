import os
import sys
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# Fix project root path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
REPORT_DIR = PROJECT_ROOT / "results" / "live" / "mt5_reports"
OUT_DIR = PROJECT_ROOT / "results" / "live" / "mt5_standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_mt5_html(file_path):
    print(f"Parsing {file_path.name}...")
    # MT5 reports are usually UTF-16LE
    with open(file_path, 'r', encoding='utf-16') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    trades = []
    
    # MT5 report structure has trades in tables
    rows = soup.find_all('tr')
    
    for row in rows:
        tds = row.find_all('td')
        # We look for rows that look like closed trades
        # Based on inspection, these usually have many columns and a specific date format in the first cell
        if len(tds) < 10:
            continue
            
        try:
            open_time = tds[0].text.strip()
            ticket = tds[1].text.strip()
            symbol = tds[2].text.strip()
            trade_type = tds[3].text.strip()
            
            # Skip rows that don't have a valid date in the first column
            if not open_time.replace('.', '').replace(' ', '').replace(':', '').isdigit():
                continue

            # Identify data cells after the 'Comment' cell
            # In some reports, Comment is 'hidden' or standard.
            # Volume is usually after type/comment
            volume = float(tds[-9].text.replace(' ', '').replace(',', ''))
            open_price = float(tds[-8].text.replace(' ', '').replace(',', ''))
            sl = float(tds[-7].text.replace(' ', '').replace(',', ''))
            tp = float(tds[-6].text.replace(' ', '').replace(',', ''))
            close_time = tds[-5].text.strip()
            close_price = float(tds[-4].text.replace(' ', '').replace(',', ''))
            commission = float(tds[-3].text.replace(' ', '').replace(',', ''))
            swap = float(tds[-2].text.replace(' ', '').replace(',', ''))
            profit = float(tds[-1].text.replace(' ', '').replace(',', ''))
            
            trades.append({
                'timestamp_open': open_time,
                'ticket': ticket,
                'symbol': symbol,
                'position': trade_type,
                'volume': volume,
                'entry_price': open_price,
                'sl': sl,
                'tp': tp,
                'timestamp_close': close_time,
                'exit_price': close_price,
                'commission': commission,
                'swap': swap,
                'pnl_actual': profit,
                'pnl_net': profit + commission + swap
            })
        except (ValueError, IndexError) as e:
            # Skip rows that don't fit the expected trade data format (e.g. headers, separators)
            continue
            
    df = pd.DataFrame(trades)
    if not df.empty:
        # Clean up timestamps to standard format
        df['timestamp_open'] = pd.to_datetime(df['timestamp_open'].str.replace('.', '-'))
        df['timestamp_close'] = pd.to_datetime(df['timestamp_close'].str.replace('.', '-'))
        # Calculate return percentage based on entry/exit if not provided
        df['return_pct_actual'] = ((df['exit_price'] - df['entry_price']) / df['entry_price'] * 100)
        # Flip for shorts
        df.loc[df['position'].str.lower().str.contains('sell'), 'return_pct_actual'] *= -1
        
    return df

def main():
    models = {
        "ReportHistory-LSTM.html": "bilstm",
        "ReportHistory-Transformer.html": "transformer"
    }
    
    for filename, model_name in models.items():
        file_path = REPORT_DIR / filename
        if file_path.exists():
            df = parse_mt5_html(file_path)
            if not df.empty:
                out_path = OUT_DIR / f"{model_name}_mt5_trades.csv"
                df.to_csv(out_path, index=False)
                print(f"Saved {len(df)} actual trades for {model_name} to {out_path}")
            else:
                print(f"No trades parsed from {filename}")
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
