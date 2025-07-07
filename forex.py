forex.py
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import webbrowser
import os

# --- 1. Data Loading and Preparation ---

def load_and_prepare_data(csv_path):
    """Loads and prepares the forex data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
        return None

    try:
        df['Localtime'] = pd.to_datetime(df['Localtime'], utc=True)
    except (ValueError, TypeError):
        df['Localtime'] = df['Localtime'].str.replace(r'\.\d{3} GMT[+-]\d{4}$', '', regex=True)
        df['Localtime'] = pd.to_datetime(df['Localtime'], dayfirst=True)

    df.rename(columns={"Localtime": "Date"}, inplace=True)
    df.set_index("Date", inplace=True)
    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": int})
    return df

# --- 2. The "Adaptive Volatility Breakout" Strategy ---

class AdaptiveBreakoutStrategy(Strategy):
    """
    A strategy that trades breakouts from low-volatility "squeezes".
    - It identifies squeezes using Bollinger Bands and Keltner Channels.
    - It uses MACD to confirm momentum in the direction of the breakout.
    - It adapts its take-profit target based on recent win/loss streaks.
    """
    # --- Strategy Parameters ---
    bb_period = 20
    bb_std_dev = 2.0
    kc_period = 20
    kc_atr_multiplier = 1.5
    atr_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    base_sl_atr = 2.0
    base_tp_atr = 2.0

    # --- Self-Contained Indicator Helper Functions ---
    def EMA(self, data, period):
        return pd.Series(data).ewm(span=period, adjust=False).mean()

    def ATR(self, df, period):
        df_atr = pd.DataFrame(index=df.index)
        df_atr['h_l'] = df['High'] - df['Low']
        df_atr['h_pc'] = abs(df['High'] - df['Close'].shift())
        df_atr['l_pc'] = abs(df['Low'] - df['Close'].shift())
        tr = df_atr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        return self.EMA(tr, period)

    def BANDS(self, data, period, std_dev):
        sma = pd.Series(data).rolling(period).mean()
        std = pd.Series(data).rolling(period).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        return upper, lower

    def KELTNER(self, df, period, atr_multiplier):
        ema = self.EMA(df['Close'], period)
        atr = self.ATR(df, period)
        upper = ema + (atr * atr_multiplier)
        lower = ema - (atr * atr_multiplier)
        return upper, lower

    def MACD(self, data, fast, slow, signal):
        ema_fast = self.EMA(data, fast)
        ema_slow = self.EMA(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.EMA(macd_line, signal)
        return macd_line, signal_line

    def init(self):
        self.bb_upper, self.bb_lower = self.I(self.BANDS, self.data.Close, self.bb_period, self.bb_std_dev)
        self.kc_upper, self.kc_lower = self.I(self.KELTNER, self.data.df, self.kc_period, self.kc_atr_multiplier)
        self.macd_line, self.macd_signal_line = self.I(self.MACD, self.data.Close, self.macd_fast, self.macd_slow, self.macd_signal)
        self.atr = self.I(self.ATR, self.data.df, self.atr_period)

        self.trade_history = []

    def next(self):
        if self.position:
            return

        price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # --- Adaptive Risk Logic ---
        tp_multiplier = self.base_tp_atr
        if len(self.trade_history) >= 2:
            if all(res == 1 for res in self.trade_history):
                tp_multiplier = 3.0  # Widen TP after 2 wins
            elif all(res == -1 for res in self.trade_history):
                tp_multiplier = 1.0  # Tighten TP after 2 losses

        # --- Squeeze Condition ---
        squeeze_on = (self.bb_lower[-1] > self.kc_lower[-1]) and (self.bb_upper[-1] < self.kc_upper[-1])

        if squeeze_on:
            # --- LONG ENTRY: Breakout above upper band with bullish momentum ---
            if price > self.bb_upper[-1] and self.macd_line[-1] > self.macd_signal_line[-1]:
                sl = price - atr_value * self.base_sl_atr
                tp = price + atr_value * tp_multiplier
                self.buy(sl=sl, tp=tp)

            # --- SHORT ENTRY: Breakout below lower band with bearish momentum ---
            elif price < self.bb_lower[-1] and self.macd_line[-1] < self.macd_signal_line[-1]:
                sl = price + atr_value * self.base_sl_atr
                tp = price - atr_value * tp_multiplier
                self.sell(sl=sl, tp=tp)

    def on_trade(self, trade):
        if trade.is_closed:
            result = 1 if trade.pl > 0 else -1
            self.trade_history.append(result)
            if len(self.trade_history) > 2:
                self.trade_history.pop(0)

# --- 3. HTML Report and CSV Generation ---

def generate_report_and_data(stats, commission_pips, filename="backtest_dashboard.html"):
    """Generates a polished HTML dashboard and simplified CSVs from the backtest results."""

    trades_df = stats['_trades']
    equity_curve_df = stats['_equity_curve']

    def format_stat(value, type='default'):
        if pd.isna(value) or (isinstance(value, (float, int)) and not np.isfinite(value)): return "N/A"
        if type == 'percent': return f"{value:.2f}%"
        if type == 'money': return f"CHF {value:,.2f}"
        if isinstance(value, pd.Timedelta):
            total_seconds = value.total_seconds()
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{int(days)}d {int(hours)}h {int(minutes)}m"
        if isinstance(value, (int, np.integer)): return f"{value:,}"
        if isinstance(value, (float, np.floating)): return f"{value:.2f}"
        return str(value)

    simplified_trades = pd.DataFrame()
    if not trades_df.empty:
        simplified_trades = trades_df[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL']].copy()
        simplified_trades['Duration (Hours)'] = round((pd.to_datetime(trades_df['ExitTime']) - pd.to_datetime(trades_df['EntryTime'])).dt.total_seconds() / 3600, 2)
        simplified_trades.columns = ['Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', 'P&L (CHF)', 'Duration (Hours)']

    if not trades_df.empty:
        winning_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] < 0]
        avg_win_val = winning_trades['PnL'].mean()
        avg_loss_val = losing_trades['PnL'].mean()
        win_loss_ratio = abs(avg_win_val / avg_loss_val) if avg_loss_val != 0 else 'inf'

        win_streak, loss_streak, max_win_streak, max_loss_streak = 0, 0, 0, 0
        for pnl in trades_df['PnL']:
            if pnl > 0:
                win_streak += 1; loss_streak = 0
            else:
                loss_streak += 1; win_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
            max_loss_streak = max(max_loss_streak, loss_streak)
    else:
        winning_trades, losing_trades = pd.DataFrame(), pd.DataFrame()
        avg_win_val = avg_loss_val = win_loss_ratio = np.nan
        max_win_streak = max_loss_streak = 0

    detailed_stats_data = {
        'Metric': [
            'Total Net Profit', 'Starting => Finishing Balance', 'Total Trades', 'Total Paid Fees',
            'Max Drawdown', 'Annual Return', 'Expectancy', 'Avg Win', 'Avg Loss', 'Ratio Avg Win / Avg Loss',
            'Win-rate', 'Avg Holding Time', 'Winning Trades Avg Holding Time', 'Losing Trades Avg Holding Time',
            'Sharpe Ratio', 'Calmar Ratio', 'Sortino Ratio', 'Winning Streak', 'Losing Streak',
            'Largest Winning Trade', 'Largest Losing Trade'
        ],
        'Value': [
            f"{format_stat(stats.get('Equity Final [$]', 0) - stats.get('Equity Start [$]', 10000), 'money')} ({format_stat(stats.get('Return [%]'), 'percent')})",
            f"{format_stat(stats.get('Equity Start [$]', 10000), 'money')} => {format_stat(stats.get('Equity Final [$]', 0), 'money')}",
            format_stat(stats.get('# Trades')),
            f"{format_stat(stats.get('Commissions [$]'), 'money')} ({commission_pips} pips)",
            format_stat(stats.get('Max. Drawdown [%]'), 'percent'),
            format_stat(stats.get('Return (Ann.) [%]'), 'percent'),
            f"{format_stat(stats.get('Avg. Trade [$]', 0), 'money')} ({format_stat(stats.get('Expectancy [%]'), 'percent')})",
            format_stat(avg_win_val, 'money'),
            format_stat(avg_loss_val, 'money'),
            format_stat(win_loss_ratio),
            format_stat(stats.get('Win Rate [%]'), 'percent'),
            format_stat(stats.get('Avg. Trade Duration')),
            format_stat(winning_trades['Duration'].mean()) if not winning_trades.empty else 'N/A',
            format_stat(losing_trades['Duration'].mean()) if not losing_trades.empty else 'N/A',
            format_stat(stats.get('Sharpe Ratio')),
            format_stat(stats.get('Calmar Ratio')),
            format_stat(stats.get('Sortino Ratio')),
            format_stat(max_win_streak),
            format_stat(max_loss_streak),
            format_stat(winning_trades['PnL'].max(), 'money') if not winning_trades.empty else 'N/A',
            format_stat(losing_trades['PnL'].min(), 'money') if not losing_trades.empty else 'N/A'
        ],
        'Description': [
            'The final financial result of all trades, and its percentage return on initial capital.',
            'The account balance at the beginning and end of the backtest.',
            'The total number of trades executed throughout the backtest.',
            'The total brokerage commission paid for all trades.',
            'The largest peak-to-trough decline in account equity; a key indicator of risk.',
            'The geometric average annual rate of return.',
            'The average profit or loss per trade, including its percentage expectancy.',
            'The average monetary gain from all winning trades.',
            'The average monetary loss from all losing trades.',
            'The ratio of the average win to the average loss. Higher is better.',
            'The percentage of total trades that were profitable.',
            'The average length of time a trade was held.',
            'The average holding time specifically for winning trades.',
            'The average holding time specifically for losing trades.',
            'Measures risk-adjusted return. (>1 Good, >2 Great, >3 Excellent)',
            'Measures return relative to drawdown. (>1 Good, >3 Excellent)',
            'Similar to Sharpe, but only considers downside volatility. (>2 Excellent)',
            'The highest number of winning trades in a row.',
            'The highest number of losing trades in a row.',
            'The value of the single most profitable trade.',
            'The value of the single largest losing trade.'
        ]
    }
    simplified_stats_df = pd.DataFrame(detailed_stats_data)

    # --- HTML Generation ---
    chart_labels = equity_curve_df.index.strftime('%d.%m.%Y %H:%M').tolist()
    chart_data = equity_curve_df['Equity'].round(2).tolist()

    stats_table_html = ""
    for i, row in simplified_stats_df.iterrows():
        stats_table_html += f"""
        <tr class="border-b border-gray-700 hover:bg-gray-800">
            <td class="py-3 px-4 font-semibold text-gray-300 text-sm">{row['Metric']}</td>
            <td class="py-3 px-4 text-right font-mono text-white text-sm">{row['Value']}</td>
            <td class="py-3 px-4 text-gray-400 italic text-xs">{row['Description']}</td>
        </tr>
        """

    trades_html_rows = ""
    if not simplified_trades.empty:
        for _, row in simplified_trades.iterrows():
            pnl_color = "text-green-400" if row['P&L (CHF)'] > 0 else "text-red-400"
            trades_html_rows += f"""
            <tr class="border-b border-gray-700 text-right font-mono text-xs hover:bg-gray-800">
                <td class="py-2 px-3 text-left text-gray-300">{row['Entry Time']}</td>
                <td class="py-2 px-3 text-left text-gray-300">{row['Exit Time']}</td>
                <td class="py-2 px-3 text-gray-300">{row['Duration (Hours)']:.2f}</td>
                <td class="py-2 px-3 text-gray-300">{row['Entry Price']:.5f}</td>
                <td class="py-2 px-3 text-gray-300">{row['Exit Price']:.5f}</td>
                <td class="py-2 px-3 font-semibold {pnl_color}">{row['P&L (CHF)']:.2f}</td>
            </tr>
            """
    else:
        trades_html_rows = "<tr><td colspan='6' class='text-center py-4 text-gray-400'>No trades were executed.</td></tr>"


    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Backtest Performance Analysis</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #111827; }}
            .font-mono {{ font-family: 'Roboto Mono', monospace; }}
            .card {{ background-color: #1f2937; border: 1px solid #374151; border-radius: 0.75rem; }}
        </style>
    </head>
    <body class="text-gray-300">
        <div class="container mx-auto p-4 sm:p-6 lg:p-8">
            <header class="mb-8 text-center">
                <h1 class="text-3xl font-bold text-white">Backtest Performance Analysis</h1>
                <p class="text-lg text-gray-400 mt-1">Strategy: Adaptive Volatility Breakout</p>
            </header>

            <div class="card p-4 sm:p-6 mb-8">
                <h2 class="text-xl font-bold mb-4 text-white">Performance Metrics</h2>
                <div class="overflow-x-auto">
                    <table class="w-full">
                        <tbody>{stats_table_html}</tbody>
                    </table>
                </div>
            </div>

            <div class="card p-4 sm:p-6 mb-8">
                <h2 class="text-xl font-bold mb-4 text-white">Equity Curve</h2>
                <div style="height: 400px;"><canvas id="equityCurveChart"></canvas></div>
            </div>

            <div class="card p-4 sm:p-6">
                <h2 class="text-xl font-bold mb-4 text-white">Trade Log</h2>
                <div class="overflow-x-auto">
                    <table class="w-full">
                        <thead class="bg-gray-800">
                            <tr class="text-xs text-left text-gray-400 uppercase tracking-wider">
                                <th class="py-3 px-3 font-semibold">Entry Time</th>
                                <th class="py-3 px-3 font-semibold">Exit Time</th>
                                <th class="py-3 px-3 font-semibold text-right">Duration (Hours)</th>
                                <th class="py-3 px-3 font-semibold text-right">Entry Price</th>
                                <th class="py-3 px-3 font-semibold text-right">Exit Price</th>
                                <th class="py-3 px-3 font-semibold text-right">P&L (CHF)</th>
                            </tr>
                        </thead>
                        <tbody>{trades_html_rows}</tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            const ctx = document.getElementById('equityCurveChart').getContext('2d');
            Chart.defaults.color = '#9ca3af';
            new Chart(ctx, {{
                type: 'line',
                data: {{ labels: {chart_labels}, datasets: [{{ label: 'Equity', data: {chart_data}, borderColor: 'rgb(99, 102, 241)', backgroundColor: 'rgba(99, 102, 241, 0.2)', borderWidth: 2, pointRadius: 0, tension: 0.1, fill: true }}] }},
                options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ ticks: {{ callback: value => 'CHF ' + value.toLocaleString(), color: '#9ca3af' }}, grid: {{ color: 'rgba(255, 255, 255, 0.1)' }} }}, x: {{ ticks: {{ maxRotation: 0, autoSkip: true, maxTicksLimit: 15, color: '#9ca3af' }}, grid: {{ color: 'rgba(255, 255, 255, 0.1)' }} }} }} }}
            }});
        </script>
    </body>
    </html>
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_template)

    return simplified_trades, simplified_stats_df

# --- 4. Main Execution Block ---

if __name__ == '__main__':
    DATA_PATH = "/home/kg/Documents/forex/data/USDCHF_Candlestick_1_Hour_BID_06.07.2015-28.06.2025.csv"
    df = load_and_prepare_data(DATA_PATH)

    if df is not None:
        spread_pips = 1.1
        pip_value = 0.0001
        commission_per_side = (spread_pips * pip_value) / 2

        bt = Backtest(df, AdaptiveBreakoutStrategy, cash=10_000, commission=commission_per_side, exclusive_orders=True)

        print("Running backtest with Adaptive Volatility Breakout Strategy...")
        stats = bt.run()
        print("\nBacktest Complete.")
        print("---")
        print("STATISTICS REPORT:")
        print(stats)
        print("---")

        report_filename = "backtest_dashboard.html"
        simplified_trades_df, simplified_stats_df = generate_report_and_data(stats, spread_pips, report_filename)
        print(f"✅ Polished dashboard saved to '{report_filename}'")

        # Save focused CSV files
        try:
            if not simplified_trades_df.empty:
                simplified_trades_df.to_csv("trades_summary.csv", index=False)
                print("📝 Simplified trades log saved to trades_summary.csv")

            if not simplified_stats_df.empty:
                simplified_stats_df.to_csv("stats_summary.csv", index=False)
                print("📝 Simplified stats report saved to stats_summary.csv")
        except Exception as e:
            print(f"Could not save summary CSV files: {e}")

        try:
            webbrowser.open('file://' + os.path.realpath(report_filename))
        except Exception as e:
            print(f"Could not automatically open the report. Please open it manually: {e}")
