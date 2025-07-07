import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import webbrowser
import os
import re

# --- 1. Data Loading and Preparation ---
# This function is responsible for loading the trading data from a CSV file
# and preparing it for the backtesting engine.
def load_and_prepare_data(csv_path):
    """
    Loads and prepares the forex data from a CSV file.

    Args:
        csv_path (str): The file path to the CSV data.

    Returns:
        pandas.DataFrame: A DataFrame formatted for the backtesting library,
                          or None if the file is not found.
    """
    # Try to load the CSV file. If it doesn't exist, print an error and exit.
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
        return None

    # Make the date column detection more flexible. It will check for common names.
    # This resolves the KeyError when switching between files with different column names.
    possible_date_columns = ['Localtime', 'Gmt time', 'Date', 'Time']
    date_col = None
    for col in possible_date_columns:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        print(f"Error: Could not find a recognizable date column in {csv_path}")
        return None

    # The 'Localtime' or other found date column can have different formats.
    # This tries to convert it to a standard datetime format.
    try:
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
    except (ValueError, TypeError):
        # If the standard conversion fails, this regex removes timezone information
        # and then tries to convert again, assuming a day-first format.
        df[date_col] = df[date_col].str.replace(r'\.\d{3} GMT[+-]\d{4}$', '', regex=True)
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

    # The backtesting library expects a 'Date' column for the index.
    df.rename(columns={date_col: "Date"}, inplace=True)
    df.set_index("Date", inplace=True)

    # Ensure the core pricing data (Open, High, Low, Close, Volume) are the correct data types.
    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": int})
    return df

# --- 2. The "Ichimoku Sync" Strategy ---
# This is the core class that defines our trading logic. It inherits from the Strategy
# class provided by the backtesting library.
class IchimokuSyncStrategy(Strategy):
    """
    A high-confirmation trend-following strategy (Version 5 - Static Optimized).

    This strategy uses a fixed set of parameters that were found to be optimal
    for EURUSD. It requires multiple indicators to be in agreement before entering a trade.

    Entry Triggers:
    - Volatility Filter: Ensures the market is moving and not flat.
    - ADX Trend Filter: Ensures a strong trend is in place.
    - Ichimoku Cloud: The price must be clearly above (for longs) or below (for shorts) the cloud.
    - Chikou Span: Confirms the trend from a historical perspective.
    - Parabolic SAR: Must align with the trade direction.
    - Bollinger Bands: The price must break out of the bands, signaling volatility.
    - Momentum: A simple momentum check must agree with the trade direction.

    Exit Trigger:
    - The exit is dynamic: the trade is closed when the price crosses back over the
      Ichimoku Kijun-sen (Base Line), signaling that the trend may be losing strength.
    """
    # --- Strategy Parameters ---
    ### MODIFICATION ###
    # These parameters are now hardcoded with the optimal values found from the previous run.

    # Ichimoku Parameters
    tenkan_period = 12       # Conversion Line Period (kept constant)
    kijun_period = 20        # Base Line Period (OPTIMIZED)
    senkou_period = 60       # Leading Span B Period (kept constant)

    # Bollinger Bands Parameters (standard values)
    bb_period = 20           # Moving Average Period
    bb_std_dev = 2.0         # Standard Deviations

    # Parabolic SAR Parameters (standard values)
    sar_af = 0.02            # Acceleration Factor
    sar_max_af = 0.2         # Maximum Acceleration Factor

    # Momentum Parameter
    momentum_period = 14     # Lookback period for momentum calculation

    # ADX Parameters
    adx_period = 14          # Period for ADX calculation
    adx_threshold = 25       # Minimum ADX value to trade (OPTIMIZED)

    # Volatility Filter Parameters
    spread_pips = 1.1        # The spread of the instrument in pips
    pip_value = 0.0001       # The value of one pip
    volatility_multiplier = 3.0 # A multiplier to set the minimum required price movement

    # --- Self-Contained Indicator Helper Functions ---
    # These functions calculate the values for each of our technical indicators.
    # They are defined as separate methods to keep the main logic clean.

    def ADX(self, df, period):
        """Calculates the Average Directional Index (ADX)."""
        df_adx = pd.DataFrame(index=df.index)
        df_adx['H-L'] = df['High'] - df['Low']
        df_adx['H-pC'] = np.abs(df['High'] - df['Close'].shift(1))
        df_adx['L-pC'] = np.abs(df['Low'] - df['Close'].shift(1))
        df_adx['TR'] = df_adx[['H-L', 'H-pC', 'L-pC']].max(axis=1)

        df_adx['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
        df_adx['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)

        atr = df_adx['TR'].rolling(period).mean()
        di_plus = 100 * (df_adx['DMplus'].rolling(period).mean() / atr)
        di_minus = 100 * (df_adx['DMminus'].rolling(period).mean() / atr)

        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        return adx

    def BANDS(self, data, period, std_dev):
        """Calculates Bollinger Bands."""
        sma = pd.Series(data).rolling(period).mean()
        std = pd.Series(data).rolling(period).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        return upper, lower

    def MOMENTUM(self, data, period):
        """Calculates the Momentum indicator."""
        return pd.Series(data).diff(period)

    def SAR(self, df, af, max_af):
        """Calculates the Parabolic SAR."""
        # This function has been made more robust to prevent runtime warnings.
        high, low = df['High'], df['Low']
        sar = low.copy()
        trend = 1
        ep = high.iloc[0]
        accel = af

        for i in range(2, len(df)):
            prev_sar = sar.iloc[i-1]

            # Check for non-finite values before calculation to prevent overflow.
            if not np.isfinite(prev_sar) or not np.isfinite(ep):
                sar.iloc[i] = prev_sar
                continue

            ep_diff = ep - prev_sar
            sar_change = accel * ep_diff

            if trend == 1:
                sar.iloc[i] = prev_sar + sar_change
                if low.iloc[i] < sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    accel = af
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        accel = min(accel + af, max_af)
                    sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
            else:
                sar.iloc[i] = prev_sar - sar_change
                if high.iloc[i] > sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    accel = af
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        accel = min(accel + af, max_af)
                    sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
        return sar

    def ICHIMOKU(self, df, tenkan_p, kijun_p, senkou_p):
        """Calculates the Ichimoku Kinko Hyo components."""
        high = df['High']
        low = df['Low']
        tenkan_sen = (high.rolling(tenkan_p).max() + low.rolling(tenkan_p).min()) / 2
        kijun_sen = (high.rolling(kijun_p).max() + low.rolling(kijun_p).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_p)
        senkou_span_b = ((high.rolling(senkou_p).max() + low.rolling(senkou_p).min()) / 2).shift(kijun_p)
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

    def init(self):
        """
        This method is called once at the start of the backtest.
        It's used to initialize all the indicators we'll need.
        """
        self.adx = self.I(self.ADX, self.data.df, self.adx_period)
        self.bb_upper, self.bb_lower = self.I(self.BANDS, self.data.Close, self.bb_period, self.bb_std_dev)
        self.momentum = self.I(self.MOMENTUM, self.data.Close, self.momentum_period)
        self.sar = self.I(self.SAR, self.data.df, self.sar_af, self.sar_max_af)
        self.tenkan, self.kijun, self.senkou_a, self.senkou_b = self.I(self.ICHIMOKU, self.data.df, self.tenkan_period, self.kijun_period, self.senkou_period)

        self.min_volatility = self.volatility_multiplier * self.spread_pips * self.pip_value

    def next(self):
        """
        This method is called for each bar of data in the history.
        It contains the core trading logic (entry and exit rules).
        """
        price = self.data.Close[-1]

        # --- EXIT LOGIC ---
        if self.position.is_long and crossover(self.kijun, self.data.Close):
            self.position.close()
            return

        if self.position.is_short and crossover(self.data.Close, self.kijun):
            self.position.close()
            return

        # --- ENTRY LOGIC ---
        if self.position:
            return

        # --- FILTERS ---
        volatility_check_1 = self.data.High[-1] - self.data.Low[-1] > self.min_volatility
        volatility_check_2 = self.data.High[-2] - self.data.Low[-2] > self.min_volatility
        if not (volatility_check_1 and volatility_check_2):
            return

        is_trending = self.adx[-1] > self.adx_threshold
        if not is_trending:
            return

        # --- INDICATOR CONDITIONS ---
        above_kumo = price > self.senkou_a[-1] and price > self.senkou_b[-1]
        below_kumo = price < self.senkou_a[-1] and price < self.senkou_b[-1]

        chikou_bullish = self.data.Close[-1] > self.data.Close[-self.kijun_period]
        chikou_bearish = self.data.Close[-1] < self.data.Close[-self.kijun_period]

        sar_bullish = price > self.sar[-1]
        sar_bearish = price < self.sar[-1]

        momentum_bullish = self.momentum[-1] > 0
        momentum_bearish = self.momentum[-1] < 0

        bb_breakout_bullish = price > self.bb_upper[-1]
        bb_breakout_bearish = price < self.bb_lower[-1]

        # --- SYNCHRONIZED ENTRY SIGNALS ---
        is_long_signal = (above_kumo and
                          chikou_bullish and
                          sar_bullish and
                          momentum_bullish and
                          bb_breakout_bullish)

        is_short_signal = (below_kumo and
                           chikou_bearish and
                           sar_bearish and
                           momentum_bearish and
                           bb_breakout_bearish)

        if is_long_signal:
            self.buy()
        elif is_short_signal:
            self.sell()


# --- 3. HTML Report and CSV Generation ---
# The function now accepts a 'quote_currency' argument to make the report dynamic.
def generate_report_and_data(stats, commission_pips, quote_currency, filename="backtest_dashboard.html"):
    """
    Generates a polished HTML dashboard and simplified CSVs from the backtest results.
    """
    trades_df = stats['_trades']
    equity_curve_df = stats['_equity_curve']

    # Helper function to format numbers nicely for the report.
    def format_stat(value, type='default'):
        if pd.isna(value) or (isinstance(value, (float, int)) and not np.isfinite(value)): return "N/A"
        if type == 'percent': return f"{value:.2f}%"
        # Use the dynamic quote_currency instead of a hardcoded "CHF".
        if type == 'money': return f"{quote_currency} {value:,.2f}"
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
        # Correctly merge the equity curve data with the trades data to get the equity at the end of each trade.
        temp_trades_df = trades_df.copy()
        temp_trades_df['ExitTime'] = pd.to_datetime(temp_trades_df['ExitTime'])
        trades_with_equity = pd.merge_asof(
            temp_trades_df.sort_values('ExitTime'),
            equity_curve_df[['Equity']].sort_index(),
            left_on='ExitTime',
            right_index=True,
            direction='backward'
        )

        # Select and rename columns for the final report.
        simplified_trades = trades_with_equity[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'Equity']].copy()
        simplified_trades['Duration (Hours)'] = round((pd.to_datetime(simplified_trades['ExitTime']) - pd.to_datetime(simplified_trades['EntryTime'])).dt.total_seconds() / 3600, 2)
        # Dynamically set the column names based on the quote currency.
        pnl_col_name = f'P&L ({quote_currency})'
        equity_col_name = f'Ending Equity ({quote_currency})'
        simplified_trades.columns = ['Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', pnl_col_name, equity_col_name, 'Duration (Hours)']

    # Calculate additional performance metrics for the report.
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

    # Create a dictionary of all stats to be displayed in the HTML report.
    detailed_stats_data = {
        'Metric': ['Total Net Profit', 'Starting => Finishing Balance', 'Total Trades', 'Total Paid Fees', 'Max Drawdown', 'Annual Return', 'Expectancy', 'Avg Win', 'Avg Loss', 'Ratio Avg Win / Avg Loss', 'Win-rate', 'Avg Holding Time', 'Winning Trades Avg Holding Time', 'Losing Trades Avg Holding Time', 'Sharpe Ratio', 'Calmar Ratio', 'Sortino Ratio', 'Winning Streak', 'Losing Streak', 'Largest Winning Trade', 'Largest Losing Trade'],
        'Value': [f"{format_stat(stats.get('Equity Final [$]', 0) - stats.get('Equity Start [$]', 10000), 'money')} ({format_stat(stats.get('Return [%]'), 'percent')})", f"{format_stat(stats.get('Equity Start [$]', 10000), 'money')} => {format_stat(stats.get('Equity Final [$]', 0), 'money')}", format_stat(stats.get('# Trades')), f"{format_stat(stats.get('Commissions [$]'), 'money')} ({commission_pips} pips)", format_stat(stats.get('Max. Drawdown [%]'), 'percent'), format_stat(stats.get('Return (Ann.) [%]'), 'percent'), f"{format_stat(stats.get('Avg. Trade [$]', 0), 'money')} ({format_stat(stats.get('Expectancy [%]'), 'percent')})", format_stat(avg_win_val, 'money'), format_stat(avg_loss_val, 'money'), format_stat(win_loss_ratio), format_stat(stats.get('Win Rate [%]'), 'percent'), format_stat(stats.get('Avg. Trade Duration')), format_stat(winning_trades['Duration'].mean()) if not winning_trades.empty else 'N/A', format_stat(losing_trades['Duration'].mean()) if not losing_trades.empty else 'N/A', format_stat(stats.get('Sharpe Ratio')), format_stat(stats.get('Calmar Ratio')), format_stat(stats.get('Sortino Ratio')), format_stat(max_win_streak), format_stat(max_loss_streak), format_stat(winning_trades['PnL'].max(), 'money') if not winning_trades.empty else 'N/A', format_stat(losing_trades['PnL'].min(), 'money') if not losing_trades.empty else 'N/A'],
        'Description': ['The final financial result of all trades, and its percentage return on initial capital.', 'The account balance at the beginning and end of the backtest.', 'The total number of trades executed throughout the backtest.', 'The total brokerage commission paid for all trades.', 'The largest peak-to-trough decline in account equity; a key indicator of risk.', 'The geometric average annual rate of return.', 'The average profit or loss per trade, including its percentage expectancy.', 'The average monetary gain from all winning trades.', 'The average monetary loss from all losing trades.', 'The ratio of the average win to the average loss. Higher is better.', 'The percentage of total trades that were profitable.', 'The average length of time a trade was held.', 'The average holding time specifically for winning trades.', 'The average holding time specifically for losing trades.', 'Measures risk-adjusted return. (>1 Good, >2 Great, >3 Excellent)', 'Measures return relative to drawdown. (>1 Good, >3 Excellent)', 'Similar to Sharpe, but only considers downside volatility. (>2 Excellent)', 'The highest number of winning trades in a row.', 'The highest number of losing trades in a row.', 'The value of the single most profitable trade.', 'The value of the single largest losing trade.']
    }
    simplified_stats_df = pd.DataFrame(detailed_stats_data)

    # --- HTML Generation ---
    # This section constructs the HTML file for the report using f-strings.
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
        pnl_col_name = f'P&L ({quote_currency})'
        equity_col_name = f'Ending Equity ({quote_currency})'
        for _, row in simplified_trades.iterrows():
            pnl_color = "text-green-400" if row[pnl_col_name] > 0 else "text-red-400"
            trades_html_rows += f"""
            <tr class="border-b border-gray-700 text-right font-mono text-xs hover:bg-gray-800">
                <td class="py-2 px-3 text-left text-gray-300">{row['Entry Time']}</td>
                <td class="py-2 px-3 text-left text-gray-300">{row['Exit Time']}</td>
                <td class="py-2 px-3 text-gray-300">{row['Duration (Hours)']:.2f}</td>
                <td class="py-2 px-3 text-gray-300">{row['Entry Price']:.5f}</td>
                <td class="py-2 px-3 text-gray-300">{row['Exit Price']:.5f}</td>
                <td class="py-2 px-3 font-semibold {pnl_color}">{row[pnl_col_name]:.2f}</td>
                <td class="py-2 px-3 font-semibold text-gray-300">{row[equity_col_name]:.2f}</td>
            </tr>
            """
    else:
        trades_html_rows = "<tr><td colspan='7' class='text-center py-4 text-gray-400'>No trades were executed.</td></tr>"


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
                <p class="text-lg text-gray-400 mt-1">Strategy: Ichimoku Sync Strategy (v5 - Static Params)</p>
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
                                <!-- Dynamically set the table headers based on the quote currency. -->
                                <th class="py-3 px-3 font-semibold text-right">P&L ({quote_currency})</th>
                                <th class="py-3 px-3 font-semibold text-right">Ending Equity ({quote_currency})</th>
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
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            ticks: {{
                                // Dynamically format the Y-axis of the chart with the correct currency.
                                callback: value => '{quote_currency} ' + value.toLocaleString(),
                                color: '#9ca3af'
                            }},
                            grid: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                        }},
                        x: {{
                            ticks: {{ maxRotation: 0, autoSkip: true, maxTicksLimit: 15, color: '#9ca3af' }},
                            grid: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                        }}
                    }}
                }}
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
    # This section now uses a simple input() prompt instead of command-line arguments.

    # Ask the user for the currency pair.
    pair_input = input("Enter the currency pair to backtest (e.g., USDCHF): ")

    # Construct the filename and the full path.
    # Assumes the data files are in a subdirectory named 'data'.
    data_directory = "/home/kg/Documents/forex/data"
    # The user's input is now used directly, respecting its case, instead of forcing it to lowercase.
    filename = f"{pair_input}.csv"
    DATA_PATH = os.path.join(data_directory, filename)

    print(f"Loading data from: {DATA_PATH}")

    df = load_and_prepare_data(DATA_PATH)

    if df is not None:
        # Extract the currency pair from the filename to use in reporting.
        pair_name = pair_input.upper()
        quote_currency = pair_name[3:]

        print(f"Detected Currency Pair: {pair_name}")
        print(f"Using Quote Currency: {quote_currency}")

        # Set commission to 0 for this test run. In a real scenario, this would be
        # set to your broker's actual commission rate.
        commission_per_side = 0.0
        commission_pips_for_report = 0

        # Initialize the backtesting engine with the data, our strategy,
        # starting cash, and commission settings.
        bt = Backtest(df, IchimokuSyncStrategy, cash=10_000, commission=commission_per_side, exclusive_orders=True)

        ### MODIFICATION ###
        # Run a single backtest with the hardcoded optimal parameters.
        print("Running backtest with static optimized parameters...")
        stats = bt.run()

        print("\nBacktest Complete.")
        print("---")
        print("STRATEGY STATS:")
        print(stats)
        print("---")

        # Create dynamic filenames for the output files.
        report_filename = f"ichimoku_sync_v5_{pair_name}_static_dashboard.html"
        trades_csv_filename = f"ichimoku_v5_{pair_name}_static_trades.csv"
        stats_csv_filename = f"ichimoku_v5_{pair_name}_static_stats.csv"

        simplified_trades_df, simplified_stats_df = generate_report_and_data(stats, commission_pips_for_report, quote_currency, report_filename)
        print(f"✅ Polished dashboard for static run saved to '{report_filename}'")

        # Save the simplified trade and stats logs to CSV files for further analysis.
        try:
            if not simplified_trades_df.empty:
                simplified_trades_df.to_csv(trades_csv_filename, index=False)
                print(f"📝 Simplified trades log saved to {trades_csv_filename}")

            if not simplified_stats_df.empty:
                simplified_stats_df.to_csv(stats_csv_filename, index=False)
                print(f"📝 Simplified stats report saved to {stats_csv_filename}")
        except Exception as e:
            print(f"Could not save summary CSV files: {e}")

        # Try to automatically open the generated HTML report in a web browser.
        try:
            webbrowser.open('file://' + os.path.realpath(report_filename))
        except Exception as e:
            print(f"Could not automatically open the report. Please open it manually: {e}")
