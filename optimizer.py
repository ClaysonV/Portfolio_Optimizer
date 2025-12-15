import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.optimize as sco
import seaborn as sns
import statsmodels.api as sm
import pandas_datareader.data as web
import warnings
import datetime

# --- CONFIGURATION ---
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style="whitegrid", context="talk", palette="deep")

# Global Style Settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'sans-serif'

def get_data(tickers, start_date="2020-01-01", end_date=datetime.date.today()):
    print(f"\n[INFO] Grabbing data for: {', '.join(tickers)}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)['Close']
    if data.empty:
        raise ValueError("[ERROR] Download failed. Check internet or ticker symbols.")
    data = data.dropna(axis=1, how='all')
    return data

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std

def optimize_portfolio(weights, mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    result = sco.minimize(negative_sharpe, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def get_fama_french_factors(start_date, end_date):
    try:
        ds = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start_date, end=end_date)
        factors = ds[0] / 100 
        factors.index = pd.to_datetime(factors.index)
        if factors.index.tz is not None:
             factors.index = factors.index.tz_localize(None)
        return factors
    except Exception as e:
        print(f"[WARNING] Could not fetch Fama-French data: {e}")
        return None

# --- MAIN EXECUTION ---
print("-" * 50)
print("PORTFOLIO OPTIMIZER: PROFESSIONAL EDITION")
print("-" * 50)

try:
    user_input = input("Enter Tickers (e.g. NVDA BTC-USD GLD): ").strip().upper()
except KeyboardInterrupt:
    exit()

if not user_input:
    tickers = ['NVDA', 'BTC-USD', 'GLD', 'TLT'] 
else:
    tickers = user_input.split()

try:
    cash_input = input("Enter Starting Capital (Default $10,000): ").strip()
    initial_capital = float(cash_input) if cash_input else 10000.0
except ValueError:
    initial_capital = 10000.0

bench_ticker = ['SPY']
all_tickers = list(set(tickers + bench_ticker))

try:
    # 1. Get Data
    prices = get_data(all_tickers)
    asset_prices = prices[tickers].dropna()
    bench_price = prices['SPY'].dropna()
    
    start_date = asset_prices.index[0]
    bench_price = bench_price[bench_price.index >= start_date]
    
    print(f"\n[INFO] Simulation Date Range: {start_date.date()} to {asset_prices.index[-1].date()}")
    
    # 2. Returns & Stats
    log_returns = np.log(asset_prices / asset_prices.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    num_assets = len(tickers)
    
    print(f"[INFO] Running 5,000 Monte Carlo simulations...")
    
    # 3. Monte Carlo Engine
    num_portfolios = 5000
    all_weights = np.zeros((num_portfolios, num_assets))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)

    for i in range(num_portfolios):
        weights = np.array(np.random.random(num_assets))
        weights = weights / np.sum(weights)
        all_weights[i, :] = weights
        ret_arr[i], vol_arr[i] = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe_arr[i] = (ret_arr[i] - 0.04) / vol_arr[i]

    # 4. Optimizer
    print("[INFO] Solving for Maximum Sharpe Ratio...")
    optimal_result = optimize_portfolio(all_weights, mean_returns, cov_matrix)
    opt_weights = optimal_result.x
    opt_ret, opt_vol = portfolio_performance(opt_weights, mean_returns, cov_matrix)

    # 5. Backtest
    normalized_assets = (asset_prices / asset_prices.iloc[0]) 
    normalized_bench = (bench_price / bench_price.iloc[0]) * initial_capital
    weighted_curve = (normalized_assets * opt_weights).sum(axis=1) * initial_capital

    final_port_val = weighted_curve.iloc[-1]
    final_bench_val = normalized_bench.iloc[-1]
    port_profit = final_port_val - initial_capital
    bench_profit = final_bench_val - initial_capital
    port_total_ret = (final_port_val - initial_capital) / initial_capital
    bench_total_ret = (final_bench_val - initial_capital) / initial_capital

    # Stats
    total_days = len(weighted_curve)
    years = total_days / 252
    port_cagr = (final_port_val / initial_capital) ** (1/years) - 1
    bench_cagr = (final_bench_val / initial_capital) ** (1/years) - 1
    port_vol = weighted_curve.pct_change().std() * np.sqrt(252)
    bench_vol = normalized_bench.pct_change().std() * np.sqrt(252)
    port_sharpe = (port_cagr - 0.04) / port_vol
    bench_sharpe = (bench_cagr - 0.04) / bench_vol
    
    def get_max_drawdown(series):
        return ((series - series.cummax()) / series.cummax()).min()
    
    port_dd = get_max_drawdown(weighted_curve)
    bench_dd = get_max_drawdown(normalized_bench)

    # Factors
    print("[INFO] Analyzing Factor Exposure...")
    ff_factors = get_fama_french_factors("2020-01-01", "2024-01-01")
    ff_beta_mkt = "N/A"
    ff_beta_smb = "N/A" 
    ff_beta_hml = "N/A"

    if ff_factors is not None:
        port_daily_ret = weighted_curve.pct_change().dropna()
        if port_daily_ret.index.tz is not None:
            port_daily_ret.index = port_daily_ret.index.tz_localize(None)
        joined_data = pd.concat([port_daily_ret, ff_factors], axis=1).dropna()
        if not joined_data.empty:
            joined_data.rename(columns={0: 'Portfolio'}, inplace=True)
            Y = joined_data['Portfolio'] - joined_data['RF'] 
            X = joined_data[['Mkt-RF', 'SMB', 'HML']]
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()
            ff_beta_mkt = f"{model.params['Mkt-RF']:.2f}"
            ff_beta_smb = f"{model.params['SMB']:.2f}"
            ff_beta_hml = f"{model.params['HML']:.2f}"

    # --- REPORT ---
    print("\n" + "="*65)
    print(f"WEALTH SIMULATION REPORT")
    print(f"Date Range: {start_date.date()} -> {asset_prices.index[-1].date()}")
    print(f"Starting Capital: ${initial_capital:,.2f}")
    print("="*65)
    print(f"{'METRIC':<25} | {'PORTFOLIO':>15} | {'S&P 500':>15}")
    print("-" * 61)
    print(f"{'End Value':<25} | ${final_port_val:>14,.2f} | ${final_bench_val:>14,.2f}")
    print(f"{'Net Profit':<25} | ${port_profit:>14,.2f} | ${bench_profit:>14,.2f}")
    print(f"{'Total Return %':<25} | {port_total_ret:>14.2%} | {bench_total_ret:>14.2%}")
    print("-" * 61)
    print(f"{'CAGR (Annual Growth)':<25} | {port_cagr:>14.2%} | {bench_cagr:>14.2%}")
    print(f"{'Volatility (Risk)':<25} | {port_vol:>14.2%} | {bench_vol:>14.2%}")
    print(f"{'Sharpe Ratio':<25} | {port_sharpe:>14.2f} | {bench_sharpe:>14.2f}")
    print(f"{'Max Drawdown (Crash)':<25} | {port_dd:>14.2%} | {bench_dd:>14.2%}")
    print("="*65)
    print(f"FACTOR EXPOSURE (Fama-French)")
    print("-" * 61)
    print(f"{'Market Beta (Risk)':<25} | {ff_beta_mkt:>14}")
    print(f"{'Size Factor (SMB)':<25} | {ff_beta_smb:>14}")
    print(f"{'Value Factor (HML)':<25} | {ff_beta_hml:>14}")
    print("="*65)

    diff = final_port_val - final_bench_val
    if final_port_val > final_bench_val:
        print(f"VERDICT: OUTPERFORMED MARKET BY ${diff:,.2f}")
    else:
        print(f"VERDICT: UNDERPERFORMED MARKET BY ${abs(diff):,.2f}")
    print("="*65)

    # --- PLOTTING ---

    # 1. Efficient Frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.4, s=15, label='Simulations')
    plt.colorbar(sc, label='Sharpe Ratio')
    ax.scatter(opt_vol, opt_ret, c='#d62728', s=200, edgecolors='white', linewidth=2, marker='*', label='Optimal')
    ax.set_title('Efficient Frontier', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Risk (Volatility)')
    ax.set_ylabel('Annualized Return')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    plt.tight_layout()

    # 2. Asset Allocation (Donut Chart) with Detailed Table
    active_indices = [i for i, w in enumerate(opt_weights) if w > 0.01]
    active_weights = opt_weights[active_indices]
    active_tickers = [tickers[i] for i in active_indices]
    
    colors = sns.color_palette("pastel", len(active_tickers))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Donut Chart Logic: pctdistance moves the % labels, width creates the hole
    wedges, texts, autotexts = ax1.pie(active_weights, labels=None, autopct='%1.1f%%', 
                                       startangle=140, colors=colors, pctdistance=0.85,
                                       wedgeprops=dict(width=0.4, edgecolor='w'))
    
    # Style the percentage text
    plt.setp(autotexts, size=10, weight="bold", color="darkgrey")
    
    ax1.set_title('Optimized Asset Allocation', fontsize=16, fontweight='bold')

    # Table Logic
    table_data = []
    total_dollar = 0
    for i, ticker in enumerate(active_tickers):
        weight_pct = active_weights[i]
        dollar_val = initial_capital * weight_pct
        total_dollar += dollar_val
        table_data.append([ticker, f"{weight_pct:.1%}", f"${dollar_val:,.2f}"])
        
    ax2.axis('off')
    table = ax2.table(cellText=table_data, colLabels=["Asset", "Alloc %", "Value ($)"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5) 
    
    # Legend below the Donut
    ax1.legend(wedges, active_tickers, title="Assets", loc="center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    plt.tight_layout()

    # 3. Backtest
    fig, ax = plt.subplots(figsize=(12, 7))
    for i in range(50):
        random_weights = all_weights[np.random.randint(0, num_portfolios)]
        random_curve = (normalized_assets * random_weights).sum(axis=1) * initial_capital
        ax.plot(random_curve, color='grey', alpha=0.05, linewidth=1)

    ax.plot(weighted_curve, color='#0066cc', linewidth=2.5, label='Your Portfolio')
    ax.plot(normalized_bench, color='#333333', linestyle='--', linewidth=2, label='S&P 500 Benchmark')
    
    ax.set_title(f'Wealth Simulation: ${initial_capital:,.0f} Investment', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Portfolio Value')
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.tight_layout()

    # 4. Global Heatmap (All Assets, No Numbers)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(log_returns.corr(), annot=False, cmap='RdBu_r', center=0, 
                linewidths=0.5, linecolor='white', square=True, cbar_kws={"shrink": .8})
    ax.set_title('Global Asset Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    print("[SUCCESS] Dashboard generated.")
    plt.show()

except Exception as e:
    print(f"\n[ERROR]: {e}")