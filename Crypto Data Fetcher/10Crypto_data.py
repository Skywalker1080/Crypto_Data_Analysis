import yfinance as yf
import pandas as pd
import numpy as np

# Data Fetching
crypto_list = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'SOL-USD', 'TRX-USD']
data = yf.download(crypto_list, period="1mo", interval="1d")

returns = data['Adj Close'].pct_change()

benchmark = returns['BTC-USD']

# Greek Calculations
def calculate_beta(returns, benchmark):
    covariance = returns.cov(benchmark)
    variance = benchmark.var()
    return covariance / variance

def calculate_alpha(returns, benchmark, beta):
    return returns.mean() - beta * benchmark.mean()

def calculate_omega_ratio(returns, benchmark, threshold=0):
    excess_returns = returns - benchmark
    positive_returns = np.maximum(excess_returns - threshold, 0)
    negative_returns = np.maximum(threshold - excess_returns, 0)
    return positive_returns.sum() / negative_returns.sum()


metrics = pd.DataFrame(index=crypto_list, columns=['Returns', 'Alpha', 'Beta', 'Omega Ratio'])

for coin in crypto_list:
    coin_returns = returns[coin].dropna()
    metrics.loc[coin, 'Returns'] = coin_returns.mean()
    
    if coin != 'BTC-USD':
        beta = calculate_beta(coin_returns, benchmark)
        alpha = calculate_alpha(coin_returns, benchmark, beta)
        omega_ratio = calculate_omega_ratio(coin_returns, benchmark)
        
        metrics.loc[coin, 'Alpha'] = alpha
        metrics.loc[coin, 'Beta'] = beta
        metrics.loc[coin, 'Omega Ratio'] = omega_ratio
    else:
        metrics.loc[coin, ['Alpha', 'Beta', 'Omega Ratio']] = [0, 1, 1]  # for BTC


rolling_window = 7  # 7-day rolling window
rolling_metrics = pd.DataFrame(index=returns.index, columns=pd.MultiIndex.from_product([crypto_list, ['Alpha', 'Beta', 'Omega Ratio']]))

for coin in crypto_list:
    if coin != 'BTC-USD':
        coin_returns = returns[coin].dropna()
        
        rolling_beta = coin_returns.rolling(window=rolling_window).cov(benchmark) / benchmark.rolling(window=rolling_window).var()
        rolling_alpha = coin_returns.rolling(window=rolling_window).mean() - rolling_beta * benchmark.rolling(window=rolling_window).mean()
        
        excess_returns = coin_returns - benchmark
        positive_returns = np.maximum(excess_returns, 0).rolling(window=rolling_window).sum()
        negative_returns = np.maximum(-excess_returns, 0).rolling(window=rolling_window).sum()
        rolling_omega_ratio = positive_returns / negative_returns
        
        rolling_metrics[(coin, 'Alpha')] = rolling_alpha
        rolling_metrics[(coin, 'Beta')] = rolling_beta
        rolling_metrics[(coin, 'Omega Ratio')] = rolling_omega_ratio
    else:
        rolling_metrics[(coin, 'Alpha')] = 0
        rolling_metrics[(coin, 'Beta')] = 1
        rolling_metrics[(coin, 'Omega Ratio')] = 1

# CSV Data Save 
metrics.to_csv('monthly_crypto_metrics.csv')
rolling_metrics.to_csv('rolling_crypto_metrics.csv')


# HTML conversion
def create_html_table(df, title):
    html = f"<h2>{title}</h2>"
    html += df.to_html(classes='table table-striped', border=1)
    return html

returns_table = create_html_table(metrics, "Cryptocurrency Metrics (1 Month)")
rolling_metrics_table = create_html_table(rolling_metrics.tail(10), "Rolling Last 10 days")

html_output = f"""
<html>
<head>
    <style>
        table {{
            border-collapse: collapse;
            border-color: black;
            border-width: 3px;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }}
        th, td {{
            padding: 12px 15px;
        }}
        thead tr {{
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }}
        tbody tr {{
            border-bottom: 1px solid #dddddd;
        }}
        tbody tr:nth-of-type(even) {{
            background-color: #f3f3f3;
        }}
        tbody tr:last-of-type {{
            border-bottom: 2px solid #009879;
        }}
    </style>
</head>
body>
    {returns_table}
    {rolling_metrics_table}
</body>
</html>
"""

with open('crypto_analysis.html', 'w') as f:
    f.write(html_output)

print("Results saved in 'crypto_analysis.html'")
print("Data saved to monthly_crypto_metrics.csv and rolling_crypto_metrics.csv")