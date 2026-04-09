# Interactive Portfolio Analytics

A Streamlit web app for constructing and analyzing equity portfolios in real time. Built for FDA II — Project 2.

## Features

- **Returns & EDA** — Summary statistics, $10k cumulative wealth index, return distribution (histogram + normal fit, Q-Q plot)
- **Risk Analysis** — Rolling annualized volatility, drawdown analysis with max-drawdown metric, Sharpe and Sortino ratios
- **Correlation** — Correlation heatmap, rolling pairwise correlation, covariance matrix
- **Portfolio Construction** — Equal-weight, GMV, tangency (max Sharpe), and custom (slider-based) portfolios; risk contribution decomposition; efficient frontier with Capital Allocation Line
- **Window Sensitivity** — Side-by-side comparison of GMV and tangency portfolios computed over different lookback windows

## Run Locally

```bash
uv add streamlit yfinance pandas plotly scipy numpy
uv run streamlit run app.py
```

Or with pip:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Methodology

- Adjusted close prices from Yahoo Finance via `yfinance`
- Simple (arithmetic) returns throughout
- Annualization: ×252 for mean, ×√252 for volatility
- Sortino ratio uses downside deviation (returns below daily risk-free rate)
- Optimization via `scipy.optimize.minimize` (SLSQP) with bounds (0,1) and sum-to-1 constraint — no short selling
- Tangency portfolio found by minimizing the negative Sharpe ratio
- Efficient frontier built by constrained optimization at each target return (not random simulation)
- S&P 500 (`^GSPC`) used as benchmark only — never included in optimization
