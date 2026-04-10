"""
Interactive Portfolio Analytics Application
Financial Data Analytics II - Project 2

A Streamlit web app for constructing and analyzing equity portfolios in real time.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, probplot, norm
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TD = 252  # trading days per year
INITIAL = 10_000  # starting wealth for cumulative index


# ============================================================================
# DATA DOWNLOAD (cached)
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers: tuple, start: str, end: str):
    """
    Download adjusted close prices for tickers + S&P 500 benchmark.
    Returns (stock_prices, bench_prices, failed_tickers, warning_msg).
    """
    all_syms = list(tickers) + ["^GSPC"]
    failed = []
    warning_msg = ""

    try:
        raw = yf.download(
            all_syms,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        return None, None, list(tickers), f"Download failed: {e}"

    if raw is None or raw.empty:
        return None, None, list(tickers), "yfinance returned no data."

    # Pull the Close panel
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return None, None, list(tickers), "No 'Close' column in download."
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_syms[:1]

    # Identify tickers that failed entirely
    for t in all_syms:
        if t not in prices.columns or prices[t].dropna().empty:
            failed.append(t)

    # Remove failed columns
    prices = prices.drop(columns=[c for c in failed if c in prices.columns])

    if "^GSPC" in failed:
        return None, None, failed, "Could not download S&P 500 benchmark (^GSPC)."

    # Handle partial-data tickers: drop any with > 5% missing
    pct_missing = prices.isna().mean()
    too_sparse = pct_missing[pct_missing > 0.05].index.tolist()
    if too_sparse:
        warning_msg = (
            f"Dropped ticker(s) with >5% missing data: {', '.join(too_sparse)}"
        )
        prices = prices.drop(columns=too_sparse)
        failed.extend([t for t in too_sparse if t != "^GSPC"])

    # Truncate to overlapping date range (drop rows where any remaining col is NaN)
    prices = prices.dropna()

    if prices.empty or "^GSPC" not in prices.columns:
        return None, None, failed, "No overlapping data after cleaning."

    bench = prices[["^GSPC"]].rename(columns={"^GSPC": "SP500"})
    stocks = prices.drop(columns=["^GSPC"])

    return stocks, bench, failed, warning_msg


# ============================================================================
# CORE CALCULATIONS
# ============================================================================
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def drawdown(r: pd.Series) -> pd.Series:
    w = (1 + r).cumprod()
    return (w - w.cummax()) / w.cummax()


def sharpe_ratio(r: pd.Series, rf: float) -> float:
    vol = r.std() * np.sqrt(TD)
    if vol == 0:
        return np.nan
    return (r.mean() * TD - rf) / vol


def sortino_ratio(r: pd.Series, rf: float) -> float:
    rf_daily = rf / TD
    excess = r - rf_daily
    # Downside deviation: square only negative excess returns, normalize by TOTAL obs
    downside_sq = np.minimum(excess, 0.0) ** 2
    downside_std = np.sqrt(downside_sq.mean()) * np.sqrt(TD)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (r.mean() * TD - rf) / downside_std


def summary_stats(ret: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ann. Return": ret.mean() * TD,
            "Ann. Volatility": ret.std() * np.sqrt(TD),
            "Skewness": ret.apply(skew),
            "Kurtosis": ret.apply(kurtosis),
            "Min Daily": ret.min(),
            "Max Daily": ret.max(),
        }
    )


# ============================================================================
# PORTFOLIO MATH
# ============================================================================
def port_return(w, mean_ret):
    return float(np.dot(mean_ret, w) * TD)


def port_vol(w, cov):
    return float(np.sqrt(w @ cov @ w * TD))


def port_sharpe(w, mean_ret, cov, rf):
    v = port_vol(w, cov)
    if v == 0:
        return np.nan
    return (port_return(w, mean_ret) - rf) / v


def port_sortino(w, ret_df, rf):
    pr = ret_df @ w
    return sortino_ratio(pr, rf)


def port_dd(w, ret_df):
    return float(drawdown(ret_df @ w).min())


def portfolio_metrics(w, ret_df, rf):
    mean_ret = ret_df.mean().values
    cov = ret_df.cov().values
    return {
        "Ann. Return": port_return(w, mean_ret),
        "Ann. Volatility": port_vol(w, cov),
        "Sharpe": port_sharpe(w, mean_ret, cov, rf),
        "Sortino": port_sortino(w, ret_df, rf),
        "Max Drawdown": port_dd(w, ret_df),
    }


def optimize_gmv(ret_values: np.ndarray, n: int):
    """Global Minimum Variance portfolio. Cached on raw return array."""
    ret_df = pd.DataFrame(ret_values)
    cov = ret_df.cov().values
    w0 = np.full(n, 1 / n)
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 1)] * n
    res = minimize(
        lambda w: w @ cov @ w,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res.x if res.success else None


def optimize_tangency(ret_values: np.ndarray, n: int, rf: float):
    """Maximum Sharpe (tangency) portfolio.

    Returns (weights, warning_msg). If the problem is ill-posed (e.g., rf
    exceeds all asset returns) or SLSQP fails, falls back to GMV weights
    and returns a warning message.
    """
    ret_df = pd.DataFrame(ret_values)
    mean_ret = ret_df.mean().values
    cov = ret_df.cov().values
    w0 = np.full(n, 1 / n)
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 1)] * n

    # Guard: if every asset's annualized return is below rf, max-Sharpe is ill-posed.
    if (mean_ret * TD).max() <= rf:
        gmv = optimize_gmv(ret_values, n)
        return gmv, (
            "Risk-free rate exceeds every asset's annualized return; "
            "tangency portfolio is undefined. Showing GMV weights instead."
        )

    def neg_sharpe(w):
        v = port_vol(w, cov)
        if v == 0:
            return 1e6
        return -((port_return(w, mean_ret) - rf) / v)

    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        gmv = optimize_gmv(ret_values, n)
        return gmv, "Tangency optimizer failed to converge; showing GMV weights instead."
    return res.x, None


def compute_efficient_frontier(ret_values: np.ndarray, n: int, n_points: int = 60):
    """Constrained optimization at each target return level."""
    ret_df = pd.DataFrame(ret_values)
    mean_ret = ret_df.mean().values
    cov = ret_df.cov().values
    w0 = np.full(n, 1 / n)
    bounds = [(0, 1)] * n

    gmv_w = optimize_gmv(ret_values, n)
    if gmv_w is None:
        return np.array([]), np.array([])

    min_ret = port_return(gmv_w, mean_ret)
    max_ret = mean_ret.max() * TD
    if max_ret <= min_ret:
        return np.array([port_vol(gmv_w, cov)]), np.array([min_ret])

    targets = np.linspace(min_ret, max_ret, n_points)
    vols, rets = [], []
    for t in targets:
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, tt=t: port_return(w, mean_ret) - tt},
        ]
        res = minimize(
            lambda w: w @ cov @ w,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.success:
            vols.append(port_vol(res.x, cov))
            rets.append(t)
    return np.array(vols), np.array(rets)


def risk_contribution(w, cov):
    """Percentage risk contribution (PRC). Sums to 1."""
    w = np.asarray(w)
    port_var = float(w @ cov @ w)
    if port_var == 0:
        return np.zeros_like(w)
    mrc = cov @ w  # marginal contributions
    rc = w * mrc
    return rc / port_var


# ============================================================================
# SIDEBAR — INPUTS
# ============================================================================
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Configure your portfolio inputs below.")

st.sidebar.subheader("Tickers")
ticker_input = st.sidebar.text_input(
    "Enter 3–10 ticker symbols (comma-separated)",
    value="",
    placeholder="e.g., AAPL, MSFT, JPM, XOM, JNJ",
    help="Type stock ticker symbols separated by commas. Minimum 3, maximum 10.",
)

st.sidebar.subheader("Date Range")
today = date.today()
default_end = today
default_start = today - timedelta(days=365 * 5)
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input(
        "Start", value=default_start, max_value=today - timedelta(days=365 * 2)
    )
with col_d2:
    end_date = st.date_input("End", value=default_end, max_value=today)

st.sidebar.subheader("Risk-Free Rate")
rf_pct = st.sidebar.number_input(
    "Annualized risk-free rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.25,
    help="Used for Sharpe and Sortino ratio calculations.",
)
rf = rf_pct / 100.0

run_btn = st.sidebar.button("▶️ Run Analysis", type="primary", use_container_width=True)

with st.sidebar.expander("ℹ️ About / Methodology"):
    st.markdown(
        """
**Data:** Adjusted close prices from Yahoo Finance via `yfinance`.

**Returns:** Simple (arithmetic) daily returns, used consistently for all portfolio math.

**Annualization:** Mean × 252; volatility × √252.

**Risk-free rate:** User-specified annualized; converted to daily as `rf / 252`.

**Sharpe ratio:** `(annual return − rf) / annual volatility`.

**Sortino ratio:** `(annual return − rf) / annualized downside deviation`, where downside deviation uses only returns below the daily rf.

**Optimization:** `scipy.optimize.minimize` (SLSQP) with bounds (0, 1) and sum-to-1 constraint. No short selling.

**Tangency:** Minimizes negative Sharpe ratio.

**Efficient frontier:** Constrained minimization at each target return (not random simulation).

**Risk contribution:** `PRCᵢ = wᵢ · (Σw)ᵢ / σ²ₚ`, sums to 1.
        """
    )


# ============================================================================
# INPUT VALIDATION
# ============================================================================
def parse_tickers(s: str):
    if not s.strip():
        return []
    parts = [p.strip().upper() for p in s.replace(";", ",").replace(" ", ",").split(",")]
    return [p for p in parts if p]


tickers = parse_tickers(ticker_input)

st.title("📈 Interactive Portfolio Analytics")
st.caption(
    "Build and analyze equity portfolios with real-time data, risk analytics, and mean-variance optimization."
)

# Gate the rest of the app on valid inputs + button
if not run_btn and "data_loaded" not in st.session_state:
    st.info(
        "👈 Enter your tickers, date range, and risk-free rate in the sidebar, then click **Run Analysis** to begin."
    )
    st.stop()

if run_btn:
    # Validate
    if len(tickers) < 3:
        st.error(f"❌ Please enter at least 3 tickers. You entered {len(tickers)}.")
        st.stop()
    if len(tickers) > 10:
        st.error(f"❌ Please enter no more than 10 tickers. You entered {len(tickers)}.")
        st.stop()
    if len(set(tickers)) != len(tickers):
        st.error("❌ Duplicate tickers detected. Please enter unique symbols.")
        st.stop()
    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()
    days_span = (end_date - start_date).days
    if days_span < 365 * 2:
        st.error(
            f"❌ Date range must be at least 2 years. You selected {days_span / 365:.1f} years."
        )
        st.stop()

    # Download
    with st.spinner("Downloading price data from Yahoo Finance..."):
        stocks, bench, failed, warning_msg = download_prices(
            tuple(sorted(tickers)), start_date.isoformat(), end_date.isoformat()
        )

    if stocks is None:
        st.error(f"❌ Data download failed: {warning_msg}")
        if failed:
            st.error(f"Problem ticker(s): {', '.join(failed)}")
        st.stop()

    # Bad tickers among the user's selection
    bad_user_tickers = [t for t in tickers if t not in stocks.columns]
    if bad_user_tickers:
        st.warning(
            f"⚠️ The following ticker(s) could not be downloaded and were dropped: **{', '.join(bad_user_tickers)}**"
        )
    if warning_msg and "Dropped" in warning_msg:
        st.warning(f"⚠️ {warning_msg}")

    if len(stocks.columns) < 3:
        st.error(
            f"❌ After cleaning, only {len(stocks.columns)} valid ticker(s) remain. Need at least 3 to proceed."
        )
        st.stop()

    if len(stocks) < 250:
        st.error(
            f"❌ Only {len(stocks)} overlapping trading days available — not enough for meaningful analysis."
        )
        st.stop()

    # Persist in session state
    st.session_state["data_loaded"] = True
    st.session_state["stocks"] = stocks
    st.session_state["bench"] = bench
    st.session_state["rf"] = rf
    st.session_state["tickers_used"] = list(stocks.columns)

# Pull from session state
stocks = st.session_state["stocks"]
bench = st.session_state["bench"]
rf = st.session_state["rf"]
tickers_used = st.session_state["tickers_used"]

ret = compute_returns(stocks)
b_ret = compute_returns(bench).reindex(ret.index).dropna()
ret = ret.loc[b_ret.index]
n = len(tickers_used)
mean_ret = ret.mean().values
cov = ret.cov().values

st.success(
    f"✅ Loaded **{n}** ticker(s): {', '.join(tickers_used)}  |  "
    f"**{len(ret):,}** trading days  |  "
    f"{ret.index[0].date()} → {ret.index[-1].date()}"
)


# ============================================================================
# TABS
# ============================================================================
tab_eda, tab_risk, tab_corr, tab_port, tab_sens, tab_about = st.tabs(
    [
        "📊 Returns & EDA",
        "📉 Risk Analysis",
        "🔗 Correlation",
        "🎯 Portfolio Construction",
        "🔬 Window Sensitivity",
        "📖 About",
    ]
)

# ----------------------------------------------------------------------------
# TAB 1: RETURNS & EDA
# ----------------------------------------------------------------------------
with tab_eda:
    st.header("Returns & Exploratory Analysis")

    st.subheader("Summary Statistics")
    combined = pd.concat([ret, b_ret.rename(columns={"SP500": "S&P 500"})], axis=1)
    stats_df = summary_stats(combined)
    st.dataframe(
        stats_df.style.format(
            {
                "Ann. Return": "{:.2%}",
                "Ann. Volatility": "{:.2%}",
                "Skewness": "{:.3f}",
                "Kurtosis": "{:.3f}",
                "Min Daily": "{:.2%}",
                "Max Daily": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Cumulative Wealth Index — Growth of $10,000")
    selected_for_chart = st.multiselect(
        "Select stocks to display",
        options=tickers_used,
        default=tickers_used,
        key="wealth_select",
    )

    wi = INITIAL * (1 + ret).cumprod()
    bwi = INITIAL * (1 + b_ret["SP500"]).cumprod()

    fig = go.Figure()
    for t in selected_for_chart:
        fig.add_trace(go.Scatter(x=wi.index, y=wi[t], name=t, mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=bwi.index,
            y=bwi.values,
            name="S&P 500",
            mode="lines",
            line=dict(color="black", dash="dash", width=3),
        )
    )
    fig.update_layout(
        title="Wealth Index: Growth of $10,000",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Return Distribution")
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        dist_stock = st.selectbox("Select a stock", tickers_used, key="dist_stock")
    with col_d2:
        dist_view = st.radio(
            "View", ["Histogram + Normal Fit", "Q-Q Plot"], horizontal=True, key="dist_view"
        )

    series = ret[dist_stock].dropna()
    if dist_view == "Histogram + Normal Fit":
        mu, sigma = series.mean(), series.std()
        x = np.linspace(series.min(), series.max(), 200)
        pdf = norm.pdf(x, mu, sigma)
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=series,
                nbinsx=60,
                histnorm="probability density",
                name="Daily Returns",
                marker_color="steelblue",
                opacity=0.75,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=pdf, name="Normal Fit", line=dict(color="red", width=2)
            )
        )
        fig.update_layout(
            title=f"{dist_stock} — Daily Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        (osm, osr), (slope, intercept, r2) = probplot(series, dist="norm")
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=osm, y=osr, mode="markers", name="Sample Quantiles",
                marker=dict(color="steelblue", size=5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=line_x, y=line_y, mode="lines", name="Normal Reference",
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(
            title=f"{dist_stock} — Q-Q Plot vs Normal",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "💡 A Q-Q plot reveals fat tails: deviation from the red line at the extremes indicates "
            "more extreme returns than a normal distribution would predict."
        )


# ----------------------------------------------------------------------------
# TAB 2: RISK ANALYSIS
# ----------------------------------------------------------------------------
with tab_risk:
    st.header("Risk Analysis")

    st.subheader("Rolling Annualized Volatility")
    vol_window = st.select_slider(
        "Rolling window (days)", options=[30, 60, 90, 120, 180], value=60, key="vol_win"
    )
    roll_vol = ret.rolling(vol_window).std() * np.sqrt(TD)
    fig = go.Figure()
    for t in tickers_used:
        fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol[t], name=t, mode="lines"))
    fig.update_layout(
        title=f"Rolling {vol_window}-Day Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drawdown Analysis")
    dd_stock = st.selectbox("Select a stock", tickers_used, key="dd_stock")
    dd_series = drawdown(ret[dd_stock])
    max_dd = dd_series.min()
    trough_date = dd_series.idxmin()

    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Maximum Drawdown", f"{max_dd:.2%}")
        st.metric("Trough Date", trough_date.strftime("%Y-%m-%d"))
    with c2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dd_series.index,
                y=dd_series.values,
                fill="tozeroy",
                fillcolor="rgba(220, 50, 50, 0.4)",
                line=dict(color="darkred"),
                name="Drawdown",
            )
        )
        fig.update_layout(
            title=f"{dd_stock} — Drawdown from Running Peak",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis_tickformat=".0%",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk-Adjusted Metrics")
    st.caption(f"Computed using risk-free rate = {rf:.2%}")
    risk_rows = {}
    for t in tickers_used:
        risk_rows[t] = {
            "Ann. Return": ret[t].mean() * TD,
            "Ann. Volatility": ret[t].std() * np.sqrt(TD),
            "Sharpe": sharpe_ratio(ret[t], rf),
            "Sortino": sortino_ratio(ret[t], rf),
            "Max Drawdown": drawdown(ret[t]).min(),
        }
    risk_rows["S&P 500"] = {
        "Ann. Return": b_ret["SP500"].mean() * TD,
        "Ann. Volatility": b_ret["SP500"].std() * np.sqrt(TD),
        "Sharpe": sharpe_ratio(b_ret["SP500"], rf),
        "Sortino": sortino_ratio(b_ret["SP500"], rf),
        "Max Drawdown": drawdown(b_ret["SP500"]).min(),
    }
    risk_df = pd.DataFrame(risk_rows).T
    st.dataframe(
        risk_df.style.format(
            {
                "Ann. Return": "{:.2%}",
                "Ann. Volatility": "{:.2%}",
                "Sharpe": "{:.3f}",
                "Sortino": "{:.3f}",
                "Max Drawdown": "{:.2%}",
            }
        ),
        use_container_width=True,
    )


# ----------------------------------------------------------------------------
# TAB 3: CORRELATION
# ----------------------------------------------------------------------------
with tab_corr:
    st.header("Correlation & Covariance")

    st.subheader("Correlation Heatmap")
    corr = ret.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title="Pairwise Correlation Matrix of Daily Returns",
        height=500,
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling Correlation")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        rc_a = st.selectbox("Stock A", tickers_used, index=0, key="rc_a")
    with cc2:
        rc_b = st.selectbox(
            "Stock B",
            tickers_used,
            index=1 if n > 1 else 0,
            key="rc_b",
        )
    with cc3:
        rc_win = st.select_slider(
            "Window (days)", options=[30, 60, 90, 120, 180], value=120, key="rc_win"
        )

    if rc_a == rc_b:
        st.info("Pick two different stocks to see a rolling correlation.")
    else:
        rc = ret[rc_a].rolling(rc_win).corr(ret[rc_b])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines", name=f"{rc_a} vs {rc_b}"))
        fig.add_hline(
            y=corr.loc[rc_a, rc_b],
            line=dict(color="red", dash="dash"),
            annotation_text=f"Full-period: {corr.loc[rc_a, rc_b]:.2f}",
            annotation_position="top right",
        )
        fig.update_layout(
            title=f"Rolling {rc_win}-Day Correlation: {rc_a} vs {rc_b}",
            xaxis_title="Date",
            yaxis_title="Correlation",
            yaxis_range=[-1, 1],
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Show Covariance Matrix (daily returns)"):
        cov_df = ret.cov()
        st.dataframe(
            cov_df.style.format("{:.6f}"),
            use_container_width=True,
        )


# ----------------------------------------------------------------------------
# TAB 4: PORTFOLIO CONSTRUCTION
# ----------------------------------------------------------------------------
with tab_port:
    st.header("Portfolio Construction & Optimization")

    ret_values = ret.values

    # Equal-Weight
    ew_w = np.full(n, 1 / n)
    ew_m = portfolio_metrics(ew_w, ret, rf)

    # GMV
    with st.spinner("Solving GMV optimization..."):
        gmv_w = optimize_gmv(ret_values, n)
    if gmv_w is None:
        st.error("⚠️ GMV optimizer failed to converge. Try a different date range or tickers.")
        st.stop()
    gmv_m = portfolio_metrics(gmv_w, ret, rf)

    # Tangency
    with st.spinner("Solving tangency (max Sharpe) optimization..."):
        tan_w, tan_warn = optimize_tangency(ret_values, n, rf)
    if tan_w is None:
        st.error("⚠️ Tangency optimizer failed to converge.")
        st.stop()
    if tan_warn:
        st.warning(f"⚠️ {tan_warn}")
    tan_m = portfolio_metrics(tan_w, ret, rf)

    # ----- Equal-Weight section
    st.subheader("Equal-Weight Portfolio (1/N)")
    cols = st.columns(5)
    cols[0].metric("Ann. Return", f"{ew_m['Ann. Return']:.2%}")
    cols[1].metric("Ann. Volatility", f"{ew_m['Ann. Volatility']:.2%}")
    cols[2].metric("Sharpe", f"{ew_m['Sharpe']:.3f}")
    cols[3].metric("Sortino", f"{ew_m['Sortino']:.3f}")
    cols[4].metric("Max Drawdown", f"{ew_m['Max Drawdown']:.2%}")

    st.divider()

    # ----- GMV & Tangency
    st.subheader("Optimized Portfolios")
    opt_cols = st.columns(2)

    for col, label, w, m in [
        (opt_cols[0], "Global Minimum Variance (GMV)", gmv_w, gmv_m),
        (opt_cols[1], "Maximum Sharpe (Tangency)", tan_w, tan_m),
    ]:
        with col:
            st.markdown(f"**{label}**")
            mc = st.columns(2)
            mc[0].metric("Ann. Return", f"{m['Ann. Return']:.2%}")
            mc[0].metric("Sharpe", f"{m['Sharpe']:.3f}")
            mc[0].metric("Max DD", f"{m['Max Drawdown']:.2%}")
            mc[1].metric("Ann. Volatility", f"{m['Ann. Volatility']:.2%}")
            mc[1].metric("Sortino", f"{m['Sortino']:.3f}")

            fig = go.Figure(
                go.Bar(
                    x=tickers_used,
                    y=w,
                    marker_color="steelblue",
                    text=[f"{x:.1%}" for x in w],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title=f"{label} — Weights",
                yaxis_tickformat=".0%",
                height=320,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----- Risk Contribution (NEW)
    st.subheader("🆕 Risk Contribution Decomposition")
    st.markdown(
        "**Percentage Risk Contribution (PRC)** measures each asset's share of total portfolio "
        "variance. The values sum to 100%. A stock with **10% weight but 25% PRC** is a "
        "disproportionate source of portfolio volatility — its risk impact exceeds its allocation. "
        "Compare each bar's height to its weight in the chart above."
    )

    rc_cols = st.columns(2)
    for col, label, w in [
        (rc_cols[0], "GMV", gmv_w),
        (rc_cols[1], "Tangency", tan_w),
    ]:
        with col:
            prc = risk_contribution(w, cov)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=tickers_used,
                    y=w,
                    name="Weight",
                    marker_color="lightsteelblue",
                    text=[f"{x:.1%}" for x in w],
                    textposition="outside",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=tickers_used,
                    y=prc,
                    name="Risk Contribution",
                    marker_color="indianred",
                    text=[f"{x:.1%}" for x in prc],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title=f"{label}: Weight vs Risk Contribution",
                yaxis_tickformat=".0%",
                barmode="group",
                height=380,
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----- Custom Portfolio (NEW)
    st.subheader("🆕 Custom Portfolio Builder")
    st.caption(
        "Adjust the sliders to set raw weights. They will be normalized to sum to 100%. "
        "Metrics and the efficient frontier chart below update live."
    )

    raw_weights = []
    slider_cols = st.columns(min(n, 4))
    for i, t in enumerate(tickers_used):
        with slider_cols[i % len(slider_cols)]:
            raw_weights.append(
                st.slider(
                    f"{t}", min_value=0.0, max_value=1.0, value=1.0 / n, step=0.01, key=f"sl_{t}"
                )
            )
    raw_weights = np.array(raw_weights)
    if raw_weights.sum() == 0:
        st.warning("⚠️ All sliders are zero. Setting equal weights.")
        custom_w = np.full(n, 1 / n)
    else:
        custom_w = raw_weights / raw_weights.sum()

    custom_m = portfolio_metrics(custom_w, ret, rf)

    norm_df = pd.DataFrame({"Ticker": tickers_used, "Normalized Weight": custom_w})
    nc1, nc2 = st.columns([1, 2])
    with nc1:
        st.dataframe(
            norm_df.style.format({"Normalized Weight": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
    with nc2:
        cm = st.columns(5)
        cm[0].metric("Ann. Return", f"{custom_m['Ann. Return']:.2%}")
        cm[1].metric("Ann. Volatility", f"{custom_m['Ann. Volatility']:.2%}")
        cm[2].metric("Sharpe", f"{custom_m['Sharpe']:.3f}")
        cm[3].metric("Sortino", f"{custom_m['Sortino']:.3f}")
        cm[4].metric("Max Drawdown", f"{custom_m['Max Drawdown']:.2%}")

    st.divider()

    # ----- Efficient Frontier
    st.subheader("Efficient Frontier & Capital Allocation Line")
    st.markdown(
        "The **efficient frontier** is the set of portfolios offering the highest expected return "
        "for each level of volatility, computed by constrained optimization at each target return. "
        "The **Capital Allocation Line (CAL)** runs from the risk-free rate through the tangency "
        "portfolio — points along it represent combinations of the risk-free asset and the tangency portfolio."
    )

    with st.spinner("Computing efficient frontier..."):
        ef_vols, ef_rets = compute_efficient_frontier(ret_values, n)

    if len(ef_vols) == 0:
        st.error("Could not compute efficient frontier.")
    else:
        ind_vols = ret.std().values * np.sqrt(TD)
        ind_rets = ret.mean().values * TD
        bench_vol = b_ret["SP500"].std() * np.sqrt(TD)
        bench_ret = b_ret["SP500"].mean() * TD

        cal_x = np.linspace(0, max(ef_vols.max(), tan_m["Ann. Volatility"]) * 1.15, 50)
        cal_y = rf + (tan_m["Ann. Return"] - rf) / tan_m["Ann. Volatility"] * cal_x

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ef_vols, y=ef_rets, mode="lines", name="Efficient Frontier",
                line=dict(color="blue", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cal_x, y=cal_y, mode="lines", name="Capital Allocation Line",
                line=dict(color="red", dash="dash", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ind_vols, y=ind_rets, mode="markers+text",
                text=tickers_used, textposition="top center",
                marker=dict(size=10, color="steelblue", line=dict(color="black", width=1)),
                name="Individual Stocks",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[ew_m["Ann. Volatility"]], y=[ew_m["Ann. Return"]], mode="markers+text",
                text=["Equal-Weight"], textposition="bottom center",
                marker=dict(size=18, color="green", symbol="square", line=dict(color="black", width=1)),
                name="Equal-Weight",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[gmv_m["Ann. Volatility"]], y=[gmv_m["Ann. Return"]], mode="markers+text",
                text=["GMV"], textposition="bottom center",
                marker=dict(size=18, color="orange", symbol="diamond", line=dict(color="black", width=1)),
                name="GMV",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[tan_m["Ann. Volatility"]], y=[tan_m["Ann. Return"]], mode="markers+text",
                text=["Tangency"], textposition="top center",
                marker=dict(size=22, color="red", symbol="star", line=dict(color="black", width=1)),
                name="Tangency",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[custom_m["Ann. Volatility"]], y=[custom_m["Ann. Return"]], mode="markers+text",
                text=["Custom"], textposition="top center",
                marker=dict(size=18, color="purple", symbol="x", line=dict(color="black", width=1)),
                name="Custom",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[bench_vol], y=[bench_ret], mode="markers+text",
                text=["S&P 500"], textposition="top center",
                marker=dict(size=16, color="black", symbol="triangle-up"),
                name="S&P 500",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0], y=[rf], mode="markers+text",
                text=["Rf"], textposition="middle right",
                marker=dict(size=12, color="black"),
                name="Risk-Free Rate",
            )
        )
        fig.update_layout(
            title="Efficient Frontier with All Portfolios Marked",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            height=600,
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----- Portfolio Comparison
    st.subheader("Portfolio Comparison")

    ew_wealth = INITIAL * (1 + ret @ ew_w).cumprod()
    gmv_wealth = INITIAL * (1 + ret @ gmv_w).cumprod()
    tan_wealth = INITIAL * (1 + ret @ tan_w).cumprod()
    custom_wealth = INITIAL * (1 + ret @ custom_w).cumprod()
    b_wealth = INITIAL * (1 + b_ret["SP500"]).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ew_wealth.index, y=ew_wealth, name="Equal-Weight", line=dict(color="green", width=2)))
    fig.add_trace(go.Scatter(x=gmv_wealth.index, y=gmv_wealth, name="GMV", line=dict(color="orange", width=2)))
    fig.add_trace(go.Scatter(x=tan_wealth.index, y=tan_wealth, name="Tangency", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=custom_wealth.index, y=custom_wealth, name="Custom", line=dict(color="purple", width=2)))
    fig.add_trace(go.Scatter(x=b_wealth.index, y=b_wealth, name="S&P 500", line=dict(color="black", dash="dash", width=2.5)))
    fig.update_layout(
        title="Cumulative Wealth: Growth of $10,000",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    bench_m = {
        "Ann. Return": b_ret["SP500"].mean() * TD,
        "Ann. Volatility": b_ret["SP500"].std() * np.sqrt(TD),
        "Sharpe": sharpe_ratio(b_ret["SP500"], rf),
        "Sortino": sortino_ratio(b_ret["SP500"], rf),
        "Max Drawdown": drawdown(b_ret["SP500"]).min(),
    }
    summary_df = pd.DataFrame(
        {
            "Equal-Weight": ew_m,
            "GMV": gmv_m,
            "Tangency": tan_m,
            "Custom": custom_m,
            "S&P 500": bench_m,
        }
    ).T
    st.dataframe(
        summary_df.style.format(
            {
                "Ann. Return": "{:.2%}",
                "Ann. Volatility": "{:.2%}",
                "Sharpe": "{:.3f}",
                "Sortino": "{:.3f}",
                "Max Drawdown": "{:.2%}",
            }
        ),
        use_container_width=True,
    )


# ----------------------------------------------------------------------------
# TAB 5: ESTIMATION WINDOW SENSITIVITY (NEW)
# ----------------------------------------------------------------------------
with tab_sens:
    st.header("🆕 Estimation Window Sensitivity")
    st.markdown(
        "Mean-variance optimization is **highly sensitive** to its inputs. Small changes in "
        "estimated mean returns or covariances can produce dramatically different portfolio weights. "
        "This section recomputes the GMV and tangency portfolios using different historical lookback "
        "windows so you can see how unstable optimizer outputs are when the estimation window shifts.\n\n"
        "**Why this matters:** Historical optimization results are only as stable as the inputs that "
        "produced them. A 'best' portfolio computed on a 1-year window can look very different from "
        "one computed on a 5-year window, even with the same assets."
    )

    total_years = (ret.index[-1] - ret.index[0]).days / 365.25
    candidate_windows = [
        ("1 Year", 1),
        ("3 Years", 3),
        ("5 Years", 5),
        ("Full Sample", None),
    ]
    available = [(label, yrs) for label, yrs in candidate_windows if yrs is None or yrs <= total_years]

    if len(available) < 2:
        st.warning(
            f"Only {total_years:.1f} years of data available — need a longer date range to compare multiple windows."
        )
    else:
        st.caption(f"Data span: {total_years:.1f} years. Showing windows that fit within this range.")

        sens_results = []
        weight_records = []

        for label, yrs in available:
            if yrs is None:
                sub = ret
            else:
                cutoff = ret.index[-1] - pd.Timedelta(days=int(yrs * 365.25))
                sub = ret.loc[ret.index >= cutoff]
            if len(sub) < 60:
                continue

            sub_vals = sub.values
            sub_n = sub.shape[1]

            gmv_sub = optimize_gmv(sub_vals, sub_n)
            tan_sub, _ = optimize_tangency(sub_vals, sub_n, rf)

            if gmv_sub is None or tan_sub is None:
                sens_results.append({"Window": label, "Status": "Optimizer failed"})
                continue

            gmv_metrics = portfolio_metrics(gmv_sub, sub, rf)
            tan_metrics = portfolio_metrics(tan_sub, sub, rf)

            sens_results.append(
                {
                    "Window": label,
                    "Obs": len(sub),
                    "GMV Return": gmv_metrics["Ann. Return"],
                    "GMV Vol": gmv_metrics["Ann. Volatility"],
                    "GMV Sharpe": gmv_metrics["Sharpe"],
                    "Tan Return": tan_metrics["Ann. Return"],
                    "Tan Vol": tan_metrics["Ann. Volatility"],
                    "Tan Sharpe": tan_metrics["Sharpe"],
                }
            )

            for i, t in enumerate(tickers_used):
                weight_records.append(
                    {"Window": label, "Ticker": t, "Portfolio": "GMV", "Weight": gmv_sub[i]}
                )
                weight_records.append(
                    {"Window": label, "Ticker": t, "Portfolio": "Tangency", "Weight": tan_sub[i]}
                )

        if sens_results:
            st.subheader("Comparison Table")
            sens_df = pd.DataFrame(sens_results).set_index("Window")
            st.dataframe(
                sens_df.style.format(
                    {
                        "GMV Return": "{:.2%}",
                        "GMV Vol": "{:.2%}",
                        "GMV Sharpe": "{:.3f}",
                        "Tan Return": "{:.2%}",
                        "Tan Vol": "{:.2%}",
                        "Tan Sharpe": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )

            wdf = pd.DataFrame(weight_records)

            st.subheader("Weights Across Estimation Windows")
            for portfolio_type in ["GMV", "Tangency"]:
                sub_w = wdf[wdf["Portfolio"] == portfolio_type]
                fig = px.bar(
                    sub_w,
                    x="Ticker",
                    y="Weight",
                    color="Window",
                    barmode="group",
                    title=f"{portfolio_type} Portfolio Weights by Lookback Window",
                )
                fig.update_layout(yaxis_tickformat=".0%", height=400)
                st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# TAB 6: ABOUT
# ----------------------------------------------------------------------------
with tab_about:
    st.header("About This Application")
    st.markdown(
        """
### Purpose
This interactive web application lets you construct and analyze equity portfolios in real time.
It downloads historical price data, computes return and risk statistics, runs mean-variance
optimization, and provides several diagnostics that go beyond static notebook analysis.

### Sections
1. **Returns & EDA** — summary statistics, cumulative wealth index, return distribution diagnostics.
2. **Risk Analysis** — rolling volatility, drawdown, and risk-adjusted performance metrics.
3. **Correlation** — correlation heatmap, rolling pairwise correlation, covariance matrix.
4. **Portfolio Construction** — equal-weight, GMV, tangency, custom portfolios; risk contribution; efficient frontier with CAL.
5. **Window Sensitivity** — how optimizer outputs change with different lookback windows.

### Methodology
- **Data source:** Yahoo Finance via `yfinance`. Adjusted close prices (auto-adjusted for dividends and splits).
- **Returns:** Simple (arithmetic) daily returns. Used consistently throughout — log returns are not additive across assets and would distort portfolio math.
- **Annualization:** Daily mean return × 252; daily volatility × √252.
- **Risk-free rate:** User-specified annualized rate. Converted to a daily rate as `rf / 252` where needed.
- **Sharpe ratio:** `(annualized return − rf) / annualized volatility`.
- **Sortino ratio:** `(annualized return − rf) / annualized downside deviation`. Downside deviation uses only daily returns below the daily risk-free rate.
- **Portfolio variance:** Full quadratic form `wᵀΣw` (not a weighted average of individual variances).
- **Optimization:** `scipy.optimize.minimize` with method `SLSQP`. Constraints: weights sum to 1, each weight bounded `(0, 1)` (no short selling).
- **Tangency portfolio:** Found by minimizing the negative Sharpe ratio.
- **Efficient frontier:** Constructed by constrained minimization of portfolio variance at each target return level along a grid (not random simulation).
- **Risk contribution (PRC):** `PRCᵢ = wᵢ · (Σw)ᵢ / σ²ₚ`. Sums to 1 by construction.
- **Drawdown:** Computed from the running peak of the cumulative simple-return wealth index.

### Data Handling
- Tickers that fail to download or have more than 5% missing values are dropped with a warning.
- The S&P 500 (`^GSPC`) is downloaded as a benchmark for comparison only — it is **never** included in any portfolio optimization.
- All computed series are aligned to the overlapping date range across the surviving tickers.

### Caching
Both data downloads (`@st.cache_data` with a 1-hour TTL) and optimization routines are cached so that
widget interactions don't trigger expensive recomputation.
        """
    )
    st.caption("Built for FDA II — Project 2.")
