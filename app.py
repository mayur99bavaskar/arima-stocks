import itertools
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from arch import arch_model
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", interval="1d",
                     auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "Close"]].dropna()


def pick_d(series: pd.Series, max_d: int = 2) -> int:
    d, s = 0, series.copy()
    while d < max_d and adfuller(s.dropna())[1] > 0.05:
        s = s.diff().dropna()
        d += 1
    return d


def fit_best_arima(series: pd.Series, max_p: int = 3, max_q: int = 3):
    # Drop the date index (it has market-holiday gaps) so statsmodels can forecast
    series = pd.Series(series.to_numpy(dtype=float))
    d = pick_d(series)
    best_aic, best_fit, best_order = float("inf"), None, None
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            fit = ARIMA(series, order=(p, d, q)).fit()
            if fit.aic < best_aic:
                best_aic, best_fit, best_order = fit.aic, fit, (p, d, q)
        except Exception:
            continue
    return best_fit, best_order


def forecast(series: pd.Series, steps: int):
    fit, order = fit_best_arima(series)
    res = fit.get_forecast(steps=steps)
    mean = res.predicted_mean
    ci = res.conf_int(alpha=0.05)
    idx = pd.bdate_range(series.index[-1] + pd.Timedelta(days=1), periods=steps)
    out = pd.DataFrame({
        "Forecast": np.asarray(mean),
        "Lower 95%": np.asarray(ci.iloc[:, 0]),
        "Upper 95%": np.asarray(ci.iloc[:, 1]),
    }, index=idx)
    return out, order


def garch_band(series: pd.Series, fc: pd.DataFrame):
    """Fit GARCH(1,1) on daily log returns and build a volatility-based
    95% band around the ARIMA forecast (ARIMA mean + GARCH variance)."""
    steps = len(fc)
    r = 100 * np.log(series / series.shift(1)).dropna()
    am = arch_model(r.to_numpy(dtype=float), mean="Constant",
                    vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    var = res.forecast(horizon=steps, reindex=False).variance.to_numpy()[-1]
    daily_vol = np.sqrt(var)                      # % per day
    cum_sd = np.sqrt(np.cumsum(var)) / 100        # cumulative, in log terms
    out = pd.DataFrame({
        "GARCH Lower 95%": fc["Forecast"].to_numpy() * np.exp(-1.96 * cum_sd),
        "GARCH Upper 95%": fc["Forecast"].to_numpy() * np.exp(1.96 * cum_sd),
        "Daily volatility %": daily_vol,
    }, index=fc.index)
    return out


def backtest_mape(series: pd.Series, holdout: int = 20) -> float:
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    fit, _ = fit_best_arima(train)
    pred = np.asarray(fit.forecast(steps=holdout))
    return float((abs(test.to_numpy() - pred) / test.to_numpy()).mean() * 100)


def plot(history: pd.Series, fc: pd.DataFrame, label: str,
         band: pd.DataFrame | None = None) -> go.Figure:
    h = history.iloc[-60:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h, name="History"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Forecast"], name="ARIMA forecast",
                             line=dict(dash="dash")))
    fig.add_trace(go.Scatter(
        x=list(fc.index) + list(fc.index[::-1]),
        y=list(fc["Upper 95%"]) + list(fc["Lower 95%"][::-1]),
        fill="toself", opacity=0.2, line=dict(width=0), name="ARIMA 95%"))
    if band is not None:
        fig.add_trace(go.Scatter(
            x=list(band.index) + list(band.index[::-1]),
            y=list(band["GARCH Upper 95%"]) + list(band["GARCH Lower 95%"][::-1]),
            fill="toself", opacity=0.25, line=dict(width=0),
            fillcolor="orange", name="GARCH 95%"))
    fig.update_layout(title=f"{label} price", height=350,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h"))
    return fig


st.set_page_config(page_title="ARIMA + GARCH Stock Forecaster", layout="centered")
st.title("ARIMA + GARCH Stock Forecaster")
st.caption("Next-day open and close forecasts using ARIMA, with GARCH "
           "volatility bands. Not investment advice.")

symbol = st.text_input("Stock symbol", placeholder="e.g. RELIANCE, TCS, INDUSINDBK")
exchange = st.radio("Exchange", ["NSE", "BSE"], horizontal=True)
col1, col2 = st.columns(2)
days = col1.number_input("Days to forecast", 1, 10, 4)
years = col2.number_input("Years of history", 1, 10, 3)
use_garch = st.checkbox("Add GARCH volatility bands", value=True)
run_backtest = st.checkbox("Show backtest accuracy (slower)", value=True)

if st.button("Forecast", type="primary") and symbol.strip():
    ticker = symbol.strip().upper() + SUFFIX[exchange]
    with st.spinner(f"Fetching {ticker}..."):
        data = load_data(ticker, int(years))

    if data.empty or len(data) < 100:
        st.error(f"Not enough data for {ticker}. Check the symbol and exchange.")
        st.stop()

    results, bands = {}, {}
    for col in ["Open", "Close"]:
        with st.spinner(f"Fitting models on {col} prices..."):
            fc, order = forecast(data[col], int(days))
            results[col] = (fc, order)
            if use_garch:
                try:
                    bands[col] = garch_band(data[col], fc)
                except Exception as e:
                    st.warning(f"GARCH failed for {col}: {e}")
                    bands[col] = None
            else:
                bands[col] = None

    st.subheader(f"{ticker}: next {int(days)} trading days")
    table = pd.DataFrame({
        "Target Open": results["Open"][0]["Forecast"],
        "Target Close": results["Close"][0]["Forecast"],
    })
    if bands.get("Close") is not None:
        table["Close low (GARCH)"] = bands["Close"]["GARCH Lower 95%"]
        table["Close high (GARCH)"] = bands["Close"]["GARCH Upper 95%"]
        table["Daily vol %"] = bands["Close"]["Daily volatility %"]
    table = table.round(2)
    table.index = table.index.strftime("%a, %d %b %Y")
    st.dataframe(table, use_container_width=True)
    st.caption(f"Last close: {data['Close'].iloc[-1]:.2f} on "
               f"{data.index[-1].strftime('%d %b %Y')}. "
               f"ARIMA orders: Open {results['Open'][1]}, Close {results['Close'][1]}.")

    for col in ["Open", "Close"]:
        st.plotly_chart(plot(data[col], results[col][0], col, bands.get(col)),
                        use_container_width=True)

    if run_backtest:
        with st.spinner("Backtesting on the last 20 days..."):
            m_open = backtest_mape(data["Open"])
            m_close = backtest_mape(data["Close"])
        st.info(f"Backtest MAPE (20-day holdout): Open {m_open:.2f}%, "
                f"Close {m_close:.2f}%")

    st.warning("Forecasts ignore market holidays, news and earnings. "
               "Stock prices are close to a random walk, so treat these as "
               "indicative only.")
