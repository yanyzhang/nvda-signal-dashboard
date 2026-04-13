import pandas as pd
import plotly.express as px
import requests
import statsmodels.api as sm
start = "2025-01-15"
end = "2025-03-15"
nvda = pd.read_csv("NVDA.csv")
nvda["ticker"] = "NVDA"
amd = pd.read_csv("AMD.csv")
amd["ticker"] = "AMD"
asml = pd.read_csv("ASML.csv")
asml["ticker"] = "ASML"
arm = pd.read_csv("ARM.csv")
arm["ticker"] = "ARM"
df = pd.concat([nvda, asml, arm, amd])
# 图一股价对比
df["norm_price"] = df["close"] / df.groupby("ticker")["close"].transform("first") * 100

df["date"] = pd.to_datetime(df["date"])
df = df[(df["date"] >= start) & (df["date"] <= end)]
fig = px.line(
    df, x="date",
    y="norm_price",
    color="ticker",
    labels={
        "date": "Date",
        "norm_price": "Normalized Price"
    },
    hover_data={
        "ticker": True,
        "norm_price": ":.2f",
        "date": True,
        "close": True,
        "open": True
    },

)

# fig.show()

url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Nvidia/daily/20250101/20251231"
headers = {
    "User-Agent": "MyDashProject/1.0 (student project; contact: your_email@example.com)"
}

raw = requests.get(url, headers=headers)

data = raw.json()
df2 = pd.DataFrame(data["items"])
df2["date"] = pd.to_datetime(df2["timestamp"].astype(str).str[:8])
df2 = df2[["date", "views"]]
df_nvda = df[df["ticker"] == "NVDA"].copy()
df_merge = pd.merge(df_nvda, df2, on="date", how="inner")
df_merge = df_merge[(df_merge["date"] >= start) & (df_merge["date"] <= end)]
df_merge["month"] = df_merge["date"].dt.month
fig2 = px.scatter(
    df_merge,
    x="norm_price",
    y="views",
    trendline="ols",
    title="NVDA: Attention vs Stock Price",
    labels={
        "views": "Number of Views",
        "norm_price": "Normalized Price"
    },
    color = "month",
    hover_data={
        "ticker": True,
        "norm_price": True,
        "views": True,
        "date": True
    },
)

