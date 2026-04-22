from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from dash import Input, Output, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots


START = "2025-01-15"
END = "2025-03-15"


def _data_dir() -> Path:
    cwd = Path.cwd()
    if (cwd / "NVDA.csv").exists():
        return cwd
    return Path(__file__).resolve().parent


def load_base_data(data_dir: Path) -> pd.DataFrame:
    nvda = pd.read_csv(data_dir / "NVDA.csv")
    nvda["ticker"] = "NVDA"
    amd = pd.read_csv(data_dir / "AMD.csv")
    amd["ticker"] = "AMD"
    asml = pd.read_csv(data_dir / "ASML.csv")
    asml["ticker"] = "ASML"
    arm = pd.read_csv(data_dir / "ARM.csv")
    arm["ticker"] = "ARM"
    soxx = pd.read_csv(data_dir / "SOXX.csv")
    soxx["ticker"] = "SOXX"

    df = pd.concat([nvda, asml, arm, amd, soxx], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    df["norm_price"] = (
        df["close"] / df.groupby("ticker")["close"].transform("first") * 100
    )
    return df


def load_merge_data(df: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wiki = pd.read_csv(data_dir / "wikimedia_pageviews.csv")
    wiki["date"] = pd.to_datetime(wiki["timestamp"].astype(str).str[:8])
    df2 = wiki[["date", "views"]]

    df_nvda = df[df["ticker"] == "NVDA"].copy()
    df_merge = pd.merge(df_nvda, df2, on="date", how="inner")
    df_merge = df_merge[(df_merge["date"] >= START) & (df_merge["date"] <= END)].copy()
    df_merge["month_label"] = df_merge["date"].dt.strftime("%B %Y")

    gdelt = pd.read_csv(data_dir / "gdelt_gkg_tone_timeline.csv")
    gdelt["date"] = pd.to_datetime(gdelt["Date"])
    gdelt["tone"] = gdelt["Value"]
    gdelt = gdelt.groupby("date", as_index=False)["tone"].mean()

    df_fig3 = (
        df_nvda[["date", "close", "norm_price", "ticker"]]
        .merge(df2, on="date", how="left")
        .merge(gdelt[["date", "tone"]], on="date", how="left")
    )
    df_fig3 = df_fig3[
        (df_fig3["date"] >= pd.to_datetime(START))
        & (df_fig3["date"] <= pd.to_datetime(END))
    ].copy()

    return df_merge, df_fig3


def build_figures(
    filtered_df: pd.DataFrame,
    filtered_merge: pd.DataFrame,
    filtered_f3: pd.DataFrame,
    clicked_date: str | None,
    fig12_layout: dict,
    dashboard_layout: dict,
    ticker_colors: dict,
    month_colors: dict,
    month_order: list[str],
    nvda_color: str,
    axis_title_font_size: int,
    axis_tick_font_size: int,
    title_font_size: int,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    start_dt = filtered_df["date"].min()
    end_dt = filtered_df["date"].max()
    event_ts = pd.Timestamp("2025-01-27")
    event_in_range = pd.Timestamp(start_dt) <= event_ts <= pd.Timestamp(end_dt)

    new_fig1 = px.line(
        filtered_df,
        x="date",
        y="norm_price",
        color="ticker",
        custom_data=["open", "close"],
        color_discrete_map=ticker_colors,
        title="<b>Semiconductor Peer Comparison: Normalized Price Performance (Jan 15 = 100)</b>",
        labels={"date": "Date", "norm_price": "Normalized Price", "ticker": "Ticker"},
    )
    new_fig1.update_layout(**fig12_layout, legend_title_text="Ticker", height=300)
    new_fig1.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        title_font=dict(size=axis_title_font_size),
        tickfont=dict(size=axis_tick_font_size),
    )
    new_fig1.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        title_font=dict(size=axis_title_font_size),
        tickfont=dict(size=axis_tick_font_size),
    )
    new_fig1.update_traces(
        hovertemplate=(
            "Date: %{x|%m/%d/%Y}<br>Ticker: %{fullData.name}<br>"
            "Normalized Price: %{y:.2f}<br>Open: %{customdata[0]:.2f}<br>"
            "Close: %{customdata[1]:.2f}<extra></extra>"
        )
    )
    new_fig1.update_traces(selector=dict(name="NVDA"), line=dict(width=4))
    new_fig1.update_traces(selector=dict(name="SOXX"), line=dict(dash="dash", width=2))

    if event_in_range:
        new_fig1.add_shape(
            type="line",
            x0="2025-01-27",
            x1="2025-01-27",
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="gray", dash="dash", width=1),
        )
        new_fig1.add_annotation(
            x="2025-01-27",
            y=1,
            xref="x",
            yref="paper",
            text="DeepSeek",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=10, color="gray"),
        )

    new_fig1.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.95,
        text="SOXX = Semiconductor ETF",
        showarrow=False,
        align="right",
        font=dict(size=11, color="#334155"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        xanchor="right",
        yanchor="top",
    )

    if len(filtered_merge) < 2:
        new_fig2 = px.scatter(
            filtered_merge,
            x="norm_price",
            y="views",
            color="month_label",
            color_discrete_map=month_colors,
            category_orders={"month_label": month_order},
            labels={
                "views": "Wikipedia Pageviews",
                "norm_price": "Normalized NVDA Price (Jan 15 = 100)",
                "month_label": "Month",
                "ticker": "Ticker",
            },
            custom_data=["date"],
            title="<b>Public Attention vs. Price (color: month)(not enough data)</b>",
        )
    else:
        new_fig2 = px.scatter(
            filtered_merge,
            x="norm_price",
            y="views",
            trendline="ols",
            trendline_scope="overall",
            color="month_label",
            color_discrete_map=month_colors,
            category_orders={"month_label": month_order},
            labels={
                "views": "Wikipedia Pageviews",
                "norm_price": "Normalized NVDA Price (Jan 15 = 100)",
                "month_label": "Month",
                "ticker": "Ticker",
            },
            custom_data=["date"],
            title="<b>Public Attention vs. Price (color: month)</b>",
        )

    new_fig2.update_layout(
        **fig12_layout,
        legend_title_text="Month",
        height=300,
        hoverlabel=dict(bgcolor="white", font_color="black", bordercolor="gray"),
    )

    if len(filtered_merge) >= 2:
        x_vals = sm.add_constant(filtered_merge["norm_price"])
        model = sm.OLS(filtered_merge["views"], x_vals).fit()
        new_fig2.add_annotation(
            x=0.98,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"R² = {model.rsquared:.3f}",
            showarrow=False,
            xanchor="right",
            font=dict(size=13, color="#334155"),
            bgcolor="white",
            bordercolor="#e5e7eb",
            borderwidth=1,
        )

    new_fig2.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        title_font=dict(size=axis_title_font_size),
        tickfont=dict(size=axis_tick_font_size),
    )
    new_fig2.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        title_font=dict(size=axis_title_font_size),
        tickfont=dict(size=axis_tick_font_size),
    )
    new_fig2.update_traces(
        selector=dict(mode="markers"),
        marker=dict(size=12, opacity=0.95, line=dict(width=1.2, color="#334155")),
        hovertemplate=(
            "Date: %{customdata[0]|%m/%d/%Y}<br>Month: %{fullData.name}<br>"
            "Normalized Price: %{x:.2f}<br>Wikipedia Pageviews: %{y:,.0f}<extra></extra>"
        ),
    )

    active_months = filtered_merge["month_label"].unique().tolist()
    for trace in new_fig2.data:
        if hasattr(trace, "name") and trace.name in active_months:
            trace.marker.opacity = 0.95
        elif hasattr(trace, "name") and trace.name != "Overall Trendline":
            trace.marker.opacity = 0.2

    tone_clr = filtered_f3["tone"].fillna(0).apply(lambda x: "#0891B2" if x >= 0 else "#EA580C")

    new_fig3 = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=1)
    new_fig3.add_trace(
        go.Scatter(
            x=filtered_f3["date"],
            y=filtered_f3["close"],
            name="NVDA Close",
            mode="lines",
            line=dict(color=nvda_color, width=4),
        ),
        secondary_y=False,
    )
    new_fig3.add_trace(
        go.Scatter(
            x=filtered_f3["date"],
            y=filtered_f3["views"],
            name="Wikipedia Views",
            mode="lines",
            line=dict(color="blue", dash="dot"),
            hovertemplate="Date: %{x|%m/%d/%Y}<br>Wikipedia Pageviews: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    new_fig3.add_trace(
        go.Bar(
            x=filtered_f3["date"],
            y=filtered_f3["tone"],
            name="GDELT Tone",
            marker_color=tone_clr,
            opacity=0.7,
            yaxis="y3",
            customdata=filtered_f3["tone"],
            hovertemplate="Date: %{x|%m/%d/%Y}<br>GDELT Tone: %{customdata:.3f}<extra></extra>",
        )
    )

    if event_in_range:
        new_fig3.add_shape(
            type="line",
            x0="2025-01-27",
            x1="2025-01-27",
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="black", dash="dash"),
        )
        new_fig3.add_annotation(
            x="2025-01-27",
            y=1,
            xref="x",
            yref="paper",
            text="DeepSeek Jan 27",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
        )

    new_fig3.update_layout(
        **dashboard_layout,
        title={"text": "<b>Price, Wikipedia Views & News Tone</b>", "font": {"size": title_font_size}},
        hovermode="x unified",
        height=340,
    )
    new_fig3.update_layout(
        yaxis3=dict(
            overlaying="y",
            side="right",
            position=0.98,
            showgrid=False,
            title_text="GDELT Tone",
            title_font=dict(size=12),
            title_standoff=4,
            tickfont=dict(size=12),
            showticklabels=True,
        ),
        margin={"l": 60, "r": 120, "t": 50, "b": 55},
    )
    new_fig3.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        title_font=dict(size=axis_title_font_size),
        tickfont=dict(size=axis_tick_font_size),
    )
    new_fig3.update_yaxes(
        title_text="NVDA Close Price (USD)",
        secondary_y=False,
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        title_font=dict(size=axis_title_font_size),
        tickfont=dict(size=axis_tick_font_size),
    )
    new_fig3.update_yaxes(
        title_text="Wiki Views",
        secondary_y=True,
        showgrid=False,
        title_font=dict(size=12),
        tickfont=dict(size=12),
    )
    new_fig3.update_layout(
        yaxis2=dict(
            overlaying="y",
            side="right",
            position=0.91,
            showgrid=False,
            title_font=dict(size=12),
            title_standoff=4,
            tickfont=dict(size=12),
        )
    )

    if clicked_date:
        clicked_ts = pd.Timestamp(clicked_date)
        new_fig1.add_shape(
            type="line",
            x0=clicked_ts,
            x1=clicked_ts,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="orange", dash="dash", width=2),
        )
        new_fig1.add_annotation(
            x=clicked_ts,
            y=1,
            xref="x",
            yref="paper",
            text=str(clicked_ts.date()),
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="orange"),
        )

        match = filtered_merge[filtered_merge["date"] == clicked_ts]
        if not match.empty:
            new_fig2.add_trace(
                go.Scatter(
                    x=match["norm_price"],
                    y=match["views"],
                    mode="markers",
                    marker=dict(
                        color="orange",
                        size=24,
                        symbol="diamond-open",
                        line=dict(width=5),
                    ),
                    name="Selected",
                    showlegend=False,
                )
            )

        new_fig3.add_shape(
            type="line",
            x0=clicked_ts,
            x1=clicked_ts,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="orange", dash="dash", width=2),
        )

    return new_fig1, new_fig2, new_fig3


data_dir = _data_dir()
df = load_base_data(data_dir)
df_merge, df_fig3 = load_merge_data(df, data_dir)

month_order = (
    df_merge[["month_label", "date"]]
    .drop_duplicates()
    .sort_values("date")["month_label"]
    .tolist()
)

all_dates = sorted(df["date"].unique())
idx_to_date = {i: d for i, d in enumerate(all_dates)}
n = len(all_dates)
marks = {i: pd.Timestamp(all_dates[i]).strftime("%b %d") for i in range(0, n, 7)}
marks[n - 1] = pd.Timestamp(all_dates[-1]).strftime("%b %d")

DASHBOARD_FONT = "IBM Plex Sans, Segoe UI, Arial, sans-serif"
TITLE_FONT_SIZE = 15
AXIS_TITLE_FONT_SIZE = 13
AXIS_TICK_FONT_SIZE = 11
FIG12_TOP_MARGIN = 50

dashboard_layout = {
    "font": {"family": DASHBOARD_FONT, "size": 13, "color": "#1f2937"},
    "plot_bgcolor": "#f8fafc",
    "paper_bgcolor": "#ffffff",
    "margin": {"l": 60, "r": 80, "t": 50, "b": 55},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "font": {"size": 11},
    },
    "title_font_size": TITLE_FONT_SIZE,
}
fig12_layout = dict(dashboard_layout)
fig12_layout["margin"] = {"l": 60, "r": 140, "t": FIG12_TOP_MARGIN, "b": 55}
fig12_layout["legend"] = {
    "orientation": "v",
    "yanchor": "top",
    "y": 1,
    "xanchor": "left",
    "x": 1.02,
    "font": {"size": 11},
}

ticker_colors = {
    "NVDA": "#76B900",
    "AMD": "#ED1C24",
    "ASML": "#005EB8",
    "ARM": "#00A0DF",
    "SOXX": "#B45309",
}
month_colors = {
    "January 2025": "#1E3A8A",
    "February 2025": "#3B82F6",
    "March 2025": "#94A3B8",
}

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap",
    ],
)
server = app.server

app.layout = html.Div(
    [
        html.H1(
            "NVIDIA Signal Dashboard: Stock, Attention & News Tone (Jan-Mar 2025)",
            style={
                "textAlign": "center",
                "fontSize": "24px",
                "marginBottom": "0px",
                "marginTop": "4px",
            },
        ),
        html.P(
            [
                "Investigating whether NVDA's price movement in winter 2025 aligned with public attention, AI news sentiment, or broader semiconductor momentum.",
                html.Br(),
                "Note: 03/15/2025 is a non-trading day, so market-price lines may end at the last available trading day (03/14/2025).",
            ],
            style={
                "textAlign": "center",
                "color": "gray",
                "fontSize": "11px",
                "marginTop": "0px",
                "marginBottom": "4px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Month",
                            style={"fontSize": "11px", "color": "gray", "marginBottom": "0"},
                        ),
                        dcc.Dropdown(
                            id="month-selector",
                            options=[
                                {"label": "Jan", "value": "JAN"},
                                {"label": "Feb", "value": "FEB"},
                                {"label": "Mar", "value": "MAR"},
                                {"label": "All", "value": "ALL"},
                            ],
                            value="ALL",
                            clearable=False,
                            style={"fontSize": "12px", "height": "30px"},
                        ),
                    ],
                    style={
                        "width": "120px",
                        "flexShrink": "0",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "2px",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Date Range",
                            style={"fontSize": "11px", "color": "gray", "marginBottom": "0"},
                        ),
                        dcc.RangeSlider(
                            id="date-slider",
                            min=0,
                            max=n - 1,
                            step=1,
                            value=[0, n - 1],
                            marks=marks,
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode="drag",
                        ),
                    ],
                    style={
                        "flex": "1",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "2px",
                        "paddingBottom": "4px",
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "Reset",
                            id="reset-btn",
                            n_clicks=0,
                            style={
                                "height": "30px",
                                "padding": "0 10px",
                                "borderRadius": "6px",
                                "border": "1px solid #cbd5e1",
                                "backgroundColor": "#f8fafc",
                                "color": "#0f172a",
                                "fontWeight": "500",
                                "cursor": "pointer",
                                "fontSize": "12px",
                            },
                        )
                    ],
                    style={"flexShrink": "0", "paddingTop": "16px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "flex-start",
                "gap": "12px",
                "maxWidth": "900px",
                "margin": "0px auto 4px auto",
            },
        ),
        dcc.Store(id="clicked-date", data=None),
        dcc.Store(id="zoom-state", data=None),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="fig1",
                            config={"scrollZoom": True},
                            style={"height": "300px"},
                        )
                    ],
                    style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="fig2",
                            config={"scrollZoom": True},
                            style={"height": "300px"},
                        )
                    ],
                    style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
                ),
            ],
            style={"marginTop": "10px", "marginBottom": "0px"},
        ),
        html.Div(
            [
                dcc.Graph(id="fig3", config={"scrollZoom": True}, style={"height": "340px"}),
            ]
        ),
    ],
    style={
        "minWidth": "1200px",
        "margin": "0 auto",
        "padding": "0 24px",
        "fontFamily": "Inter, sans-serif",
    },
)


@app.callback(
    Output("date-slider", "value"),
    [Input("reset-btn", "n_clicks"), Input("month-selector", "value")],
    prevent_initial_call=True,
)
def update_slider(n_clicks: int, selected_period: str) -> list[int]:
    _ = n_clicks
    trigger = callback_context.triggered[0]["prop_id"]
    if "reset-btn" in trigger:
        return [0, n - 1]

    month_map = {
        "ALL": ("2025-01-15", "2025-03-15"),
        "JAN": ("2025-01-15", "2025-01-31"),
        "FEB": ("2025-02-01", "2025-02-28"),
        "MAR": ("2025-03-01", "2025-03-15"),
    }
    s, e = month_map.get(selected_period, ("2025-01-15", "2025-03-15"))
    s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
    s_idx = min(range(n), key=lambda i: abs(pd.Timestamp(idx_to_date[i]) - s_ts))
    e_idx = min(range(n), key=lambda i: abs(pd.Timestamp(idx_to_date[i]) - e_ts))
    return [s_idx, e_idx]


@app.callback(
    Output("clicked-date", "data"),
    [
        Input("fig1", "clickData"),
        Input("fig2", "clickData"),
        Input("fig3", "clickData"),
        Input("reset-btn", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def store_clicked_date(click1, click2, click3, reset_clicks):
    _ = reset_clicks
    trigger = callback_context.triggered[0]["prop_id"]
    if "reset-btn" in trigger:
        return None
    if "fig1" in trigger and click1:
        return click1["points"][0]["x"]
    if "fig2" in trigger and click2:
        point = click2["points"][0]
        if "customdata" in point:
            return point["customdata"][0]
    if "fig3" in trigger and click3:
        return click3["points"][0]["x"]
    return no_update


@app.callback(
    Output("zoom-state", "data"),
    [Input("fig1", "relayoutData"), Input("fig3", "relayoutData")],
    prevent_initial_call=True,
)
def store_zoom(relay1, relay3):
    trigger = callback_context.triggered[0]["prop_id"]
    relay = relay1 if "fig1" in trigger else relay3
    if not relay:
        return no_update
    if "xaxis.range[0]" in relay:
        return {"x0": relay["xaxis.range[0]"], "x1": relay["xaxis.range[1]"]}
    if "xaxis.autorange" in relay:
        return None
    return no_update


@app.callback(
    [Output("fig1", "figure"), Output("fig2", "figure"), Output("fig3", "figure")],
    [
        Input("date-slider", "value"),
        Input("clicked-date", "data"),
        Input("zoom-state", "data"),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_all(slider_range, clicked_date, zoom_state, reset_clicks):
    _ = reset_clicks
    start_dt = idx_to_date[slider_range[0]]
    end_dt = idx_to_date[slider_range[1]]

    trigger = callback_context.triggered[0]["prop_id"]
    if "reset-btn" in trigger:
        clicked_date = None

    if clicked_date:
        clicked_ts = pd.Timestamp(clicked_date)
        if not (pd.Timestamp(start_dt) <= clicked_ts <= pd.Timestamp(end_dt)):
            clicked_date = None

    filtered_df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    filtered_merge = df_merge[(df_merge["date"] >= start_dt) & (df_merge["date"] <= end_dt)]
    filtered_f3 = df_fig3[(df_fig3["date"] >= start_dt) & (df_fig3["date"] <= end_dt)]

    new_fig1, new_fig2, new_fig3 = build_figures(
        filtered_df=filtered_df,
        filtered_merge=filtered_merge,
        filtered_f3=filtered_f3,
        clicked_date=clicked_date,
        fig12_layout=fig12_layout,
        dashboard_layout=dashboard_layout,
        ticker_colors=ticker_colors,
        month_colors=month_colors,
        month_order=month_order,
        nvda_color=ticker_colors["NVDA"],
        axis_title_font_size=AXIS_TITLE_FONT_SIZE,
        axis_tick_font_size=AXIS_TICK_FONT_SIZE,
        title_font_size=TITLE_FONT_SIZE,
    )

    if zoom_state:
        x0 = zoom_state["x0"]
        x1 = zoom_state["x1"]
        new_fig1.update_xaxes(range=[x0, x1])
        new_fig3.update_xaxes(range=[x0, x1])

    return new_fig1, new_fig2, new_fig3


if __name__ == "__main__":
    app.run(debug=True)
