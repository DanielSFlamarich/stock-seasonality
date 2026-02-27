# src/visualization/interactive_plots.py

import pandas as pd
import plotly.graph_objects as go
from ipywidgets import interact


def show_interactive_seasonality_plot(df_rolling: pd.DataFrame, df: pd.DataFrame):
    """
    Launches an interactive Plotly plot showing seasonality score vs.
    actual closing prices.

    Parameters
    ----------
    df_rolling : pd.DataFrame
        Output from SeasonalityETL.fit_rolling(), containing metrics.
    df : pd.DataFrame
        Raw 1d interval data from DataLoader.load(), containing actual close prices.
    """

    tickers = sorted(df_rolling["ticker"].unique().tolist())
    frequencies = sorted(df_rolling["freq"].unique().tolist())
    score_options = [
        "acf_lag_val",
        "p2m_val",
        "stl_strength",
        "seasonality_score_linear",
        "seasonality_score_geom",
        "seasonality_score_harmonic",
    ]

    @interact(ticker=tickers, freq=frequencies, score=score_options)
    def _plot(ticker, freq, score):
        df_score = df_rolling[
            (df_rolling["ticker"] == ticker) & (df_rolling["freq"] == freq)
        ].sort_values("window_start")

        if df_score.empty:
            print("No metrics available for this selection.")
            return

        df_price = df[df["ticker"] == ticker].copy()
        df_price["date"] = pd.to_datetime(df_price["date"])
        df_price = df_price.set_index("date").sort_index()

        fig = go.Figure()

        # Add rolling seasonality score
        fig.add_trace(
            go.Scatter(
                x=df_score["window_start"],
                y=df_score[score],
                mode="lines+markers",
                name=score,
                line=dict(width=2),
                yaxis="y1",
            )
        )

        # Add raw daily close prices
        fig.add_trace(
            go.Scatter(
                x=df_price.index,
                y=df_price["close"],
                mode="lines",
                name="Daily Close",
                line=dict(dash="dot", width=1.0, color="red"),
                yaxis="y2",
            )
        )

        fig.update_layout(
            title=f"{ticker} — {score} and Daily Close over time ({freq})",
            xaxis_title="Date",
            yaxis=dict(title=score, side="left"),
            yaxis2=dict(title="Daily Close Price", overlaying="y", side="right"),
            legend=dict(x=1.02, y=1, borderwidth=1),
            template="plotly_white",
            hovermode="x unified",
            height=700,
            autosize=True,
        )
        fig.show()
