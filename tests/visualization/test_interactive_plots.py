from src.pipeline.seasonality_etl import SeasonalityETL
from src.visualization.interactive_plots import show_interactive_seasonality_plot
from src.visualization.synthetic_data_generator import (
    generate_perfect_seasonality_all_intervals,
)


def test_interactive_plot_can_run_on_synthetic(monkeypatch):
    # Patch out the actual Plotly render
    monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)

    # Generate valid synthetic input
    df_synth = generate_perfect_seasonality_all_intervals(seed=123)
    etl = SeasonalityETL()
    df_rolling = etl.fit_rolling(df_synth, frequencies=["YE"])

    # Assert plotting runs without error
    show_interactive_seasonality_plot(df_rolling, df_synth)
