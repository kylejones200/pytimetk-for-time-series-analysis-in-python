import signalplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

np.random.seed(42)
signalplot.apply(font_family='serif')




@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"


def load_frame(cfg: Config) -> pd.DataFrame:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    df = df.dropna().sort_values("date")
    return df


def main(plot: bool = False):
    cfg = Config()
    df = load_frame(cfg)

    used_pytimetk = False
    # Attempt to use pytimetk to augment calendar features and produce a small summary
    try:
        import pytimetk as tkt

        # If available, create a simple feature set via built-in augmentation
        # Prefer a general API name; fall back gracefully if attribute names differ
        if hasattr(tkt, "augment_time_series_signature"):
            dff = tkt.augment_time_series_signature(df, date_col="date")
        elif hasattr(tkt, "augment_timeseries_signature"):
            dff = tkt.augment_timeseries_signature(df, date_col="date")
        else:
            # basic calendar features
            dff = df.copy()
            dff["month"] = dff["date"].dt.month
            dff["year"] = dff["date"].dt.year
        # Compute yoy change using groupby if signature present or from base df
        dff = dff.sort_values("date")
        dff["yoy_pct"] = dff["value"].pct_change(12) * 100.0
        used_pytimetk = True
    except Exception:
        # Fallback: manual calendar features and yoy
        dff = df.copy()
        dff["month"] = dff["date"].dt.month
        dff["year"] = dff["date"].dt.year
        dff["yoy_pct"] = dff["value"].pct_change(12) * 100.0

    # Minimalist visualization compatible with the article
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        ax[0].plot(dff["date"], dff["value"], label="EIA monthly")
        ax[0].legend()
        ax[1].plot(dff["date"], dff["yoy_pct"], color="tab:orange", label="YoY %")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].legend()
        signalplot.save("eia_pytimetk_viz.png")


if __name__ == "__main__":
    main()
