import numpy as np
import pandas as pd

def compute_windowed_rates(df, window, deltat=1):
    """
    Compute head/tail shift rates in sliding windows for each mol.
    
    Parameters
    ----------
    df : pd.DataFrame with columns ["time", "mol", "pos_head", "pos_tail"]
    window : int
        Window size in the same units as df["time"] (e.g. timesteps).
    deltat : int
        Time step spacing (default 1).
    Usage: eg, rates_windowed = compute_windowed_rates(filtered_df, window, deltat=deltat)
    
    Returns
    -------
    rates_df : pd.DataFrame with columns
        ["mol", "window_start", "window_end", "rate_head", "rate_tail"]
    """
    records = []

    for m, g in df.groupby("mol"):
        g = g.sort_values("time")
        times = g["time"].values

        if times[-1] - times[0] + deltat <= window:
            # trajectory too short → use whole span
            span = times[-1] - times[0] + deltat
            rate_head = (g["pos_head"]).sum() / span
            rate_tail = (g["pos_tail"]).sum() / span
            records.append((m, times[0], times[-1], rate_head, rate_tail))
        else:
            # sliding windows
            start = times[0]
            while start + window <= times[-1]:
                end = start + window
                sub = g[(g["time"] >= start) & (g["time"] < end)]
                span = window
                rate_head = (sub["pos_head"]).sum() / span
                rate_tail = (sub["pos_tail"]).sum() / span
                records.append((m, start, end, rate_head, rate_tail))
                start += window // 2   # e.g. half-overlap; adjust as you like

    return pd.DataFrame(records,
                        columns=["mol", "window_start", "window_end",
                                 "rate_head", "rate_tail"])
