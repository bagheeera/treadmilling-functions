from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Configuration Setup
# ==========================================
@dataclass
class PipelineConfig:
    # Cleaning & Preprocessing
    speed_cutoff: float = 20.0   # Max realistic speed threshold
    smooth_window: int = 5       # Rolling window size (in frames) for median smoothing
    
    # State Clustering & Filtering
    n_states: int = 3            # Number of mobility states (e.g., slow/med/fast)
    min_frames: int = 3          # Minimum run length to filter rapid boundary noise
    dt: float = 1.0              # Time step between consecutive frames
    random_seed: int = 42        # Seed for GMM reproducibility


# ==========================================
# 2. Data Cleaning & Smoothing Step
# ==========================================
def preprocess_and_smooth_speeds(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Cleans raw speed data by enforcing thresholds, isolating valid contiguous blocks 
    per trajectory, and applying centered rolling median smoothing.
    """
    # Create an explicit copy sorted chronologically per particle
    v = df.sort_values(["id", "time"]).copy()

    # Clean extreme outliers & invalid speed values
    v["speed_clean"] = v["speed"].where(
        np.isfinite(v["speed"]) & (v["speed"] <= config.speed_cutoff),
        np.nan
    )
    v["speed_valid"] = v["speed_clean"].notna()

    # Create new continuous block IDs separated by any invalid row per particle ID
    v["valid_block"] = (
        v.groupby("id")["speed_valid"]
         .transform(lambda x: (~x).cumsum())
    )

    # Perform centered median smoothing only within strictly valid continuous blocks
    v["speed_smooth"] = (
        v.groupby(["id", "valid_block"])["speed_clean"]
         .transform(
             lambda s: s.rolling(
                 window=config.smooth_window,
                 center=True,
                 min_periods=1
             ).median()
         )
    )

    # Ensure non-valid frames remain explicitly NaNs
    v.loc[~v["speed_valid"], "speed_smooth"] = np.nan

    return v


# ==========================================
# 3. State Clustering & Aggregation
# ==========================================
def fit_speed_gmm(df: pd.DataFrame, speed_col: str, config: PipelineConfig):
    """Fits GMM on raw smoothed speeds and ranks components: State 0 (slowest) -> N-1 (fastest)."""
    valid_mask = df[speed_col].notna()
    X = df.loc[valid_mask, speed_col].to_numpy().reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=config.n_states,
        random_state=config.random_seed
    )
    raw_labels = gmm.fit_predict(X)

    component_means = gmm.means_.ravel()
    component_order = np.argsort(component_means)

    label_map = np.empty(config.n_states, dtype=int)
    label_map[component_order] = np.arange(config.n_states)

    speed_state = pd.Series(pd.NA, index=df.index, dtype="Int64")
    speed_state.loc[valid_mask] = label_map[raw_labels]

    ordered_centers = component_means[component_order]
    
    return speed_state, ordered_centers


def compute_state_runs(df: pd.DataFrame, id_col: str, time_col: str, state_col: str) -> pd.Series:
    """Identifies continuous runs of identical states without corrupting row index."""
    df_sorted = df.sort_values([id_col, time_col])
    
    state = df_sorted[state_col]
    prev_state = df_sorted.groupby(id_col)[state_col].shift(1)

    new_run_mask = state.isna() | prev_state.isna() | state.ne(prev_state)
    run_ids = new_run_mask.cumsum()
    
    return run_ids.reindex(df.index)


def extract_lifetimes(
    df: pd.DataFrame, 
    id_col: str, 
    time_col: str, 
    speed_col: str, 
    state_col: str, 
    run_col: str, 
    config: PipelineConfig
) -> pd.DataFrame:
    """Extracts duration metrics and filters out flickering runs shorter than min_frames."""
    valid_df = df.dropna(subset=[state_col])
    
    lifetimes = (
        valid_df
        .groupby([id_col, run_col, state_col], observed=True)
        .agg(
            start_time=(time_col, "min"),
            end_time=(time_col, "max"),
            n_frames=(time_col, "size"),
            mean_speed=(speed_col, "mean"),
            median_speed=(speed_col, "median"),
        )
        .reset_index()
        .rename(columns={state_col: "speed_state"})
    )

    # Correct single-frame edge case (duration = n_frames * dt)
    lifetimes["duration"] = lifetimes["n_frames"] * config.dt

    # Filter short state runs
    lifetimes_filtered = lifetimes[lifetimes["n_frames"] >= config.min_frames].copy()

    return lifetimes_filtered


# ==========================================
# 4. Visualization Helpers
# ==========================================
def plot_lifetime_distribution(lifetimes_df: pd.DataFrame, max_speed: float = 4.0, max_dur: float = 100.0):
    """Renders KDE density map to remove discrete binning moiré patterns."""
    plt.figure(figsize=(9, 5))
    
    sns.kdeplot(
        data=lifetimes_df,
        x="mean_speed",
        y="duration",
        hue="speed_state",
        palette="viridis",
        fill=True,
        alpha=0.6,
        common_norm=False
    )
    
    plt.xlim(0, max_speed)
    plt.ylim(0, max_dur)
    plt.xlabel("Mean Speed during Run")
    plt.ylabel("Run Duration")
    plt.title("State Lifetime vs. Mean Speed")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

import pickle
from pathlib import Path

def save_analysis_results(
    # v_df: pd.DataFrame, 
    lifetimes_df: pd.DataFrame, 
    centers: np.ndarray, 
    config: PipelineConfig, 
    output_path: str = "speed_state_analysis.pkl"
):
    """
    Bundles and serializes the key trajectory dataframes, cluster parameters, 
    and analysis configuration into a single compressed PKL file.
    """
    output_data = {
        # Config & Metadata for 100% reproducibility
        "config": config,
        "state_centers": centers,
        
        # Microscopic point-by-point trajectories (with speed & state tags)
        # "trajectories": v_df,
        
        # Macroscopic run/state statistics (durations, speeds per segment)
        "lifetimes": lifetimes_df
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Successfully saved analysis bundle to '{output_path}'.")

def plot_speed_vs_dwelltime_hist(
    lifetimes_df: pd.DataFrame, 
    dt: float = 1.0, 
    max_speed: float = 4.0, 
    max_dur: float = 100.0,
    speed_bins_count: int = 50
):
    """
    Plots a 2D histogram of mean speed vs dwell time with duration bins 
    aligned exactly to discrete frame intervals to prevent moiré patterns.
    """
    plt.figure(figsize=(9, 6))

    # 1. Define explicit, physically aligned bin edges
    # Speed: continuous bins
    speed_bins = np.linspace(0, max_speed, speed_bins_count)
    
    # Dwell time: discrete bins aligned exactly to frame multiples (1 dt, 2 dt, 3 dt, ...)
    # Offset by 0.5 * dt so each integer frame count falls dead-center in its bin
    dur_bins = np.arange(0.5 * dt, max_dur + 1.5 * dt, dt)

    # 2. Render 2D Histogram
    counts, xedges, yedges, image = plt.hist2d(
        x=lifetimes_df["mean_speed"],
        y=lifetimes_df["duration"],
        bins=[speed_bins, dur_bins],
        norm=LogNorm(),  # Log scale highlights both short and long dwell times
        cmap="plasma"
    )

    # 3. Aesthetics
    cbar = plt.colorbar(image)
    cbar.set_label("Run Count (Log Scale)", rotation=270, labelpad=15)

    plt.xlim(0, max_speed)
    plt.ylim(0, max_dur)
    
    plt.xlabel("Mean Speed during Run")
    plt.ylabel("Dwell Time / Duration (frames or s)")
    plt.title("State Dwell Time vs. Mean Speed")
    plt.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.show()