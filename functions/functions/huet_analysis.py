import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_msd(coords, lags):
    """Calculates MSD for a given set of integer lags."""
    msds = []
    for n in lags:
        diff = coords[n:] - coords[:-n]
        sq_dist = np.sum(diff**2, axis=1)
        msds.append(np.mean(sq_dist))
    return np.array(msds)

def fit_huet_msd(msd_values, lags, dt, N_total, n_f_idx):
    """
    Fits weighted regression through points up to n_f_idx.
    """
    x_fit = (lags[:n_f_idx] * dt).reshape(-1, 1)
    y_fit = msd_values[:n_f_idx]
    
    # Huet weighting: 1 / V_rel
    n_range = lags[:n_f_idx]
    v_rel = (n_range * (2 * n_range**2 + 1)) / (N_total - n_range + 1)
    weights = 1 / v_rel
    
    model = LinearRegression(fit_intercept=True)
    model.fit(x_fit, y_fit, sample_weight=weights)
    return model

def calculate_huet_dev(msd_obs, msd_diff):
    """Calculates Dev based on the full range provided."""
    return (1 / len(msd_obs)) * np.sum((msd_obs - msd_diff) / msd_diff)

def analyze_by_time_lag(group, tau_fit=0.05, tau_dev=0.2):
    """
    Analyzes trajectory using time-based cutoffs.
    tau_fit: Max time lag for fitting (e.g., 0.05s)
    tau_dev: Max time lag for deviation calculation (e.g., 0.2s)
    """
    N = len(group)
    dt = group['time'].diff().iloc[1]
    
    # Convert time lags to point indices
    # n_f = int(max(2, np.floor(tau_fit / dt)))   # Minimum 2 points to fit a line
    # n_dev = int(max(n_f + 1, np.floor(tau_dev / dt)))

    # Use round() instead of floor() to handle 0.49999999 issues
    n_f = int(max(2, np.round(tau_fit / dt)))
    n_dev = int(max(n_f + 1, np.round(tau_dev / dt)))
    
    # DEBUG PRINT: Uncomment this to see why it fails inside the loop
    # print(f"Processing: N={N}, n_f={n_f}, n_dev={n_dev}")
    
    if N <= n_dev:
        return None # Trajectory too short for these time scales
    
    coords = group[['x', 'y']].values
    lags = np.arange(1, n_dev + 1)
    
    # 1. Calculate MSD
    msd_obs = calculate_msd(coords, lags)
    
    # 2. Weighted Fit (using points up to n_f)
    model = fit_huet_msd(msd_obs, lags, dt, N, n_f_idx=n_f)
    
    # 3. Predict for the dev range
    times = (lags * dt).reshape(-1, 1)
    msd_diff = model.predict(times)
    
    # 4. Dev score
    dev_score = calculate_huet_dev(msd_obs, msd_diff)
    
    return pd.Series({
        'D': model.coef_[0] / 4,
        'Dev': dev_score,
        'n_f_used': n_f,
        'n_dev_used': n_dev,
        'dt': dt
    })
    
def analyze_windowed_trajectory(group, window_size_sec, step_size_sec, tau_fit=0.05, tau_dev=0.2):
    """
    Sliding window analysis for a single particle trajectory.
    """
    results = []
    
    # Calculate dt from the group
    dt = group['time'].diff().iloc[1]
    
    # Convert time-based window/step to number of points
    window_pts = int(window_size_sec / dt)
    step_pts = int(step_size_sec / dt)
    
    # Minimum points needed to even attempt the Huet calc
    n_dev_required = int(np.floor(tau_dev / dt))
    
    if len(group) < window_pts:
        return pd.DataFrame()

    # Iterate through the trajectory in steps
    for start_idx in range(0, len(group) - window_pts + 1, step_pts):
        end_idx = start_idx + window_pts
        window_df = group.iloc[start_idx:end_idx].copy()
        
        # Calculate start time for this segment
        start_time = window_df['time'].iloc[0]
        
        # Run the Huet analysis on this segment
        # Using the function we built previously:
        res = analyze_by_time_lag(window_df, tau_fit=tau_fit, tau_dev=tau_dev)
        
        if res is not None:
            res['window_start_time'] = start_time
            res['particle_id'] = group['id'].iloc[0]
            results.append(res)
            
    return pd.DataFrame(results)

import matplotlib.pyplot as plt
import numpy as np

def plot_huet_analysis(group, tau_fit=0.05, tau_dev=0.2):
    """
    Visualization using time-based cutoffs.
    """
    N = len(group)
    dt = group['time'].diff().iloc[1]
    
    # Calculate indices based on time
    n_f = int(max(2, np.floor(tau_fit / dt)))
    n_dev = int(max(n_f + 1, np.floor(tau_dev / dt)))
    
    if N <= n_dev:
        print(f"Trajectory {group['id'].iloc[0]} too short for tau_dev={tau_dev}s")
        return

    # 1. Standard Analysis Steps
    coords = group[['x', 'y']].values
    lags = np.arange(1, n_dev + 1)
    times = lags * dt
    
    msd_obs = calculate_msd(coords, lags)
    model = fit_huet_msd(msd_obs, lags, dt, N, n_f_idx=n_f)
    
    msd_diff = model.predict(times.reshape(-1, 1))
    dev_score = calculate_huet_dev(msd_obs, msd_diff)
    d_coeff = model.coef_[0] / 4

    # 2. Plotting logic
    plt.figure(figsize=(10, 6))
    
    # All observed MSD points in the dev range
    plt.plot(times, msd_obs, 'o-', color='lightgray', alpha=0.4, label='MSD Path')
    
    # Points used for the Weighted Fit
    plt.scatter(times[:n_f], msd_obs[:n_f], color='forestgreen', s=70, 
                label=f'Fitting Window (up to {tau_fit}s)', zorder=5)
    
    # Points that contribute to the Deviation calculation
    plt.scatter(times[n_f:], msd_obs[n_f:], facecolors='none', edgecolors='royalblue', 
                s=70, label=f'Deviation Window (up to {tau_dev}s)', zorder=5)
    
    # The Regression Line (extrapolated to tau_dev)
    plt.plot(times, msd_diff, color='crimson', linestyle='--', linewidth=2, 
             label='Huet Linear Fit ($MSD = 4Dt + B$)')
    
    # Vertical lines representing the residuals used for 'Dev'
    for i in range(n_dev):
        plt.vlines(times[i], msd_obs[i], msd_diff[i], color='black', linestyle=':', alpha=0.3)

    # Info Box
    stats_text = (f"D: {d_coeff:.4f} "r"$\sigma^2/s$"
                  f"\nDev: {dev_score:.4f}\n"
                  r"$n_f$: "f"{n_f} pts\n"
                  r"$n_{{dev}}$: "f"{n_dev} pts")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)

    plt.title(f"Time-Based Huet Analysis | Trajectory ID: {group['id'].iloc[0]}")
    plt.xlabel(r"Lag Time $\tau$ (seconds)")
    plt.ylabel(r"Mean Squared Displacement ($\mu m^2$)")
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Example: plot_huet_analysis(df[df['id'] == group_id], tau_fit=0.1, tau_dev=0.5)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_meta_analysis_points(all_configs, d_quantiles=(0.02, 0.98), dev_quantiles=(0.02, 0.98)):
    """
    Plots D and Dev distributions, truncating data based on quantiles 
    to handle outliers and simplify the ylim situation.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # --- 1. Filter and Plot Diffusion (D) ---
    d_min, d_max = all_configs['D'].quantile(d_quantiles)
    # Ensure we don't truncate D below 0
    d_min = max(0, d_min)
    
    d_filtered = all_configs[(all_configs['D'] >= d_min) & (all_configs['D'] <= d_max)]
    
    sns.violinplot(
        data=d_filtered, x='param_set', y='D', 
        ax=axes[0], cut=0, inner="quart", palette="viridis"
    )
    axes[0].set_title(f"Diffusion Coefficient $D$ (Truncated {d_quantiles})")
    axes[0].set_ylabel("$D$ ($\mu m^2/s$)")
    axes[0].grid(axis='y', alpha=0.3)

    # --- 2. Filter and Plot Deviation (Dev) ---
    dev_min, dev_max = all_configs['Dev'].quantile(dev_quantiles)
    dev_filtered = all_configs[(all_configs['Dev'] >= dev_min) & (all_configs['Dev'] <= dev_max)]
    
    sns.violinplot(
        data=dev_filtered, x='param_set', y='Dev', 
        ax=axes[1], cut=0, inner="quart", palette="magma"
    )
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.6)
    axes[1].set_title(f"Deviation $Dev$ (Truncated {dev_quantiles})")
    axes[1].set_ylabel("Deviation Score")
    axes[1].set_xlabel("Parameter Set (N_window, n_f)")
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example Usage:
# plot_meta_analysis_points(all_configs, d_quantiles=(0.05, 0.95), dev_quantiles=(0.1, 0.9))

def analyze_by_points(group, n_f, n_dev):
    """
    Huet analysis using explicit point indices.
    n_f: number of points for the linear fit (e.g., 2, 3, 4)
    n_dev: the maximum lag index to check for deviation
    """
    N = len(group)
    dt = group['time'].iloc[1] - group['time'].iloc[0]
    
    # Physical safety check: N must be at least n_dev + 1
    if N < n_dev + 1:
        return None

    coords = group[['x', 'y']].values
    lags = np.arange(1, n_dev + 1)
    times = lags * dt
    
    # Calculate MSD and Huet Fit
    msd_obs = calculate_msd(coords, lags)
    
    # Pass n_f as the index for the weighted regression
    model = fit_huet_msd(msd_obs, lags, dt, N_total=N, n_f_idx=n_f)
    
    msd_pred = model.predict(times.reshape(-1, 1))
    dev_score = calculate_huet_dev(msd_obs, msd_pred)
    d_coeff = model.coef_[0] / 4

    return {
        'D': d_coeff,
        'Dev': dev_score,
        'n_f_used': n_f,
        'n_dev_used': n_dev,
        'N_window': N,
        'intercept': model.intercept_
    }

def analyze_windowed_trajectory_points(group, window_pts, step_pts, n_f, n_dev):
    results = []
    
    if len(group) < window_pts:
        return pd.DataFrame()

    for start in range(0, len(group) - window_pts + 1, step_pts):
        window_df = group.iloc[start : start + window_pts]
        res = analyze_by_points(window_df, n_f, n_dev)
        
        if res:
            res['window_start_idx'] = start
            res['id'] = group['id'].iloc[0]
            results.append(res)
            
    return pd.DataFrame(results)