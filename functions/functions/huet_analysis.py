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
    plt.ylabel(r"Mean Squared Displacement")
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Example: plot_huet_analysis(df[df['id'] == group_id], tau_fit=0.1, tau_dev=0.5)

import matplotlib.pyplot as plt
import numpy as np

def plot_huet_analysis_by_points(window_df, n_f, n_dev, title="Window Analysis"):
    """
    Visualizes the MSD, the linear fit, and the deviation check area
    using point-count logic.
    """
    # 1. Physical parameters
    coords = window_df[['x', 'y']].values
    dt = window_df['time'].iloc[1] - window_df['time'].iloc[0]
    N = len(window_df)
    
    # 2. Calculate MSD for all possible lags in the window
    # Max possible lag is N-1
    max_lag = N - 1
    lags = np.arange(1, max_lag + 1)
    msd_obs = calculate_msd(coords, lags) # Assumes your calculate_msd helper
    times = lags * dt
    
    # 3. Perform the analysis logic (mimicking your current function)
    # Fit only up to n_f
    fit_times = times[:n_f].reshape(-1, 1)
    fit_msd = msd_obs[:n_f]
    
    # Simple linear fit for visualization (or use your fit_huet_msd for weights)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(fit_times, fit_msd)
    
    # Predict over the whole range to show deviation
    msd_pred = model.predict(times.reshape(-1, 1))
    
    # 4. Plotting
    plt.figure(figsize=(8, 5))
    
    # Plot Observed MSD
    plt.plot(times, msd_obs, 'ko-', label="Observed MSD", alpha=0.7)
    
    # Plot the Linear Fit (the "Brownian" expectation)
    plt.plot(times, msd_pred, 'r--', label=f"Linear Fit (first {n_f} pts)")
    
    # Highlight the Fitting Zone
    plt.axvspan(0, n_f * dt, color='gray', alpha=0.15, label="Fitting Region")
    
    # Highlight the Deviation Point (at n_dev)
    plt.axvline(n_dev * dt, color='blue', linestyle=':', alpha=0.5, label=r"$n_{dev}$ point")
    plt.scatter(times[n_dev-1], msd_obs[n_dev-1], color='blue', s=100, zorder=5)
    
    # Formatting
    plt.title(r"{title}\n$N={N}, n_f={n_f}, n_{dev}={n_dev}$")
    plt.xlabel("Time (s)")
    plt.ylabel(r"MSD ")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

# Usage Example:
# sample_window = df[df["id"]==1].iloc[0:20]
# plot_huet_analysis_by_points(sample_window, n_f=4, n_dev=19)

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_huet_diagnostic_points(window_df, n_f, n_dev, title="Huet Point Analysis"):
    """
    Diagnostic plot showing the 'lever' (n_f) and the gaps at n_dev.
    Uses rf"" for combined f-string and raw-string latex support.
    """
    coords = window_df[['x', 'y']].values
    dt = window_df['time'].iloc[1] - window_df['time'].iloc[0]
    N = len(window_df)
    
    # 1. Calculate MSD
    lags = np.arange(1, n_dev + 1)
    msd_obs = calculate_msd(coords, lags) # Assumes your existing helper
    times = lags * dt
    
    # 2. Linear Fit (using n_f points)
    fit_times = times[:n_f].reshape(-1, 1)
    fit_msd = msd_obs[:n_f]
    model = LinearRegression().fit(fit_times, fit_msd)
    msd_pred = model.predict(times.reshape(-1, 1))
    
    # 3. Plotting
    plt.figure(figsize=(8, 5))
    
    # Plot Observed vs Predicted
    plt.plot(times, msd_obs, 'ko', label="Observed", markersize=4)
    plt.plot(times, msd_pred, color='red', linestyle='--', label="Linear Fit")
    
    # 4. Visualize the Deviation (The "Gaps")
    # Draw vertical lines from the fit to the observed points for the dev region
    for i in range(n_dev):
        color = 'blue' if i >= n_f else 'gray'
        alpha = 0.5 if i >= n_f else 0.2
        plt.vlines(times[i], msd_obs[i], msd_pred[i], color=color, alpha=alpha, linestyles='solid')

    # Highlight the specific n_dev point
    plt.scatter(times[n_dev-1], msd_obs[n_dev-1], facecolors='none', edgecolors='blue', s=150, lw=2, label=r"$n_{dev}$")

    # Formatting using rf"" strings for LaTeX
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(rf"MSD ($\mu m^2$)")
    
    # Add info text box
    info_text = (rf"$N_{{window}} = {N}$" + "\n" +
                 rf"$n_f = {n_f}$ points" + "\n" +
                 rf"$n_{{dev}} = {n_dev}$ points")
    
    plt.gca().text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.legend(loc='lower right')
    plt.grid(alpha=0.2)
    plt.show()

# Example Usage:
# plot_meta_analysis_points(all_configs, d_quantiles=(0.05, 0.95), dev_quantiles=(0.1, 0.9))

def plot_meta_analysis_points(all_configs, d_quantiles=(0.02, 0.98), dev_quantiles=(0.02, 0.98)):
    """
    Plots D and Dev distributions with modern Seaborn syntax and quantile truncation.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # --- 1. Filter and Plot Diffusion (D) ---
    d_min, d_max = all_configs['D'].quantile(d_quantiles)
    d_filtered = all_configs[(all_configs['D'] >= d_min) & (all_configs['D'] <= d_max)] # max(0, d_min)
    
    sns.violinplot(
        data=d_filtered, 
        x='param_set', 
        y='D', 
        hue='param_set',      # Assign x to hue
        ax=axes[0], 
        cut=0, 
        inner="quart", 
        palette="viridis",
        legend=False          # Disable legend to keep it clean
    )
    axes[0].set_title(f"Diffusion Coefficient D (Truncated {d_quantiles})")
    axes[0].set_ylabel(r"$D$ ")
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(0, color="k", lw=1)

    # --- 2. Filter and Plot Deviation (Dev) ---
    dev_min, dev_max = all_configs['Dev'].quantile(dev_quantiles)
    dev_filtered = all_configs[(all_configs['Dev'] >= dev_min) & (all_configs['Dev'] <= dev_max)]
    
    sns.violinplot(
        data=dev_filtered, 
        x='param_set', 
        y='Dev', 
        hue='param_set',      # Assign x to hue
        ax=axes[1], 
        cut=0, 
        inner="quart", 
        palette="magma",
        legend=False          # Disable legend
    )
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.6)
    axes[1].set_title(f"Deviation $Dev$ (Truncated {dev_quantiles})")
    axes[1].set_ylabel("Deviation Score")
    axes[1].set_xlabel("Parameter Set (N_window, n_f)")
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_specific_huet_result(result_row, original_df):
    """
    Illustrates a specific window analysis from a result dataframe row.
    
    Parameters:
    -----------
    result_row : pd.Series
        A single row from your analysis (e.g., all_configs.iloc[0])
    original_df : pd.DataFrame
        The raw trajectory dataframe containing ['id', 'time', 'x', 'y']
    """
    # 1. Extract metadata from the result row
    pid = result_row['id']
    start_idx = int(result_row['window_start_idx'])
    N = int(result_row['N_window'])
    n_f = int(result_row['n_f_used'])
    n_dev = int(result_row['n_dev_used'])
    
    # 2. Slice the original raw data
    # We filter by ID first, then slice the index
    raw_traj = original_df[original_df['id'] == pid].iloc[start_idx : start_idx + N]
    
    coords = raw_traj[['x', 'y']].values
    dt = raw_traj['time'].iloc[1] - raw_traj['time'].iloc[0]
    
    # 3. Reconstruct MSD and Fit
    lags = np.arange(1, n_dev + 1)
    msd_obs = calculate_msd(coords, lags) # Assumes your existing helper
    times = lags * dt
    
    fit_times = times[:n_f].reshape(-1, 1)
    fit_msd = msd_obs[:n_f]
    model = LinearRegression().fit(fit_times, fit_msd)
    msd_pred = model.predict(times.reshape(-1, 1))
    
    # 4. Plotting
    plt.figure(figsize=(9, 6))
    
    # Plot observed MSD points
    plt.plot(times, msd_obs, 'ko', markersize=5, label="Observed MSD", alpha=0.8)
    # Plot the predicted linear fit
    plt.plot(times, msd_pred, 'r--', linewidth=1.5, label="Linear Fit (Brownian Model)")
    
    # 5. Draw the "Deviation Gaps"
    # Gray lines for the fitting region, Blue for the deviation region
    for i in range(n_dev):
        is_dev = i >= n_f
        color = 'blue' if is_dev else 'gray'
        alpha = 0.6 if is_dev else 0.3
        plt.vlines(times[i], msd_obs[i], msd_pred[i], 
                   color=color, alpha=alpha, linestyles='solid', linewidth=1)

    # Highlight the specific n_dev point evaluated for the Dev score
    plt.scatter(times[n_dev-1], msd_obs[n_dev-1], facecolors='none', 
                edgecolors='blue', s=180, lw=2, zorder=5, label=rf"$n_{{dev}}$ point")

    # Final Styling
    title = rf"Analysis for ID: {pid} | Window Start: {start_idx}"
    plt.title(title, fontsize=12)
    plt.xlabel("Time ($s$)")
    plt.ylabel(rf"MSD ($\mu m^2$)")
    
    # Annotate with the actual calculated values
    stats_text = (rf"$D = {result_row['D']:.3f}\ \mu m^2/s$" + "\n" +
                  rf"$Dev = {result_row['Dev']:.3f}$" + "\n" +
                  rf"$n_f = {n_f}$ pts | $n_{{dev}} = {n_dev}$ pts")
    
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend(loc='lower right', frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# Pick the most interesting result (e.g., highest deviation)
# extreme_row = global_df.loc[global_df['Dev'].idxmin()] 
# plot_specific_huet_result(extreme_row, df)

import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory_window(result_row, original_df):
    """
    Plots the spatial XY trajectory for the specific window analyzed 
    in a result row.
    """
    # 1. Extract metadata
    pid = result_row['id']
    start_idx = int(result_row['window_start_idx'])
    N = int(result_row['N_window'])
    
    # 2. Slice the specific window
    # We filter by ID and reset index to ensure iloc matches the window_start_idx logic
    df_pid = original_df[original_df['id'] == pid].reset_index(drop=True)
    window = df_pid.iloc[start_idx : start_idx + N]
    
    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the full parent trajectory in faint grey for context
    ax.plot(df_pid['x'], df_pid['y'], color='grey', alpha=0.2, lw=1, label="Full Trajectory")
    
    # Plot the specific window in a bold color
    ax.plot(window['x'], window['y'], color='tab:blue', lw=2.5, zorder=4, label="Analyzed Window")
    
    # Mark Start (Circle) and End (Square) of the window
    ax.plot(window['x'].iloc[0], window['y'].iloc[0], 'go', 
            markersize=2, markeredgecolor='k', label="Window Start", zorder=5)
    ax.plot(window['x'].iloc[-1], window['y'].iloc[-1], 'rs', 
            markersize=2, markeredgecolor='k', label="Window End", zorder=5)
    
    # 4. Styling
    ax.set_aspect('equal')
    ax.set_xlabel(r"x ($\mu m$)")
    ax.set_ylabel(r"y ($\mu m$)")
    
    title = rf"Trajectory ID: {pid} | Window: {start_idx} to {start_idx + N}"
    ax.set_title(title)
    
    # Add stats box for context
    stats = rf"$D = {result_row['D']:.3f}$" + "\n" + rf"$Dev = {result_row['Dev']:.3f}$"
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.show()

# --- Usage ---
# fct.huet_analysis.plot_trajectory_window(sel_row, df)