from matplotlib.lines import Line2D
import numpy as np


from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def scatter_fct(
    df,
    ax,
    t_frame,
    
    # Particle type configuration (flexible)
    particle_config=None,
    
    # Display options
    display_time=True,
    display_legend=True,
    
    # coordinate scaling
    scale_xy=5,
    
    # axis limits
    xlim=None,
    ylim=(-150, 150),
    
    # quantile lines
    show_quantiles=True,
    quantile_lines=None,
    quantile_range=(-30, 30),
    quantiles=(0.2, 0.8),
    
    # legend options
    legend_loc="lower center",
    legend_ncol=3,
    
    # axis label / ticks
    hideticklabels=True,
    xlabel="Cell circumference (nm)",
    ylabel="Long cell axis (nm)",
    
    # histogram options
    histogram_config=None,
    histogram_axis=None,
    histogram_bins=30,
    histogram_range=None,
    hist_timeavg_range=None,
    histogram_xrange=None, # only bin a given region along x
):
    """
    Flexible scatter plot with optional histogram.
    
    histogram_config : list of dict or None
        [
            {'types': [1,2,3], 'color': '#4cc9f0', 'label': 'FtsZ'},
            {'types': [5,6], 'color': '#f72585', 'label': 'Synthase'},
        ]
    histogram_axis : matplotlib axis or None
        If provided, plot histograms on this axis
    histogram_bins : int
        Number of bins for histogram
    histogram_range : tuple (min, max) or None
        Y-range for histogram filtering (default: quantile_range)
    """
    
    # --- Default configuration ---
    if particle_config is None:
        particle_config = {
            'filaments': {
                'types': [1, 2, 3],
                'color': '#4cc9f0',
                'marker': 'o',
                's': 10,
                'label': 'FtsZ',
                'plot': True,
                'trace': False,
                'trace_window': 0,
                'arrow': False,
            },
            'synthase': {
                'types': [5, 6],
                'color': '#f72585',
                'marker': 's',
                's': 20,
                'label': 'Synthase',
                'plot': True,
                'trace': False,
                'trace_window': 0,
                'arrow': False,
            },
            'processive': {
                'types': [5, 9],
                'color': '#f72585',
                'marker': 's',
                's': 20,
                'label': 'Processive',
                'plot': True,
                'alpha': 1,
                'trace': True,
                'trace_window': 0,
                'arrow': False,
            },
            'activator': {
                'types': [8],
                'color': '#023047',
                'marker': 'v',
                's': 20,
                'label': 'Activator',
                'plot': False,
                'trace': False,
                'trace_window': 0,
                'arrow': False,
            },
        }
    
    if quantile_lines is None:
        quantile_lines = {
            'synthase': {
                'types': [5, 6],
                'color': '#f72585',
                'axis': 'y',
                'alpha': 0.7,
                'ls': '--',
                'show': True,
            }
        }
    
    if histogram_range is None:
        histogram_range = quantile_range
    
    # --- Extract frame ---
    D = df.loc[df["time"] == t_frame]
    
    # --- Plot each particle type ---
    for config_name, config in particle_config.items():
        if not config.get('plot', True):
            continue
        
        types_to_plot = config['types']
        D_particles = D.loc[D["type"].isin(types_to_plot)]
        
        if D_particles.empty:
            continue
        
        # Current frame scatter
        ax.scatter(
            *(D_particles[["x", "y"]].values.T * scale_xy),
            c=config['color'],
            s=config['s'],
            marker=config['marker'],
            label=config['label'],
            alpha=config.get("alpha", 1),
        )
        
        # Trace (previous frames)
        if config.get('trace', False):
            trace_window = config.get('trace_window', 0)
            if trace_window > 0:
                prev_times = [t_frame - i for i in range(1, trace_window + 1)]
                for j, tp in enumerate(prev_times):
                    D_prev = df.loc[(df["time"] == tp) & (df["type"].isin(types_to_plot))]
                    if D_prev.empty:
                        continue
                    alpha = 1 - (j + 1) / (trace_window + 1)
                    ax.scatter(
                        *(D_prev[["x", "y"]].values.T * scale_xy),
                        c=config['color'],
                        s=config['s'],
                        marker=config['marker'],
                        alpha=alpha
                    )
    
    # --- Plot arrows (separate pass, after all particles) ---
    for config_name, config in particle_config.items():
        if not config.get('arrow', False):
            continue
        
        arrow_orientation_func = config.get('arrow_orientation')
        arrow_length = config.get('arrow_length', 50)
        arrow_width = config.get('arrow_width', 0.5)
        arrow_head_width = config.get('arrow_head_width', 5)
        arrow_head_length = config.get('arrow_head_length', 5)
        
        if arrow_orientation_func is None:
            continue
        
        types_to_plot = config['types']
        D_particles = D.loc[D["type"].isin(types_to_plot)]
        
        if D_particles.empty:
            continue
        
        # For each particle, get orientation and draw arrow
        for idx, particle in D_particles.iterrows():
            orientation = arrow_orientation_func(particle, df, t_frame)
            if orientation is None:
                continue
            
            if isinstance(orientation, pd.DataFrame):
                print(f"Warning: arrow_orientation returned DataFrame instead of angle/tuple")
                continue
            
            if isinstance(orientation, (tuple, list)) and len(orientation) == 2:
                try:
                    dx = float(orientation[0])
                    dy = float(orientation[1])
                except (ValueError, TypeError):
                    continue
                
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > 0:
                    dx = (dx / magnitude) * arrow_length
                    dy = (dy / magnitude) * arrow_length
                else:
                    continue
            else:
                try:
                    angle_rad = np.radians(float(orientation))
                    dx = arrow_length * np.cos(angle_rad)
                    dy = arrow_length * np.sin(angle_rad)
                except (ValueError, TypeError):
                    continue
            
            ax.arrow(
                float(particle['x'] * scale_xy),
                float(particle['y'] * scale_xy),
                float(dx),
                float(dy),
                head_width=arrow_head_width,
                head_length=arrow_head_length,
                fc=config['color'],
                ec=config['color'],
                alpha=0.6,
                width=arrow_width
            )
    
    # --- Time text ---
    if display_time:
        txt = ax.text(0.8, 0.8, f"t = {round(t_frame)} s",
                      transform=ax.transAxes, ha="center", va="center")
        txt.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
    
    # --- Quantile lines ---
    for ql_name, ql_config in quantile_lines.items():
        if not ql_config.get('show', True):
            continue
        
        types_for_quantiles = ql_config['types']
        D_ql = D.loc[D["type"].isin(types_for_quantiles)]
        
        if D_ql.empty:
            continue
        
        axis = ql_config.get('axis', 'y')
        
        if axis == 'y':
            yvals = D_ql.loc[(np.abs(D_ql["y"]) < abs(ql_config.get('quantile_range', quantile_range)[1]))]["y"] * scale_xy
            for q in quantiles:
                qv = np.quantile(yvals, q)
                if show_quantiles:
                    ax.axhline(qv, ls=ql_config.get('ls', '--'),
                            lw=1, color=ql_config['color'],
                            alpha=ql_config.get('alpha', 0.7))
        elif axis == 'x':
            xvals = D_ql.loc[(np.abs(D_ql["x"]) < abs(ql_config.get('quantile_range', quantile_range)[1]))]["x"] * scale_xy
            for q in quantiles:
                qv = np.quantile(xvals, q)
                if show_quantiles:
                    ax.axvline(qv, ls=ql_config.get('ls', '--'),
                            lw=1, color=ql_config['color'],
                            alpha=ql_config.get('alpha', 0.7))
    
    # --- Axes ---
    ax.set_aspect("equal")
    if xlim is None:
        xlim = (df.loc[df["time"] == t_frame]["x"].min() * scale_xy, 
                df.loc[df["time"] == t_frame]["x"].max() * scale_xy)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    if hideticklabels:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # --- Legend ---
    if display_legend:
        handles = []
        for config_name, config in particle_config.items():
            if config.get('plot', True):
                legend_size = config.get('legend_markersize', np.sqrt(config['s']))
                handles.append(Line2D([0], [0],
                                      label=config['label'],
                                      c=config['color'],
                                      markersize=legend_size,
                                      marker=config['marker'],
                                      linestyle=""))
        
        if handles:
            ax.legend(handles=handles,
                      bbox_to_anchor=(0.5, 1.02),
                      loc=legend_loc,
                      ncol=legend_ncol,
                      frameon=True)
    
    # --- Histograms ---
    # --- Histograms ---
    if histogram_axis is not None and histogram_config is not None:
        for hist_cfg in histogram_config:
            hist_types = hist_cfg['types']
            hist_color = hist_cfg['color']
            hist_label = hist_cfg['label']
            display_hist_legend = hist_cfg.get('display_legend', True)

            # --- NEW: Allow time averaging over a defined range ---
            if hist_timeavg_range is not None and len(hist_timeavg_range) == 2:
                t_start, t_end = hist_timeavg_range
                df_hist = df[
                    (df["type"].isin(hist_types)) &
                    (df["time"].between(t_start, t_end))
                ]
            else:
                # Regular single time frame
                df_hist = df[
                    (df["type"].isin(hist_types)) &
                    (df["time"] == t_frame)
                ]
                
            if histogram_xrange is not None:
                df_hist = df_hist[df_hist["x"].between(histogram_xrange[0], histogram_xrange[1])]

            # Restrict y to desired range
            df_hist = df_hist[df_hist["y"].between(histogram_range[0], histogram_range[1])]
            if not df_hist.empty:
                histogram_axis.hist(
                    df_hist["y"] * scale_xy,
                    bins=histogram_bins,
                    histtype="step",
                    color=hist_color,
                    lw=2,
                    density=True,
                    alpha=0.7,
                    orientation="horizontal",
                    label=hist_label
                )

        histogram_axis.set_xlabel("Density")
        # histogram_axis.set_yticks([])
        histogram_axis.set_yticklabels([])
        if display_hist_legend:
            histogram_axis.legend()
        histogram_axis.set_ylim(ax.get_ylim())
        histogram_axis.set_aspect('auto')
    
    return ax


"""
Minimal arrow orientation functions.
"""
import pandas as pd


def make_arrow_orientation_by_molecule(
    parameters,
    source_type_key="complex_ptypeA",
    target_type_key="complex_ptypeB",
):
    """
    Create arrow orientation function that points from source to target particles.
    
    Parameters
    ----------
    parameters : dict
        Contains type IDs: {"complex_ptypeA": 5, "complex_ptypeB": 6}
    source_type_key : str
        Key for source type in parameters
    target_type_key : str
        Key for target type in parameters
    
    Returns
    -------
    function
        arrow_orientation(particle, df, t_frame) -> (dx, dy) or None
    """
    
    target_type = parameters[target_type_key]
    
    def arrow_orientation(particle, df, t_frame):
        # Extract values directly from particle Series
        try:
            mol_id = particle.get('mol')
            source_x = particle.get('x')
            source_y = particle.get('y')
        except:
            return None
        
        # Check for NaN
        if pd.isna(mol_id) or pd.isna(source_x) or pd.isna(source_y):
            return None
        
        # Convert to float
        try:
            mol_id = float(mol_id)
            source_x = float(source_x)
            source_y = float(source_y)
        except (ValueError, TypeError):
            return None
        
        # Find targets with same molecule ID
        targets = df[
            (df["time"] == t_frame) 
            & (df["mol"] == mol_id) 
            & (df["type"] == target_type)
        ]
        
        if targets.empty:
            return None
        
        # Get first target
        target = targets.iloc[0]
        try:
            target_x = float(target['x'])
            target_y = float(target['y'])
        except (ValueError, TypeError, KeyError):
            return None
        
        # Return direction vector
        dx = target_x - source_x
        dy = target_y - source_y
        
        # Print debug info (can remove later)
        # print(f"Arrow: ({source_x:.1f}, {source_y:.1f}) -> ({target_x:.1f}, {target_y:.1f}) = ({dx:.1f}, {dy:.1f})")
        
        return (dx, dy)
    
    return arrow_orientation


def make_arrow_orientation_bidirectional(
    parameters,
    type_a_key="complex_ptypeA",
    type_b_key="complex_ptypeB",
):
    """
    Create bidirectional arrow function (A ↔ B).
    """
    
    type_a = parameters[type_a_key]
    type_b = parameters[type_b_key]
    
    def arrow_orientation(particle, df, t_frame):
        # Extract values directly from particle Series
        try:
            mol_id = particle.get('mol')
            current_type = int(particle.get('type'))
            source_x = particle.get('x')
            source_y = particle.get('y')
        except:
            return None
        
        # Check for NaN
        if pd.isna(mol_id) or pd.isna(source_x) or pd.isna(source_y):
            return None
        
        # Convert to float
        try:
            mol_id = float(mol_id)
            source_x = float(source_x)
            source_y = float(source_y)
        except (ValueError, TypeError):
            return None
        
        # Determine target type
        target_type = type_b if current_type == type_a else type_a
        
        # Find targets
        targets = df[
            (df["time"] == t_frame) 
            & (df["mol"] == mol_id) 
            & (df["type"] == target_type)
        ]
        
        if targets.empty:
            return None
        
        # Get first target
        target = targets.iloc[0]
        try:
            target_x = float(target['x'])
            target_y = float(target['y'])
        except (ValueError, TypeError, KeyError):
            return None
        
        # Return direction vector
        dx = target_x - source_x
        dy = target_y - source_y
        
        return (dx, dy)
    
    return arrow_orientation



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# from fct.cmplx_scatter import scatter_fct
def render_movie_with_hist(
    df,
    tag,
    trace_types=None,
    particle_config=None,
    histogram_config=None,
    with_histogram=True,
    N_show=10,
    duration=10,
    fps=30,
    dpi=150,
    ylim=(-150, 150),
    xlim=(-200, 200),
    histogram_bins=30,
    histogram_range=None,
    width_ratios=(3, 1),
    wspace=0.05,
    height_scale=0.45,
    hist_xlim=0.021,
    display_time=True,
    display_legend=True,
    figsize=(12, 5),
    inactive_color="#f478aaff",
    inactive_alpha=0.9,
    trace_fade_time=None,     # seconds of visible trace)
):
    """
    Render an animation of particle positions with optional histogram and trajectory tracing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns such as 'time', 'x', 'y', 'type', and 'id'.
    tag : str
        Path or base filename for the output video ('.mp4' will be appended).
    trace_types : list[int] or None
        Particle type(s) whose motion traces will be drawn over time.
        If None, no traces are drawn.
    particle_config : dict or None
        Configuration mapping particle type groups to visualization style:
        {
            "filaments": {"types": [...], "color": ..., "marker": ..., "s": ..., ...},
            "synthase":  {"types": [...], "color": ..., ...}
        }
    histogram_config : list[dict] or None
        Histogram config list [{'types': [...], 'color': '...', 'label': '...'}, ...].
    with_histogram : bool
        Whether to draw the histogram side panel.
    N_show : int
        Maximum number of traced IDs to show (randomly chosen among trace_types).
    duration, fps, dpi : numeric
        Movie length, frame rate, and figure DPI.
    xlim, ylim : tuple (min, max)
        Axes limits in nm.
    histogram_bins, histogram_range, width_ratios, wspace, height_scale, hist_xlim : misc histogram parameters.
    display_time, display_legend : bool
        Whether to show a time stamp and legend.
    figsize : tuple
        Figure size (width, height) in inches.
    inactive_color, inactive_alpha : style
        Color and alpha for particles not currently of a traced type.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage

    # ---------------- Default configs ----------------
    if particle_config is None:
        print("using default particle settings dict")
        particle_config = {
            "filaments": {
                "types": [1, 2, 3],
                "color": "#4cc9f0",
                "marker": "o",
                "s": 10,
                "zorder": 1,
                "alpha": 1.0,
                "label": "FtsZ",
            },
            "synthase": {
                "types": [5, 6, 11],
                "color": "#f72585",
                "marker": "s",
                "s": 20,
                "zorder": 2,
                "alpha": 1.0,
                "label": "Synthase",
                "trace_color": "#397387ff",
                "trace_alpha": 0.7,
                "trace_zorder": 1.5,
            },
        }

    if histogram_config is None:
        histogram_config = [
            {"types": [5, 6, 11], "color": "#f72585", "label": "Synthase"},
            {"types": [1, 2, 3], "color": "#4cc9f0", "label": "FtsZ"},
        ]

    plt.rcParams["figure.dpi"] = dpi
    if histogram_range is None:
        histogram_range = ylim

    T_tot = df["time"].values[-1]
    SCALE = 5.0

    # Pick IDs to trace if trace_types requested
    if trace_types:
        subset = df[df["type"].isin(trace_types)]
        unique_ids = subset["id"].unique()
        if len(unique_ids) > N_show:
            rng = np.random.default_rng(42)
            show_ids = rng.choice(unique_ids, size=N_show, replace=False)
        else:
            show_ids = unique_ids
        print(f"Selected {len(show_ids)} IDs for tracing: {show_ids}")
        df_trace = df[df["id"].isin(show_ids)]
    else:
        show_ids = []
        df_trace = pd.DataFrame(columns=df.columns)

    # Build quick type→style lookup
    base_color_map, trace_color_map = {}, {}
    marker_map, size_map, zorder_map = {}, {}, {}
    label_map, alpha_map = {}, {}
    lw_map, trace_alpha_map, trace_zorder_map = {}, {}, {}
    for name, cfg in particle_config.items():
        for ttp in cfg.get("types", []):
            base_color_map[ttp] = cfg.get("color", "#397387ff")
            trace_color_map[ttp] = cfg.get("trace_color", base_color_map[ttp])
            marker_map[ttp] = cfg.get("marker", "o")
            size_map[ttp] = cfg.get("s", 10)
            zorder_map[ttp] = cfg.get("zorder", 1)
            alpha_map[ttp] = cfg.get("alpha", 1.0)
            label_map[ttp] = cfg.get("label", str(ttp))
            lw_map[ttp] = cfg.get("lw", 2.5)
            trace_alpha_map[ttp] = cfg.get("trace_alpha", .4)
            trace_zorder_map[ttp] = cfg.get("trace_zorder", zorder_map[ttp] - 0.5)

    # --- Figure layout ---
    if with_histogram:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios, wspace=wspace)
        ax_scatter = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_scatter)
    else:
        fig, ax_scatter = plt.subplots(figsize=figsize)
        ax_hist = None

    ax_scatter.set_xlim(xlim)
    ax_scatter.set_ylim(ylim)
    ax_scatter.set_xlabel("Cell circumference (nm)")
    ax_scatter.set_ylabel("Long cell axis (nm)")

    def clear_ax_artists(ax):
        for coll in list(ax.collections):
            coll.remove()
        for ln in list(ax.lines):
            ln.remove()
        for p in list(ax.patches):
            p.remove()
        for txt in list(ax.texts):
            txt.remove()

    # --- Animation frame generator ---
    def make_frame(t_video):
        t_data = T_tot * t_video / duration
        closest_time = df["time"].values[np.abs(df["time"].values - t_data).argmin()]
        clear_ax_artists(ax_scatter)
        if with_histogram and ax_hist is not None:
            clear_ax_artists(ax_hist)

        # Subset of this time
        frame_df = df[df["time"] == closest_time]

        # Only display types defined and marked "plot": True in particle_config
        visible_types = [
            ttp
            for cfg in particle_config.values()
            for ttp in cfg.get("types", [])
            if cfg.get("plot", True)
        ]

        for ttype in sorted([t for t in frame_df["type"].unique() if t in visible_types]):
            sub = frame_df[frame_df["type"] == ttype]
            if sub.empty:
                continue
            # If using trace_types, mark "inactive" color for traced IDs of non-active types
            is_traced_type = ttype in trace_types if trace_types else False
            ax_scatter.scatter(
                SCALE * sub["x"], SCALE * sub["y"],
                c=base_color_map.get(ttype, "#397387ff"),
                s=size_map.get(ttype, 10),
                marker=marker_map.get(ttype, "o"),
                zorder=zorder_map.get(ttype, 1),
                alpha=alpha_map.get(ttype, 1.0),
                edgecolors="none",
                label=label_map.get(ttype, str(ttype))
            )

        # ---- Draw traces if requested ----
        if trace_types and len(show_ids) > 0:
            MAX_DIST = 25.0
            for pid in show_ids:
                # select data up to current frame
                pdata = df_trace[
                    (df_trace["id"] == pid)
                    & (df_trace["time"] <= closest_time)
                    & (df_trace["type"].isin(trace_types))
                ].copy()

                # Apply trace fade lifetime
                if trace_fade_time is not None:
                    pdata = pdata[pdata["time"] >= closest_time - trace_fade_time]

                if pdata.empty:
                    continue
                pdata = pdata.sort_values("time")
                px = SCALE * pdata["x"].values
                py = SCALE * pdata["y"].values

                diffs = np.sqrt(np.diff(px)**2 + np.diff(py)**2)
                break_idx = np.where(diffs > MAX_DIST)[0]
                segs, start = [], 0
                for b in break_idx:
                    segs.append(slice(start, b + 1))
                    start = b + 1
                segs.append(slice(start, len(px)))

                ptype = pdata["type"].iloc[0]
                color = trace_color_map.get(ptype, "#397387ff")
                lw = lw_map.get(ptype, 2.5)
                talpha = trace_alpha_map.get(ptype, 1.0)
                tz = trace_zorder_map.get(ptype, 1.5)

                for seg in segs:
                    if seg.stop - seg.start < 2:
                        continue
                    ax_scatter.plot(
                        px[seg], py[seg],
                        lw=lw, color=color,
                        alpha=talpha, zorder=tz,
                    )

        # ---- Optional histogram side plot ----
        if with_histogram and ax_hist is not None:
            scatter_fct(
                df,
                ax_scatter,
                closest_time,
                particle_config=particle_config,
                display_time=False,
                display_legend=False,
                scale_xy=SCALE,
                ylim=ylim,
                histogram_axis=ax_hist,
                histogram_config=histogram_config,
                histogram_bins=histogram_bins,
                histogram_range=histogram_range,
                xlim=xlim,
            )
            ax_hist.set_xlim(0, hist_xlim)
            ax_hist.set_xticklabels([])
            ax_hist.set_yticks([])
            ax_hist.set_xlabel("Density")
            ax_hist.set_aspect('auto')

        # ---- Time text & legend ----
        if display_time:
            ax_scatter.text(
                0.8, 0.9, f"t = {closest_time:.0f} s",
                transform=ax_scatter.transAxes,
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="white"),
            )
        if display_legend:
            handles, labels = ax_scatter.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_scatter.legend(by_label.values(), by_label.keys(), frameon=False,
            loc="upper left")

        fig.canvas.draw_idle()
        return mplfig_to_npimage(fig)

    # --- Render video ---
    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile(f"{tag}.mp4", fps=fps, codec="libx264", audio=False)


# tested in /nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/0__ring/M__cmplx_cnstrct_intermediate/A_staticring/B__arrest/notebooks/processive_traces_movie.ipynb
def render_movie_with_trace(
    df,
    tag,
    trace_types,
    particle_config=None,
    N_show=10,
    duration=10,
    fps=30,
    dpi=150,
    ylim=(-150, 150),
    xlim=None,
    display_time=True,
    display_legend=True,
    figsize=(8, 6),
    inactive_color="#f478aaff",
    inactive_alpha=0.9,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    if particle_config is None:
        particle_config = {
            "filaments": {
                "types": [1, 2, 3],
                "color": "#4cc9f0",
                "marker": "o",
                "s": 18,
                "zorder": 1,
                "alpha": 1.0,
                "label": "FtsZ",
            },
            "synthase": {
                "types": [11],
                "color": "#f72585",
                "marker": "s",
                "s": 23,
                "zorder": 2,
                "alpha": 1.0,
                "label": "Synthase",
                "trace_color": "#397387ff",
                "trace_alpha": 0.7,
                "trace_zorder": 1.5,
            },
        }

    # ---------------- Randomly select trace‑type IDs ----------------
    subset = df[df["type"].isin(trace_types)]
    unique_ids = subset["id"].unique()
    if len(unique_ids) > N_show:
        rng = np.random.default_rng(seed=42)
        show_ids = rng.choice(unique_ids, size=N_show, replace=False)
    else:
        show_ids = unique_ids
    print(f"Selected {len(show_ids)} particle ids for tracing:", show_ids)

    background = df[df["type"].isin([1, 2, 3])]
    selected = df[df["id"].isin(show_ids)]
    df_show = pd.concat([background, selected]).drop_duplicates().sort_values("time")
    df_trace = selected.copy()

    # ---------------- Style lookups ----------------
    base_color_map, trace_color_map = {}, {}
    marker_map, size_map, zorder_map = {}, {}, {}
    label_map, lw_map, alpha_map = {}, {}, {}
    trace_alpha_map, trace_zorder_map = {}, {}

    for name, cfg in particle_config.items():
        for ttp in cfg.get("types", []):
            base_color_map[ttp]    = cfg.get("color", "#397387ff")
            trace_color_map[ttp]   = cfg.get("trace_color", base_color_map[ttp])
            marker_map[ttp]        = cfg.get("marker", "o")
            size_map[ttp]          = cfg.get("s", 10)
            zorder_map[ttp]        = cfg.get("zorder", 1)
            alpha_map[ttp]         = cfg.get("alpha", 1.0)
            label_map[ttp]         = cfg.get("label", str(ttp))
            lw_map[ttp]            = cfg.get("lw", 2.5)
            trace_alpha_map[ttp]   = cfg.get("trace_alpha", 1.0)
            trace_zorder_map[ttp]  = cfg.get("trace_zorder", zorder_map[ttp] - 0.5)
    synthase_s = particle_config["synthase"]["s"]
    # ---------------- Figure setup ----------------
    plt.rcParams["figure.dpi"] = dpi
    fig, ax = plt.subplots(figsize=figsize)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Cell circumference (nm)")
    ax.set_ylabel("Long cell axis (nm)")

    t_min = df_show["time"].values[0]
    t_max = df_show["time"].values[-1]
    T_range = t_max - t_min

    def clear_ax_artists(ax):
        for c in list(ax.collections): c.remove()
        for l in list(ax.lines):       l.remove()
        for t in list(ax.texts):       t.remove()

    def make_frame(t_video):
        t_data = t_min + T_range * (t_video / duration)
        closest_time = df_show["time"].values[
            np.abs(df_show["time"].values - t_data).argmin()
        ]
        clear_ax_artists(ax)

        frame_df = df_show[df_show["time"] == closest_time]

        # ---- Scatter: normal particles (excluding selected IDs entirely) ----
        normal_df = frame_df[~frame_df["id"].isin(show_ids)]
        for ttype in sorted(normal_df["type"].unique()):
            sub = normal_df[normal_df["type"] == ttype]
            if sub.empty:
                continue
            ax.scatter(
                5 * sub["x"], 5 * sub["y"],
                c=base_color_map.get(ttype, "#397387ff"),
                s=size_map.get(ttype, 10),
                marker=marker_map.get(ttype, "o"),
                zorder=zorder_map.get(ttype, 1),
                alpha=alpha_map.get(ttype, 1.0),
                edgecolors="none",
            )

        # ---- Scatter: selected IDs — colored by current type, inactive if off-trace ----
        sel_df = frame_df[frame_df["id"].isin(show_ids)]
        for ttype in sorted(sel_df["type"].unique()):
            sub = sel_df[sel_df["type"] == ttype]
            if sub.empty:
                continue
            is_inactive = ttype not in trace_types
            ax.scatter(
                5 * sub["x"], 5 * sub["y"],
                c=inactive_color if is_inactive else base_color_map.get(ttype, "#397387ff"),
                s=synthase_s if is_inactive else size_map.get(ttype, 10),
                marker="s" if is_inactive else marker_map.get(ttype, "o"),
                zorder=99 if is_inactive else zorder_map.get(ttype, 1),
                alpha=inactive_alpha if is_inactive else alpha_map.get(ttype, 1.0),
                edgecolors="none",
            )

        # ---- Traces (zorder between background and synthase) ----
        MAX_DIST = 25.0
        SCALE = 5.0
        for pid in show_ids:
            pdata = df_trace[
                (df_trace["id"] == pid)
                & (df_trace["time"] <= closest_time)
                & (df_trace["type"].isin(trace_types))
            ]
            if pdata.empty:
                continue
            pdata = pdata.sort_values("time")
            px = SCALE * pdata["x"].values
            py = SCALE * pdata["y"].values
            diffs = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)
            break_idx = np.where(diffs > MAX_DIST)[0]
            segs, start = [], 0
            for b in break_idx:
                segs.append(slice(start, b + 1))
                start = b + 1
            segs.append(slice(start, len(px)))

            ptype = pdata["type"].iloc[0]
            color  = trace_color_map.get(ptype, "#397387ff")
            lw     = lw_map.get(ptype, 2.5)
            label  = label_map.get(ptype, f"type {ptype}")
            talpha = trace_alpha_map.get(ptype, 1.0)
            tz     = trace_zorder_map.get(ptype, 1.5)

            for seg in segs:
                if seg.stop - seg.start < 2:
                    continue
                ax.plot(
                    px[seg], py[seg],
                    lw=lw, color=color, alpha=talpha,
                    label=label, zorder=tz,
                )

        if display_time:
            ax.text(
                0.8, 0.9, f"t = {closest_time:.0f} s",
                transform=ax.transAxes, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="white"),
            )
        if display_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), frameon=False)

        fig.canvas.draw_idle()
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile(f"{tag}.mp4", fps=fps, codec="libx264", audio=False)