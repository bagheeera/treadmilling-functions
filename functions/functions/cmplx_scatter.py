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
                ax.axhline(qv, ls=ql_config.get('ls', '--'),
                          lw=1, color=ql_config['color'],
                          alpha=ql_config.get('alpha', 0.7))
        elif axis == 'x':
            xvals = D_ql.loc[(np.abs(D_ql["x"]) < abs(ql_config.get('quantile_range', quantile_range)[1]))]["x"] * scale_xy
            for q in quantiles:
                qv = np.quantile(xvals, q)
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
    if histogram_axis is not None and histogram_config is not None:
        for hist_cfg in histogram_config:
            hist_types = hist_cfg['types']
            hist_color = hist_cfg['color']
            hist_label = hist_cfg['label']
            display_hist_legend = hist_cfg.get('display_legend', True)
            
            df_hist = df[(df["type"].isin(hist_types)) & (df["time"] == t_frame)]
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