import numpy as np
import functions.sPG_tracker as pgt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def d_alpha(t, tau_c, alpha,):
    diam = (1-(t/tau_c)**alpha)**(1/alpha)
    return diam
t_model = np.linspace(0,50, 1000)
tau_c = 51
alpha = 1.3

def diam_plot(D, key, ax, modelonly=False, overlay="Overlay",
              axislabels=False, width="45%", height="45%",
              legendtitle=None, coltharp_label=None, ncol=1,
              coltharp_color="k",
              inset_ylim=None,
              color=None,):
    # 1. Main Plot Logic
    t, r = np.array(D[key]["t_r"]).T
    diam_md = r * 2 * 5
    ax.plot(t / 1000 / 60, diam_md / diam_md[0], lw=3, label=overlay,
            color=color,
    )
    if axislabels:
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"$D(t) / D(t_0)$")
    
    if not modelonly and 't_model' in globals():
        ax.plot(t_model, d_alpha(t_model, tau_c, alpha), color=coltharp_color, label=coltharp_label,
                ls="--")

    if "H_total" in D[key]:
        # 2. Handle the Inset (Check if it already exists to avoid duplicaates)
        # We store the inset reference in the main axes object to retrieve it later
        if not hasattr(ax, "my_inset"):
            ax.my_inset = inset_axes(ax, width=width, height=height, loc='lower left', borderpad=4)
            #ax.my_inset.set_title("H_total Mean", fontsize=9)
        
        ax_ins = ax.my_inset

        # 3. Data Processing for Inset
        z_range_tuple = (-3 * 70, 3 * 70)
        strand_width_su = pgt.strand_thickness_width / 5.0
        z_min, z_max = z_range_tuple
        
        z_edges = np.arange(z_min, z_max + strand_width_su, strand_width_su)
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        z_nm = z_centers * 5.0

        # 4. Plot into the existing inset
        ax_ins.plot(z_nm, D[key]["H_total"].mean(axis=0), label=overlay,
                    color=color,
        )
        
        # Refresh legends for both
        ax.legend(loc='upper right', fontsize=8, title=legendtitle, ncol=ncol)
        ax_ins.set_xlim(-300,300)
        if inset_ylim is not None:
            ax_ins.set_ylim(inset_ylim)
        if axislabels:
            ax_ins.set_xlabel("Long cell axis (nm)", fontsize=7)
            ax_ins.set_ylabel("Septum height (nm)", fontsize=7)
        # ax_ins.legend(fontsize=7)