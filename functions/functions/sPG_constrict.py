from tqdm import tqdm
import numpy as np
import functions as fct 

def fit_circle(coords):
    """
    Fit a circle to a set of 2D points using least-squares optimization.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) containing 2D points [[x1, y1], [x2, y2], ...].

    Returns
    -------
    xc, yc, r : float
        Circle center coordinates (xc, yc) and radius r.
    """

    x = coords[:, 0]
    y = coords[:, 1]

    # Residual function: distance from circle
    def residuals(c):
        xc, yc, r = c
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    # Initial guess: center at mean, radius mean distance to center
    x_m, y_m = x.mean(), y.mean()
    r0 = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))
    initial_guess = [x_m, y_m, r0]

    result = least_squares(residuals, initial_guess)

    xc, yc, r = result.x
    return xc, yc, r



def deform_cmd(Lx_half):
    return f"fix 1 all deform 1 x final -{Lx_half} {Lx_half} remap x"

def calc_inward_deformations(df, fulldf, takemax=True, timesteps=100, 
                             N=200, ## nr of bins around the (current) circumference
                             yconsider=10, ## drop all traces outside +-yconsider and bin in that range
                             Nbins_y=20 ## nr of bins along the long cell axis
                             ):
    """
    Compute inward deformations using histogram data.
    
    df: filtered dataframe (type 5/9)
    fulldf: full dataframe used to determine x-limits per frame
    """
    t_steps = np.linspace(df["time"].min(), df["time"].max(), timesteps)
    delta_t = np.diff(t_steps)[0]
    inward_deformations = []

    for t in tqdm(t_steps, leave=False):
        # Get full-frame x-limits
        dft_full = fulldf[(fulldf["time"] >= t - delta_t) & (fulldf["time"] < t)]
        if len(dft_full) == 0:
            # skip empty frame
            inward_deformations.append(np.zeros(N))
            continue
        xbins = np.linspace(dft_full["x"].min(), dft_full["x"].max(), N+1)


        # Select only type 5/9 for deformation calculation
        df_t = df[(df["time"] >= t - delta_t) & (df["time"] < t)]
        if len(df_t) == 0:
            inward_deformations.append(np.zeros(N))
            continue

        # Compute 2D histogram
        H, _, _ = np.histogram2d(df_t["x"], df_t["y"],
                                 bins=[xbins, np.linspace(-yconsider, yconsider, Nbins_y)])

        # Collapse y-axis
        if takemax:
            profile = H.max(axis=1)  # optional, can cause edge spikes
        else:
            profile = H.sum(axis=1)
            if profile.sum() > 0:
                profile = profile / profile.sum()  # normalize to avoid sparse-frame bias

        inward_deformations.append(profile)

    return np.array(inward_deformations)

# persistent storage of cumulative coverage
coverage_cumulative = None # global variable across iterations
def calc_updated_circ(threshold, yconsider=10, N_angular_bins=200):
    global coverage_cumulative # allow persistent update

    df = fct.xyz_reader.read_xyz(filename="processive.xyz")

    # current frame’s instantaneous coverage (mean over y bins)
    inwrd = calc_inward_deformations(df, N_angular_bins, yconsider=yconsider, Nbins_y=20)
    # temporal sum
    inwrd_dT = np.array(list(inwrd.values())).sum(axis=0)

    # initialize cumulative coverage
    if coverage_cumulative is None:
        coverage_cumulative = np.zeros_like(inwrd_dT)

    # accumulate ful 2D coverage over time
    coverage_cumulative += inwrd_dT

    ## check long-axis mean to evaluate deform
    ## take the mean number of nonzero bins along the long axis
    coverage_mask = coverage_cumulative.mean(axis=1)  #(coverage_cumulative > 0).mean(axis=1)
    
    deform = coverage_mask > threshold

    # reduce bins that surpassed threshold
    coverage_cumulative[deform] = coverage_cumulative[deform] - 1
    coverage_cumulative[coverage_cumulative < 0] = 0

    # fit circle
    circumference = lmp.extract_box()[1][0] - lmp.extract_box()[0][0]
    radii_updated = calc_radii({0: deform}, N_angular_bins, [0], circumference)

    xc, yc, r = fit_circle(radii_updated[0])
    circumference_updated = 2 * np.pi * r

    return circumference_updated, xc, yc, r, inwrd, coverage_mask, deform
