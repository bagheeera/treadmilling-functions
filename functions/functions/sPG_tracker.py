from scipy.optimize import least_squares
from tqdm import tqdm
import numpy as np
import functions as fct

# ── Constants ────────────────────────────────────────────────────────────────
Y_CONSIDER = 10   # drop all traces outside ±Y_CONSIDER
N_BINS_Y   = 20   # number of bins along the long cell axis


# ── Circle fitting ────────────────────────────────────────────────────────────
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

    def residuals(c):
        xc, yc, r = c
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    x_m, y_m = x.mean(), y.mean()
    r0 = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))

    result = least_squares(residuals, [x_m, y_m, r0])
    xc, yc, r = result.x
    return xc, yc, r


# ── LAMMPS helpers ────────────────────────────────────────────────────────────
def deform_cmd(Lx_half):
    """Return a LAMMPS deform command string for the given half-length."""
    return f"fix 1 all deform 1 x final -{Lx_half} {Lx_half} remap x"


# ── Inward deformation ────────────────────────────────────────────────────────
def calc_inward_deformations(df, fulldf, timesteps=100,
                             N=200,             # number of bins around the circumference
                             yconsider=Y_CONSIDER,
                             Nbins_y=N_BINS_Y):
    """
    Compute inward deformations using histogram data.
    The full 2D histogram (N, Nbins_y) is preserved per frame so that
    the y-dimension can be collapsed later (e.g. mean over y for coverage mask).

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe (type 5/9 particles).
    fulldf : pd.DataFrame
        Full dataframe used to determine x-limits per frame.
    timesteps : int
        Number of evenly-spaced time steps to evaluate.
    N : int
        Number of x bins (around the circumference).
    yconsider : float
        Half-width of y range to consider.
    Nbins_y : int
        Number of bins along the long (y) axis.

    Returns
    -------
    np.ndarray of shape (timesteps, N, Nbins_y)
    """
    t_steps = np.linspace(df["time"].min(), df["time"].max(), timesteps)
    delta_t = np.diff(t_steps)[0]
    y_edges = np.linspace(-yconsider, yconsider, Nbins_y + 1)
    inward_deformations = []

    for t in tqdm(t_steps, leave=False):
        dft_full = fulldf[(fulldf["time"] >= t - delta_t) & (fulldf["time"] < t)]
        if len(dft_full) == 0:
            inward_deformations.append(np.zeros((N, Nbins_y)))
            continue

        xbins = np.linspace(dft_full["x"].min(), dft_full["x"].max(), N + 1)

        df_t = df[(df["time"] >= t - delta_t) & (df["time"] < t)]
        if len(df_t) == 0:
            inward_deformations.append(np.zeros((N, Nbins_y)))
            continue

        H, _, _ = np.histogram2d(df_t["x"], df_t["y"], bins=[xbins, y_edges])
        inward_deformations.append(H)   # shape (N, Nbins_y) — y preserved

    return np.array(inward_deformations)  # shape (timesteps, N, Nbins_y)


# ── Radii computation ─────────────────────────────────────────────────────────
def calc_radii(inward_deformations, N, timesteps,
               circumference,
               strandwidth=4.5  # nm
               ):
    """
    Compute the evolving (x, y) coordinates of the inward-deformed boundary
    over time, given inward deformation magnitudes at each frame.

    Parameters
    ----------
    inward_deformations : np.ndarray
        Array of shape (timesteps, N) containing inward deformation magnitudes
        per angle bin.
    N : int
        Number of angular bins (points around the circumference).
    timesteps : int
        Number of time steps corresponding to `inward_deformations`.
    circumference : float
        Current circumference used to derive the initial radius.
    strandwidth : float
        Deposition factor in nm (default 4.5).

    Returns
    -------
    np.ndarray of shape (timesteps, N, 2)
        (x, y) coordinates of the boundary for each timestep.
    """
    deposition_factor = strandwidth
    # D0 = 1200  # nm — retained for future reference, not currently used

    radius = circumference / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii  = np.full(N, radius)

    stored_coordinates = []
    for frame in range(timesteps):           # FIX: iterate over range, not the int
        delta_r = inward_deformations[frame] * deposition_factor
        radii   = np.maximum(0, radii - delta_r)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        stored_coordinates.append(np.column_stack((x, y)))

    return np.array(stored_coordinates)


# ── Coverage tracker ──────────────────────────────────────────────────────────
class CoverageTracker:
    """
    Tracks cumulative angular coverage across repeated calls,
    replacing the previous module-level global variable.

    Usage (in your main script)
    ---------------------------
        tracker = CoverageTracker()
        for step in simulation_steps:
            result = calc_updated_circ(lmp, tracker, threshold=0.5)
    """

    def __init__(self):
        self._cumulative = None

    def update(self, inwrd_dT):
        """Accumulate a new coverage frame and return the updated mask."""
        if self._cumulative is None:
            self._cumulative = np.zeros_like(inwrd_dT)
        self._cumulative += inwrd_dT

        # Reduce bins that surpassed threshold — caller sets threshold
        return self._cumulative

    def apply_threshold(self, threshold):
        """Return boolean mask of bins exceeding threshold, then decay those bins.
        _cumulative has shape (N, Nbins_y).
        Collapse y by mean → coverage_mask shape (N,)."""
        coverage_mask = self._cumulative.mean(axis=1)  # (N,)
        deform = coverage_mask > threshold
        self._cumulative[deform] -= 1
        self._cumulative = np.maximum(0, self._cumulative)
        return deform, coverage_mask

    def reset(self):
        self._cumulative = None


# ── Main update step ──────────────────────────────────────────────────────────
def calc_updated_circ(lmp, tracker, threshold, df, fulldf,
                      yconsider=Y_CONSIDER, N_angular_bins=200):
    """
    One update step: compute deformations, update coverage, fit new circle.

    Parameters
    ----------
    lmp : lammps instance
        Active LAMMPS object used to extract box dimensions.
    tracker : CoverageTracker
        Persistent coverage state across calls.
    threshold : float
        Coverage threshold above which deformation is triggered.
    df : pd.DataFrame
        Filtered dataframe (types 5/9 particles only).
    fulldf : pd.DataFrame
        Full dataframe (all particle types) for x-limit determination.
    yconsider : float
        Half-width of y range to consider.
    N_angular_bins : int
        Number of angular bins.

    Returns
    -------
    circumference_updated, xc, yc, r, inwrd, coverage_mask, deform
    """
    inwrd = calc_inward_deformations(
        df, fulldf,
        N=N_angular_bins,
        yconsider=yconsider,
        Nbins_y=N_BINS_Y
    )
    # inwrd shape: (timesteps, N, Nbins_y)
    # sum over timesteps → (N, Nbins_y)
    inwrd_dT = inwrd.sum(axis=0)

    tracker.update(inwrd_dT)
    deform, coverage_mask = tracker.apply_threshold(threshold)

    box = lmp.extract_box()
    circumference = box[1][0] - box[0][0]

    timesteps = inwrd.shape[0]
    # collapse y for radii: mean over Nbins_y → (timesteps, N)
    inwrd_collapsed = inwrd.mean(axis=2)
    radii_coords = calc_radii(inwrd_collapsed, N_angular_bins, timesteps, circumference)

    xc, yc, r = fit_circle(radii_coords[-1])
    circumference_updated = 2 * np.pi * r

    return circumference_updated, xc, yc, r, inwrd, coverage_mask, deform