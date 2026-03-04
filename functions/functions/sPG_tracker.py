from scipy.optimize import least_squares
from tqdm import tqdm
import numpy as np
import functions as fct

# ── Physical constants & y-binning ───────────────────────────────────────────
# The z-ring is modeled as having a finite septal thickness.
# Each strand occupies one bin of width = strand_thickness_width.
# y-bins represent physical strand-slot positions along the long cell axis.
#
# Bin edges run from -(septal_thickness + strand_thickness_width)
#                  to +(septal_thickness + strand_thickness_width)
# in steps of strand_thickness_width, converted to simulation units.
#
# y_edges and N_Y_BINS are module globals — always read at call time, never
# passed as default arguments, so set_septal_bins() propagates everywhere.

NM_PER_SIM_UNIT        = 5     # 1 simulation unit = 5 nm
strand_thickness_width = 4.5   # nm — width of one strand; also radial step in calc_radii
septal_thickness       = 40    # nm — total ring thickness (Wenzel PNAS 2020)

# Initialized below via set_septal_bins() — do not set manually
y_edges  = None
N_Y_BINS = None


def set_septal_bins(strand_width_nm=strand_thickness_width,
                    septal_thickness_nm=septal_thickness):
    """
    (Re)compute y_edges and N_Y_BINS from physical parameters and update module globals.

    Call this once at startup (done automatically on import), or any time you
    want to change the strand/septal geometry. Because all functions read
    y_edges and N_Y_BINS from module globals at call time, a single call here
    propagates to calc_inward_deformations, CoverageTracker, and calc_radii
    without needing to restart or pass anything manually.

    Parameters
    ----------
    strand_width_nm : float
        Width of one strand in nm. Sets both y-bin size and the radial
        constriction step in calc_radii (via strand_thickness_width).
    septal_thickness_nm : float
        Total ring thickness in nm (default: 40 nm, Wenzel PNAS 2020).
    """
    global y_edges, N_Y_BINS, strand_thickness_width, septal_thickness
    strand_thickness_width = strand_width_nm
    septal_thickness       = septal_thickness_nm
    y_edges = np.arange(
        -septal_thickness_nm - strand_width_nm,
         septal_thickness_nm + strand_width_nm,
         strand_width_nm
    ) / NM_PER_SIM_UNIT          # nm → simulation units
    N_Y_BINS = len(y_edges) - 1  # number of bins = edges - 1
    print(f"Septal bins set: {N_Y_BINS} y-bins, "
          f"{y_edges[0]*NM_PER_SIM_UNIT:.1f} to {y_edges[-1]*NM_PER_SIM_UNIT:.1f} nm "
          f"in steps of {strand_width_nm} nm ({strand_width_nm/NM_PER_SIM_UNIT} sim units)")

# Initialize globals with defaults on import
set_septal_bins()


# ── Circle fitting ────────────────────────────────────────────────────────────
def fit_circle(coords):
    """
    Fit a circle to a set of 2D points using least-squares optimization.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        2D boundary points [[x1, y1], [x2, y2], ...].

    Returns
    -------
    xc, yc, r : float
        Circle center (xc, yc) and radius r.
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
    """Return a LAMMPS fix deform command string for the given half box-length."""
    return f"fix 1 all deform 1 x final -{Lx_half} {Lx_half} remap x"


# ── Inward deformation histogram ─────────────────────────────────────────────
def calc_inward_deformations(df, fulldf, timesteps=100,
                             N=200):  # number of x-bins around the circumference
    """
    Build a 2D particle-count histogram (x × y) for each time frame.

    x-bins represent positions around the circumference (N bins).
    y-bins represent strand-slot positions along the long cell axis,
    sized by strand_thickness_width so each bin = one strand layer.
    The y-dimension is preserved here and collapsed only later.

    x-bin edges are derived from fulldf (all particle types) each frame,
    so they track the simulation box extent rather than just typed particles.

    Uses module globals y_edges and N_Y_BINS — call set_septal_bins() first
    if you need non-default geometry.

    Parameters
    ----------
    df : pd.DataFrame
        Processive strand particles only (e.g. types 5/9) — histogram counts.
    fulldf : pd.DataFrame
        All particle types — used only to set x-bin edges per frame.
    timesteps : int
        Number of evenly-spaced time points to evaluate.
    N : int
        Number of x-bins (circumference direction).

    Returns
    -------
    np.ndarray, shape (timesteps, N, N_Y_BINS)
        axis 0 = time frames
        axis 1 = x-bins (circumference)
        axis 2 = y-bins (long axis / strand slots)
    """
    t_steps = np.linspace(df["time"].min(), df["time"].max(), timesteps)
    delta_t = np.diff(t_steps)[0]
    n_ybins = len(y_edges) - 1  # read global at call time
    inward_deformations = []

    print("─" * 15, "calculating deformations for times", t_steps, "─" * 15)

    for t in tqdm(t_steps, leave=False):
        # x-edges from ALL particles → consistent with simulation box size
        dft_full = fulldf[(fulldf["time"] >= t - delta_t) & (fulldf["time"] < t)]
        if len(dft_full) == 0:
            inward_deformations.append(np.zeros((N, n_ybins)))
            continue

        # N bins → N+1 edges, spanning full box x-extent this frame
        xbins = np.linspace(dft_full["x"].min(), dft_full["x"].max(), N + 1)

        # Bin processive particles only
        df_t = df[(df["time"] >= t - delta_t) & (df["time"] < t)]
        if len(df_t) == 0:
            inward_deformations.append(np.zeros((N, n_ybins)))
            continue

        # H[i, j] = processive particle count in x-bin i, y-bin j
        #   axis 0 = x (circumference), length N
        #   axis 1 = y (long axis / strand slots), length n_ybins
        # y is NOT collapsed here — preserved so threshold logic can sum over it later
        H, _, _ = np.histogram2d(df_t["x"], df_t["y"], bins=[xbins, y_edges])
        inward_deformations.append(H)  # (N, n_ybins)

    return np.array(inward_deformations)  # (timesteps, N, n_ybins)


# ── Radii computation ─────────────────────────────────────────────────────────
def calc_radii(inward_deformations, N, timesteps, circumference):
    """
    Compute the evolving boundary coordinates as the ring constricts.

    Each frame, the local radius is reduced by:
        delta_r[i] = strand_count_in_bin[i] * strandwidth_su
    Radii accumulate inward over frames (never reset between frames).

    Uses module global strand_thickness_width (converted to simulation units
    via NM_PER_SIM_UNIT) — call set_septal_bins() to change it.

    Parameters
    ----------
    inward_deformations : np.ndarray, shape (timesteps, N)
        Per-frame, per-bin strand counts (y already collapsed before passing in).
    N : int
        Number of angular bins.
    timesteps : int
        Number of time frames.
    circumference : float
        Current box circumference in simulation units → sets initial radius.

    Returns
    -------
    np.ndarray, shape (timesteps, N, 2)
        (x, y) Cartesian boundary coordinates for each frame, in simulation units.
    """
    # D0 = 1200  # nm — initial diameter, retained for reference, not currently used

    # strand_thickness_width is in nm; convert to simulation units for consistency
    strandwidth_su = strand_thickness_width / NM_PER_SIM_UNIT  # nm → sim units

    radius = circumference / (2 * np.pi)  # initial radius, uniform around ring
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii  = np.full(N, radius)           # shape (N,), updated cumulatively each frame

    stored_coordinates = []
    for frame in range(timesteps):
        # Each strand reduces the local radius by one strandwidth (in sim units)
        delta_r = inward_deformations[frame] * strandwidth_su  # (N,)
        radii   = np.maximum(0, radii - delta_r)               # clamp: radius >= 0
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        stored_coordinates.append(np.column_stack((x, y)))     # (N, 2)

    return np.array(stored_coordinates)  # (timesteps, N, 2)


# ── Coverage tracker ──────────────────────────────────────────────────────────
class CoverageTracker:
    """
    Tracks cumulative strand coverage across simulation steps.

    Physical interpretation
    -----------------------
    _cumulative[i, j] = total strand particles observed in
                        x-bin i, y-bin j across all calls so far.

    Each y-bin corresponds to one strand-width layer within the septal thickness.
    A constriction step at x-bin i is triggered when enough strand slots are
    filled — i.e. when the summed coverage across all y-bins exceeds
    threshold * N_Y_BINS (where threshold is a fraction, e.g. 0.5 = 50%).

    Once triggered, the full y-column at bin i is decremented by 1:
    this "consumes" one complete coverage layer and advances the
    constriction counter to the next level.

    Uses module global N_Y_BINS — call set_septal_bins() before instantiating
    if you need non-default geometry.

    Usage
    -----
        set_septal_bins()          # optional, if changing defaults
        tracker = CoverageTracker()
        for step in simulation_steps:
            result = calc_updated_circ(lmp, tracker, threshold=0.5, ...)
    """

    def __init__(self):
        self._cumulative = None  # shape (N, N_Y_BINS), lazy-initialized on first update

    def update(self, inwrd_dT):
        """
        Accumulate a new coverage frame into the running total.

        Parameters
        ----------
        inwrd_dT : np.ndarray, shape (N, N_Y_BINS)
            Strand counts summed over time frames for this simulation step.
        """
        if self._cumulative is None:
            self._cumulative = np.zeros_like(inwrd_dT)
        self._cumulative += inwrd_dT  # (N, N_Y_BINS)

    def apply_threshold(self, threshold):
        """
        Identify x-bins where coverage justifies a constriction step,
        then decrement those bins to advance to the next level.

        Threshold logic
        ---------------
        coverage_mask[i] = _cumulative[i, :].sum()
                         = total strand-slot occupancy at x-bin i across all y-bins

        deform[i] = True  if  coverage_mask[i] > threshold * N_Y_BINS
                    i.e. more than `threshold` fraction of strand slots
                    have been filled at circumference position i.

        Decay (advancing constriction level)
        -------------------------------------
        For each triggered x-bin, subtract 1 from its entire y-column.
        This treats the column as one complete constriction unit:
        one full layer of coverage is consumed, moving the ring
        one step inward at that position.

        Parameters
        ----------
        threshold : float
            Fraction of y-bins that must be filled to trigger constriction
            (e.g. 0.5 means 50% of strand slots must be occupied).

        Returns
        -------
        deform : np.ndarray of bool, shape (N,)
            True at x-bins where a constriction step is triggered.
        coverage_mask : np.ndarray of float, shape (N,)
            Summed y-coverage per x-bin (before decay).
        """
        # Sum over all y-bins (strand slots) at each x-bin
        # (N, N_Y_BINS) → (N,)
        coverage_mask = self._cumulative.sum(axis=1)

        # Trigger where total occupancy exceeds threshold fraction of all slots
        # N_Y_BINS read from module global at call time
        deform = coverage_mask > threshold * N_Y_BINS
        print("─" * 15, f"deforming {deform.sum()} / {len(deform)} x-bins", "─" * 15)
        print("coverage per x-bin (summed over y-slots):", coverage_mask)

        # Subtract 1 from the entire y-column at triggered bins:
        # _cumulative[deform] shape: (n_triggered, N_Y_BINS)
        # -= 1 decrements every y-slot uniformly → one constriction level consumed
        self._cumulative[deform] -= 1
        self._cumulative = np.maximum(0, self._cumulative)  # no negative counts

        return deform, coverage_mask

    def reset(self):
        """Reset all cumulative coverage to zero (e.g. between independent runs)."""
        self._cumulative = None


# ── Main update step ──────────────────────────────────────────────────────────
def calc_updated_circ(lmp, tracker, threshold, df, fulldf, N_angular_bins=200):
    """
    One constriction update step:
      1. Compute per-frame strand histograms over the current time window
      2. Accumulate coverage and check constriction threshold
      3. Compute evolving boundary and fit updated circle

    Parameters
    ----------
    lmp : lammps instance
        Active LAMMPS object for box dimension extraction.
    tracker : CoverageTracker
        Persistent coverage state — must be the same instance across calls.
    threshold : float
        Fraction of y-bins (strand slots) that must be filled to trigger
        a constriction step at a given x-bin (e.g. 0.5 = 50% occupancy).
    df : pd.DataFrame
        Strand particles only (types 5/9) — for histogram counts.
    fulldf : pd.DataFrame
        All particle types — for x-bin edge determination per frame.
    N_angular_bins : int
        Number of x-bins around the circumference.

    Returns
    -------
    circumference_updated : float
    xc, yc, r : float
        Fitted circle center and radius.
    inwrd : np.ndarray, shape (timesteps, N, N_Y_BINS)
        Raw per-frame histograms.
    coverage_mask : np.ndarray, shape (N,)
        Summed y-coverage per x-bin (before decay).
    deform : np.ndarray of bool, shape (N,)
        Which x-bins triggered a constriction step this call.
    """
    # 1. Histograms: shape (timesteps, N, N_Y_BINS)
    #    y_edges read from module global
    inwrd = calc_inward_deformations(df, fulldf, N=N_angular_bins)

    # 2. Sum over time frames → total counts this call: (N, N_Y_BINS)
    inwrd_dT = inwrd.sum(axis=0)

    # 3. Accumulate and apply threshold → deform mask: (N,)
    tracker.update(inwrd_dT)
    deform, coverage_mask = tracker.apply_threshold(threshold)

    # 4. Get current circumference from LAMMPS box
    box = lmp.extract_box()
    circumference = box[1][0] - box[0][0]

    # 5. Collapse y-bins: sum over strand slots → total strand count per x-bin per frame
    #    (timesteps, N, N_Y_BINS) → (timesteps, N)
    inwrd_collapsed = inwrd.sum(axis=2)
    timesteps = inwrd.shape[0]
    radii_coords = calc_radii(inwrd_collapsed, N_angular_bins, timesteps, circumference)
    # radii_coords: (timesteps, N, 2)

    # 6. Fit circle to the final frame's constricted boundary
    xc, yc, r = fit_circle(radii_coords[-1])
    circumference_updated = 2 * np.pi * r

    return circumference_updated, xc, yc, r, inwrd, coverage_mask, deform