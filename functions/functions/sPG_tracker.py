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
strand_thickness_width = 4.5   # nm — width of one strand; also radial step in apply_deform
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
    propagates to calc_inward_deformations and CoverageTracker without needing
    to restart or pass anything manually.

    Parameters
    ----------
    strand_width_nm : float
        Width of one strand in nm. Sets both y-bin size and the radial
        constriction step in apply_deform (via strand_thickness_width).
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

    # timesteps controls how often x-bin edges are recomputed to track
    # the shrinking simulation box — not the temporal resolution of the physics.
    # Higher timesteps → finer tracking of box compression, at higher compute cost.

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

    print("─" * 15, "calculating deformations for times",
          df["time"].min(), "–", df["time"].max(), "─" * 15)

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


# ── Boundary deformation ──────────────────────────────────────────────────────
def apply_deform(deform, radii, N):
    """
    Apply one constriction step to the ring boundary.

    Only x-bins where deform=True are shrunk, by exactly one strandwidth.
    This replaces the old calc_radii which incorrectly used raw particle counts
    instead of the threshold decision.

    Uses module global strand_thickness_width (converted to simulation units).

    Parameters
    ----------
    deform : np.ndarray of bool, shape (N,)
        Which x-bins triggered constriction this call.
    radii : np.ndarray, shape (N,)
        Current radius at each x-bin in simulation units.
        Modified in place — owned by CoverageTracker.
    N : int
        Number of angular bins.

    Returns
    -------
    coords : np.ndarray, shape (N, 2)
        Updated (x, y) boundary coordinates in simulation units.
    """
    strandwidth_su = strand_thickness_width / NM_PER_SIM_UNIT  # nm → sim units
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Shrink by exactly one strandwidth at triggered bins only
    radii[deform] = np.maximum(0, radii[deform] - strandwidth_su)

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))  # (N, 2)


# ── Coverage tracker ──────────────────────────────────────────────────────────
class CoverageTracker:
    """
    Tracks cumulative strand coverage and ring boundary across simulation steps.

    Physical interpretation
    -----------------------
    _cumulative[i, j] = total strand particles observed in
                        x-bin i, y-bin j across all calls so far.

    Each y-bin corresponds to one strand-width layer within the septal thickness.
    A constriction step at x-bin i is triggered when the fraction of occupied
    y-bins (slots visited at least once) exceeds `threshold`.

    Once triggered:
    - the full y-column at bin i is decremented by 1 (one constriction level consumed)
    - the radius at bin i is reduced by one strandwidth via apply_deform()

    The ring boundary (radii) is owned by the tracker so it persists correctly
    across calls without external state management.

    Uses module globals N_Y_BINS and strand_thickness_width —
    call set_septal_bins() before instantiating if you need non-default geometry.

    Usage
    -----
        set_septal_bins()          # optional, if changing defaults
        box = lmp.extract_box()
        circumference = box[1][0] - box[0][0]
        tracker = CoverageTracker(N=N_angular_bins, circumference=circumference)
        for step in simulation_steps:
            result = calc_updated_circ(lmp, tracker, threshold=0.5, ...)
    """

    def __init__(self, N, circumference):
        """
        Parameters
        ----------
        N : int
            Number of angular x-bins around the circumference.
        circumference : float
            Initial box circumference in simulation units → sets initial radius.
        """
        self._cumulative = None                              # lazy-initialized on first update
        self.N      = N
        self.radii  = np.full(N, circumference / (2 * np.pi))  # uniform initial radius

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
        then decrement those bins and shrink the boundary.

        Threshold logic
        ---------------
        coverage_mask[i] = number of y-bins with at least one observed particle
                           at x-bin i, i.e. (_cumulative[i, :] > 0).sum()

        This measures *breadth* of coverage — how many distinct strand-slot
        positions have been visited — rather than total particle count.
        A single overactive y-bin therefore cannot alone trigger constriction.

        deform[i] = True  if  coverage_mask[i] > threshold * N_Y_BINS
                    i.e. more than `threshold` fraction of strand slots
                    have been occupied at least once at circumference position i.

        Decay (advancing constriction level)
        -------------------------------------
        For each triggered x-bin, subtract 1 from its entire y-column:
        one full coverage layer is consumed, advancing the constriction counter.
        The boundary radius at that bin is reduced by exactly one strandwidth.

        Parameters
        ----------
        threshold : float
            Fraction of y-bins that must be occupied to trigger constriction
            (e.g. 0.5 means 50% of strand slots must have been visited).

        Returns
        -------
        deform : np.ndarray of bool, shape (N,)
            True at x-bins where a constriction step is triggered.
        coverage_mask : np.ndarray of int, shape (N,)
            Number of occupied y-slots per x-bin (before decay).
        coords : np.ndarray, shape (N, 2)
            Updated boundary coordinates after applying deform.
        """
        # Count y-bins with at least one particle at each x-bin
        # (N, N_Y_BINS) → (N,)  — breadth, not depth
        coverage_mask = (self._cumulative > 0).sum(axis=1)

        # Trigger where fraction of occupied y-slots exceeds threshold
        # N_Y_BINS read from module global at call time
        deform = coverage_mask > threshold * N_Y_BINS
        print("─" * 15, f"deforming {deform.sum()} / {self.N} x-bins", "─" * 15)

        # Subtract 1 from the entire y-column at triggered bins:
        # _cumulative[deform] shape: (n_triggered, N_Y_BINS)
        # -= 1 decrements every y-slot uniformly → one constriction level consumed
        self._cumulative[deform] -= 1
        self._cumulative = np.maximum(0, self._cumulative)  # no negative counts

        # Apply radial shrinkage — modifies self.radii in place at deform=True bins
        coords = apply_deform(deform, self.radii, self.N)

        return deform, coverage_mask, coords

    def reset(self, circumference):
        """
        Reset coverage and boundary to a new initial circumference.

        Parameters
        ----------
        circumference : float
            New circumference in simulation units to reinitialize radii.
        """
        self._cumulative = None
        self.radii = np.full(self.N, circumference / (2 * np.pi))


# ── Cumulative-max tracker ────────────────────────────────────────────────────

def apply_deform_cummax(peak_counts, radii_initial, N):
    """
    Set ring radii from cumulative peak y-bin counts relative to initial radius.

    Rather than a binary threshold + decay, each x-bin is pushed inward by
    exactly peak_counts[i] * strandwidth_su from its initial (run-start) radius.
    This is continuous and monotonically inward — no consumption, no reset.

    Parameters
    ----------
    peak_counts : np.ndarray, shape (N,)
        Max cumulative count over all y-bins at each x-bin:
        peak_counts = _cumulative.max(axis=1)
    radii_initial : np.ndarray, shape (N,)
        Run-start radii in simulation units. Never modified.
    N : int
        Number of angular bins.

    Returns
    -------
    radii : np.ndarray, shape (N,)
        Updated radii in simulation units.
    coords : np.ndarray, shape (N, 2)
        Updated (x, y) boundary coordinates.
    """
    strandwidth_su = strand_thickness_width / NM_PER_SIM_UNIT
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    radii = np.maximum(0, radii_initial - peak_counts * strandwidth_su)
    coords = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return radii, coords


class CoverageTrackerCumMax:
    """
    Tracks cumulative strand coverage and drives constriction from the
    peak y-bin count at each x-bin position.

    Physical interpretation
    -----------------------
    _cumulative[i, j] = total strand particles in x-bin i, y-bin j across
                        all calls so far (strictly monotonic, no decay).

    At each call, the radius at x-bin i is set to:
        r[i] = r_initial[i] - max_j(_cumulative[i, j]) * strandwidth_su

    where r_initial is fixed at run start. The ring is pushed inward as far
    as the highest-occupancy y-bin layer currently dictates — no threshold,
    no consumption.

    Uses module globals N_Y_BINS and strand_thickness_width.
    Call set_septal_bins() before instantiating if you need non-default geometry.
    """

    def __init__(self, N, circumference):
        """
        Parameters
        ----------
        N : int
            Number of angular x-bins.
        circumference : float
            Initial box circumference in simulation units → sets r_initial.
        """
        self._cumulative = None
        self.N            = N
        r0                = circumference / (2 * np.pi)
        self.radii_initial = np.full(N, r0)   # fixed for entire run
        self.radii         = np.full(N, r0)   # updated each call

    def update(self, inwrd_dT):
        """Accumulate a new coverage frame. Shape: (N, N_Y_BINS)."""
        if self._cumulative is None:
            self._cumulative = np.zeros_like(inwrd_dT)
        self._cumulative += inwrd_dT

    def apply_cummax(self):
        """
        Update radii from current cumulative peak counts.

        Returns
        -------
        peak_counts : np.ndarray, shape (N,)
            Max cumulative count over y-bins at each x-bin.
        coords : np.ndarray, shape (N, 2)
            Updated boundary coordinates.
        """
        peak_counts = self._cumulative.max(axis=1)   # (N,)
        self.radii, coords = apply_deform_cummax(peak_counts, self.radii_initial, self.N)
        print("─" * 15,
              f"max peak: {peak_counts.max():.1f}  |  "
              f"mean inward: {(self.radii_initial - self.radii).mean():.3f} su",
              "─" * 15)
        return peak_counts, coords

    def reset(self, circumference):
        """Reset coverage and boundary to a new initial circumference."""
        self._cumulative   = None
        r0                 = circumference / (2 * np.pi)
        self.radii_initial = np.full(self.N, r0)
        self.radii         = np.full(self.N, r0)


# ── Main update step ──────────────────────────────────────────────────────────
def calc_updated_circ_cummax(lmp, tracker, df, fulldf, N_angular_bins=200):
    """
    One constriction update step using the cumulative-max scheme.

    Instead of a binary threshold + decay, the radius at each x-bin is set to:
        r[i] = r_initial[i] - max_j(_cumulative[i, j]) * strandwidth_su

    where r_initial is fixed at run start and _cumulative grows monotonically.

    Parameters
    ----------
    lmp : lammps instance
    tracker : CoverageTrackerCumMax
    df : pd.DataFrame
        Strand particles only (types 5/9).
    fulldf : pd.DataFrame
        All particle types — for x-bin edge determination.
    N_angular_bins : int

    Returns
    -------
    circumference_updated : float
    xc, yc, r : float
        Fitted circle center and radius.
    inwrd : np.ndarray, shape (timesteps, N, N_Y_BINS)
    peak_counts : np.ndarray, shape (N,)
        Max cumulative y-bin count per x-bin (used for radial displacement).
    """
    inwrd    = calc_inward_deformations(df, fulldf, N=N_angular_bins)
    inwrd_dT = inwrd.sum(axis=0)

    tracker.update(inwrd_dT)
    peak_counts, coords = tracker.apply_cummax()

    xc, yc, r = fit_circle(coords)
    circumference_updated = 2 * np.pi * r

    return circumference_updated, xc, yc, r, inwrd, peak_counts


    """
    One constriction update step:
      1. Compute per-frame strand histograms over the current time window
      2. Accumulate coverage and check constriction threshold
      3. Apply radial shrinkage at triggered bins and fit updated circle

    If deform is all False (threshold not met anywhere), circumference_updated
    will equal the current fitted circumference with no change — deform_cmd
    should be gated on deform.any() in the main script.

    Parameters
    ----------
    lmp : lammps instance
        Active LAMMPS object for box dimension extraction.
    tracker : CoverageTracker
        Persistent coverage and boundary state — same instance across calls.
    threshold : float
        Fraction of y-bins (strand slots) that must be occupied to trigger
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
        Updated circumference from circle fit. Unchanged if deform.any()==False.
    xc, yc, r : float
        Fitted circle center and radius.
    inwrd : np.ndarray, shape (timesteps, N, N_Y_BINS)
        Raw per-frame histograms.
    coverage_mask : np.ndarray, shape (N,)
        Number of occupied y-slots per x-bin (before decay).
    deform : np.ndarray of bool, shape (N,)
        Which x-bins triggered a constriction step this call.
    """
    # 1. Histograms: shape (timesteps, N, N_Y_BINS)
    #    y_edges read from module global
    inwrd = calc_inward_deformations(df, fulldf, N=N_angular_bins)

    # 2. Sum over time frames → total counts this call: (N, N_Y_BINS)
    inwrd_dT = inwrd.sum(axis=0)

    # 3. Accumulate coverage, apply threshold, shrink boundary at triggered bins
    #    tracker.radii updated in place; coords reflects post-deform boundary
    tracker.update(inwrd_dT)
    deform, coverage_mask, coords = tracker.apply_threshold(threshold)

    # 4. Fit circle to updated boundary
    xc, yc, r = fit_circle(coords)
    circumference_updated = 2 * np.pi * r

    return circumference_updated, xc, yc, r, inwrd, coverage_mask, deform