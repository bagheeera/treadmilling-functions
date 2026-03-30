from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np
import functions as fct

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                     CONSTRICTION SCHEME OVERVIEW                            │
# ├──────────────────┬──────────────────────────────┬───────────────────────────┤
# │ Scheme           │ Tracker                      │ Update function           │
# ├──────────────────┼──────────────────────────────┼───────────────────────────┤
# │ Raw counts       │ CoverageTrackerRawCounts      │ calc_updated_circ_raw     │
# │ (original)       │                              │                           │
# │                  │ r[i] -= nr_processive_dT[i]  │ no threshold              │
# │                  │         * strandwidth_su      │ saves: inwrd,             │
# │                  │                              │        nr_processive       │
# ├──────────────────┼──────────────────────────────┼───────────────────────────┤
# │ Peak-weighted    │ CoverageTrackerPeakWeighted   │ calc_updated_circ_        │
# │ (current)        │                              │   peakweighted            │
# │                  │ w[i] = Gaussian(y=0)-weighted│ requires: profW           │
# │                  │   nr_processive fraction      │ saves: inwrd,             │
# │                  │ r[i] -= nr_processive_dT[i]  │        weighted_flux      │
# │                  │         * w[i]               │                           │
# │                  │         * strandwidth_su      │                           │
# ├──────────────────┼──────────────────────────────┼───────────────────────────┤
# │ Cumulative max   │ CoverageTrackerCumMax         │ calc_updated_circ_cummax  │
# │                  │                              │                           │
# │                  │ r[i] = r0 - max_j(cumul[i,j])│ no threshold              │
# │                  │         * strandwidth_su      │ saves: inwrd, peak_counts │
# ├──────────────────┼──────────────────────────────┼───────────────────────────┤
# │ Threshold+decay  │ CoverageTracker               │ calc_updated_circ         │
# │                  │                              │                           │
# │                  │ deform[i] = breadth > thresh  │ requires: threshold       │
# │                  │ r[i] -= strandwidth_su        │ saves: inwrd,             │
# │                  │ cumul[deform] -= 1 (decay)    │        coverage_mask,     │
# │                  │                              │        deform             │
# ├──────────────────┼──────────────────────────────┼───────────────────────────┤
# │ Symmetric        │ CoverageTrackerSymmetric      │ calc_updated_circ_        │
# │ (self-consistent)│                              │   symmetric               │
# │                  │ profile = cumul.mean(axis=0) │ no threshold, no profW    │
# │                  │ r = r0 - profile.max()       │ saves: inwrd,             │
# │                  │     * strandwidth_su          │        profile, peak      │
# │                  │ uniform radius — no circle fit│                           │
# └──────────────────┴──────────────────────────────┴───────────────────────────┘
#
# Usage (main script):
#
#   # Peak-weighted (current default):
#   profW_su = config_data.get("profW", 40/5)
#   tracker = pgt.CoverageTrackerPeakWeighted(N=N_angular_bins,
#                                             circumference=circumference,
#                                             profW=profW_su)
#   circumference_updated, xc, yc, r, inwrd, weighted_flux = \
#       pgt.calc_updated_circ_peakweighted(lmp, tracker, df, fulldf, N_angular_bins)
#   lmp.command(pgt.deform_cmd(circumference_updated / 2))
#
#   # Raw counts (original):
#   tracker = pgt.CoverageTrackerRawCounts(N=N_angular_bins, circumference=circumference)
#   circumference_updated, xc, yc, r, inwrd, nr_processive = \
#       pgt.calc_updated_circ_raw(lmp, tracker, df, fulldf, N_angular_bins)
#   lmp.command(pgt.deform_cmd(circumference_updated / 2))
#
#   # Cumulative max:
#   tracker = pgt.CoverageTrackerCumMax(N=N_angular_bins, circumference=circumference)
#   circumference_updated, xc, yc, r, inwrd, peak_counts = \
#       pgt.calc_updated_circ_cummax(lmp, tracker, df, fulldf, N_angular_bins)
#   lmp.command(pgt.deform_cmd(circumference_updated / 2))
#
#   # Symmetric (self-consistent with post-analysis histograms):
#   tracker = pgt.CoverageTrackerSymmetric(N=N_angular_bins, circumference=circumference)
#   circumference_updated, r, inwrd, profile, peak = \
#       pgt.calc_updated_circ_symmetric(lmp, tracker, df, fulldf, N_angular_bins)
#   lmp.command(pgt.deform_cmd(circumference_updated / 2))
#
#   # Threshold + decay:
#   tracker = pgt.CoverageTracker(N=N_angular_bins, circumference=circumference)
#   circumference_updated, xc, yc, r, inwrd, coverage_mask, deform = \
#       pgt.calc_updated_circ(lmp, tracker, threshold, df, fulldf, N_angular_bins)
#   if deform.any():
#       lmp.command(pgt.deform_cmd(circumference_updated / 2))


# ── Physical constants & y-binning ───────────────────────────────────────────
NM_PER_SIM_UNIT        = 5     # 1 simulation unit = 5 nm
strand_thickness_width = 4.5   # nm — width of one strand; also radial step in apply_deform
septal_thickness       = 40    # nm — total ring thickness (Wenzel PNAS 2020)

y_edges  = None
N_Y_BINS = None


def set_septal_bins(strand_width_nm=strand_thickness_width,
                    septal_thickness_nm=septal_thickness):
    """
    (Re)compute y_edges and N_Y_BINS from physical parameters and update module globals.
    Call before instantiating any tracker if using non-default geometry.
    """
    global y_edges, N_Y_BINS, strand_thickness_width, septal_thickness
    strand_thickness_width = strand_width_nm
    septal_thickness       = septal_thickness_nm
    y_edges = np.arange(
        -septal_thickness_nm - strand_width_nm,
         septal_thickness_nm + strand_width_nm,
         strand_width_nm
    ) / NM_PER_SIM_UNIT
    N_Y_BINS = len(y_edges) - 1
    print(f"Septal bins set: {N_Y_BINS} y-bins, "
          f"{y_edges[0]*NM_PER_SIM_UNIT:.1f} to {y_edges[-1]*NM_PER_SIM_UNIT:.1f} nm "
          f"in steps of {strand_width_nm} nm ({strand_width_nm/NM_PER_SIM_UNIT} sim units)")

set_septal_bins()


# ── Circle fitting ────────────────────────────────────────────────────────────
def fit_circle(coords):
    """Fit a circle to (N, 2) boundary points. Returns xc, yc, r."""
    x, y = coords[:, 0], coords[:, 1]

    def residuals(c):
        xc, yc, r = c
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    x_m, y_m = x.mean(), y.mean()
    r0 = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))
    result = least_squares(residuals, [x_m, y_m, r0])
    return result.x  # xc, yc, r


# ── LAMMPS helpers ────────────────────────────────────────────────────────────
def deform_cmd(Lx_half):
    """Return a LAMMPS fix deform command string for the given half box-length."""
    return f"fix 1 all deform 1 x final -{Lx_half} {Lx_half} remap x"


# ── Inward deformation histogram ─────────────────────────────────────────────
def calc_inward_deformations(df, fulldf, timesteps=100, N=200, N_fine=400,
                             y_edges_override=None, verbose=True):
    """
    Build a 2D particle-count histogram (x × y) for each time frame.

    Returns np.ndarray shape (timesteps, N_fine, N_Y_BINS).
      axis 0 = time frames
      axis 1 = x-bins (circumference), interpolated onto fine grid
      axis 2 = y-bins (long axis / strand slots)

    Binning strategy
    ----------------
    Initial binning uses N bins at the first-frame box width, giving a
    reference physical bin width. As the box shrinks, n_bins_t decreases
    to preserve that physical bin width. Each frame's histogram is then
    interpolated onto a fixed N_fine-bin grid for accumulation, ensuring
    all frames contribute comparable counts per unit arc length.

    Parameters
    ----------
    N : int
        Initial number of x-bins (sets reference physical bin width). Default 200.
    N_fine : int
        Fixed output grid size after interpolation. Default 400.
    y_edges_override : np.ndarray, optional
        Custom y-bin edges in simulation units. If None, uses pgt.y_edges.
        Pass custom edges for extended z-range in mesh analysis.
    """
    _y_edges = y_edges_override if y_edges_override is not None else y_edges
    t_steps  = np.linspace(df["time"].min(), df["time"].max(), timesteps)
    # print("t_steps",  t_steps, "min/max", df["time"].min(), df["time"].max())
    delta_t  = np.diff(t_steps)[0]
    n_ybins  = len(_y_edges) - 1
    inward_deformations = []

    # reference bin width — set from first valid frame
    bin_width_0   = None
    t_centers_out = np.linspace(0, 2 * np.pi, N_fine, endpoint=False)

    if verbose:
        print("─" * 15, "calculating deformations for times",
            df["time"].min(), "–", df["time"].max(), "─" * 15)

    for t in tqdm(t_steps, leave=False):
        dft_full = fulldf[(fulldf["time"] >= t - delta_t) & (fulldf["time"] < t)]
        if len(dft_full) == 0:
            inward_deformations.append(np.zeros((N_fine, n_ybins)))
            continue

        x_min_t     = dft_full["x"].min()
        x_max_t     = dft_full["x"].max()
        box_width_t = x_max_t - x_min_t

        # set reference bin width from first valid frame
        if bin_width_0 is None:
            bin_width_0 = box_width_t / N

        # dynamic bin count: preserve physical bin width as box shrinks
        n_bins_t    = max(2, int(box_width_t / bin_width_0))
        t_centers_t = np.linspace(0, 2 * np.pi, n_bins_t, endpoint=False)

        df_t = df[(df["time"] >= t - delta_t) & (df["time"] < t)]
        if len(df_t) == 0:
            inward_deformations.append(np.zeros((N_fine, n_ybins)))
            continue

        # map x → theta for this frame
        data_theta = ((df_t["x"].values - x_min_t) / box_width_t) * 2 * np.pi
        H_t, _, _  = np.histogram2d(data_theta, df_t["y"].values,
                                    bins=[np.linspace(0, 2*np.pi, n_bins_t+1), _y_edges])

        # interpolate onto fine fixed grid (linear on bin centers)
        if n_bins_t != N_fine:
            f   = interp1d(t_centers_t, H_t, axis=0, kind='linear',
                           bounds_error=False, fill_value=0.0)
            H_t = f(t_centers_out)

        inward_deformations.append(H_t)

    return np.array(inward_deformations)  # (timesteps, N_fine, n_ybins)


# ── Boundary deformation (threshold scheme only) ─────────────────────────────
def apply_deform(deform, radii, N):
    """
    Shrink radii by one strandwidth at deform=True bins (in place).
    Returns coords shape (N, 2).
    """
    strandwidth_su      = strand_thickness_width / NM_PER_SIM_UNIT
    angles              = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii[deform]       = np.maximum(0, radii[deform] - strandwidth_su)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


# ── Shared helper ─────────────────────────────────────────────────────────────
def _coords_from_radii(radii, N):
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


# ── Tracker: threshold + decay ────────────────────────────────────────────────
class CoverageTracker:
    """
    Constriction triggered when breadth of y-bin coverage exceeds threshold.
    Consumed bins are decremented; radius steps by exactly one strandwidth.
    """
    def __init__(self, N, circumference):
        self._cumulative = None
        self.N           = N
        self.radii       = np.full(N, circumference / (2 * np.pi))

    def update(self, inwrd_dT):
        if self._cumulative is None:
            self._cumulative = np.zeros_like(inwrd_dT)
        self._cumulative += inwrd_dT

    def apply_threshold(self, threshold):
        coverage_mask            = (self._cumulative > 0).sum(axis=1)
        deform                   = coverage_mask > threshold * N_Y_BINS
        print("─" * 15, f"deforming {deform.sum()} / {self.N} x-bins", "─" * 15)
        self._cumulative[deform] -= 1
        self._cumulative         = np.maximum(0, self._cumulative)
        coords                   = apply_deform(deform, self.radii, self.N)
        return deform, coverage_mask, coords

    def reset(self, circumference):
        self._cumulative = None
        self.radii       = np.full(self.N, circumference / (2 * np.pi))


# ── Tracker: cumulative max ───────────────────────────────────────────────────
class CoverageTrackerCumMax:
    """
    r[i] = r_initial[i] - max_j(_cumulative[i,j]) * strandwidth_su
    r_initial fixed at run start — no circle center drift.
    """
    def __init__(self, N, circumference):
        self._cumulative   = None
        self.N             = N
        r0                 = circumference / (2 * np.pi)
        self.radii_initial = np.full(N, r0)
        self.radii         = np.full(N, r0)

    def update(self, inwrd_dT):
        if self._cumulative is None:
            self._cumulative = np.zeros_like(inwrd_dT)
        self._cumulative += inwrd_dT

    def apply_cummax(self):
        strandwidth_su     = strand_thickness_width / NM_PER_SIM_UNIT
        peak_counts        = self._cumulative.max(axis=1)
        self.radii         = np.maximum(0, self.radii_initial - peak_counts * strandwidth_su)
        print("─" * 15,
              f"max peak: {peak_counts.max():.1f}  |  "
              f"mean inward: {(self.radii_initial - self.radii).mean():.3f} su",
              "─" * 15)
        return peak_counts, _coords_from_radii(self.radii, self.N)

    def reset(self, circumference):
        self._cumulative   = None
        r0                 = circumference / (2 * np.pi)
        self.radii_initial = np.full(self.N, r0)
        self.radii         = np.full(self.N, r0)


# ── Tracker: raw counts (original) ───────────────────────────────────────────
class CoverageTrackerRawCounts:
    """
    r[i] -= nr_processive_dT[i] * strandwidth_su  (incremental, y collapsed)
    Shrinkage proportional to total particle flux regardless of y position.
    """
    def __init__(self, N, circumference):
        self._cumulative   = np.zeros(N)
        self.N             = N
        r0                 = circumference / (2 * np.pi)
        self.radii_initial = np.full(N, r0)
        self.radii         = np.full(N, r0)

    def update_and_apply(self, inwrd_dT):
        strandwidth_su      = strand_thickness_width / NM_PER_SIM_UNIT
        nr_processive_dT    = inwrd_dT.sum(axis=1)
        self._cumulative   += nr_processive_dT
        self.radii          = np.maximum(0, self.radii - nr_processive_dT * strandwidth_su)
        print("─" * 15,
              f"total nr_processive: {self._cumulative.sum():.0f}  |  "
              f"mean inward: {(self.radii_initial - self.radii).mean():.3f} su",
              "─" * 15)
        return self._cumulative.copy(), _coords_from_radii(self.radii, self.N)

    def reset(self, circumference):
        self._cumulative   = np.zeros(self.N)
        r0                 = circumference / (2 * np.pi)
        self.radii_initial = np.full(self.N, r0)
        self.radii         = np.full(self.N, r0)


# ── Tracker: peak-weighted (current default) ──────────────────────────────────
class CoverageTrackerPeakWeighted:
    """
    Like CoverageTrackerRawCounts but weights nr_processive by proximity to y=0.

        w[i] = (inwrd_dT[i,:] @ gauss_w) / (nr_processive_dT[i] + eps)
        r[i] -= nr_processive_dT[i] * w[i] * strandwidth_su

    gauss_w[j] = exp(-y_j^2 / sigma_peak^2),  sigma_peak = profW / 2
    Precomputed at init from y_edges bin centers.
    """
    def __init__(self, N, circumference, profW):
        """
        profW : float
            Sigma of strand y-distribution in simulation units
            (config_data.get("profW", 40/5)).
        """
        self._cumulative   = np.zeros(N)
        self.N             = N
        self.sigma_peak    = profW / 2.0
        r0                 = circumference / (2 * np.pi)
        self.radii_initial = np.full(N, r0)
        self.radii         = np.full(N, r0)
        y_centers          = (y_edges[:-1] + y_edges[1:]) / 2
        self._gauss_w      = np.exp(-y_centers**2 / self.sigma_peak**2)

    def update_and_apply(self, inwrd_dT):
        strandwidth_su   = strand_thickness_width / NM_PER_SIM_UNIT
        eps              = 1e-8
        nr_processive_dT = inwrd_dT.sum(axis=1)
        peak_w           = (inwrd_dT @ self._gauss_w) / (nr_processive_dT + eps)
        weighted_dT      = nr_processive_dT * peak_w
        self._cumulative += weighted_dT
        self.radii        = np.maximum(0, self.radii - weighted_dT * strandwidth_su)
        print("─" * 15,
              f"mean peak_w: {peak_w.mean():.3f}  |  "
              f"mean inward: {(self.radii_initial - self.radii).mean():.3f} su",
              "─" * 15)
        return self._cumulative.copy(), _coords_from_radii(self.radii, self.N)

    def reset(self, circumference):
        self._cumulative   = np.zeros(self.N)
        r0                 = circumference / (2 * np.pi)
        self.radii_initial = np.full(self.N, r0)
        self.radii         = np.full(self.N, r0)


# ── Main update functions ─────────────────────────────────────────────────────

def _base_step(df, fulldf, N_angular_bins):
    """Shared first step: compute histograms and sum over timesteps."""
    inwrd    = calc_inward_deformations(df, fulldf, N=N_angular_bins)
    inwrd_dT = inwrd.sum(axis=0)   # (N, N_Y_BINS)
    return inwrd, inwrd_dT


def calc_updated_circ_peakweighted(lmp, tracker, df, fulldf, N_angular_bins=200):
    """
    Peak-weighted scheme (current default).
    Returns: circumference_updated, xc, yc, r, inwrd, weighted_flux
    """
    inwrd, inwrd_dT       = _base_step(df, fulldf, N_angular_bins)
    weighted_flux, coords = tracker.update_and_apply(inwrd_dT)
    xc, yc, r             = fit_circle(coords)
    return 2 * np.pi * r, xc, yc, r, inwrd, weighted_flux


def calc_updated_circ_raw(lmp, tracker, df, fulldf, N_angular_bins=200):
    """
    Raw counts scheme (original).
    Returns: circumference_updated, xc, yc, r, inwrd, nr_processive
    """
    inwrd, inwrd_dT         = _base_step(df, fulldf, N_angular_bins)
    nr_processive, coords   = tracker.update_and_apply(inwrd_dT)
    xc, yc, r               = fit_circle(coords)
    return 2 * np.pi * r, xc, yc, r, inwrd, nr_processive


def calc_updated_circ_cummax(lmp, tracker, df, fulldf, N_angular_bins=200):
    """
    Cumulative-max scheme.
    Returns: circumference_updated, xc, yc, r, inwrd, peak_counts
    """
    inwrd, inwrd_dT     = _base_step(df, fulldf, N_angular_bins)
    tracker.update(inwrd_dT)
    peak_counts, coords = tracker.apply_cummax()
    xc, yc, r           = fit_circle(coords)
    return 2 * np.pi * r, xc, yc, r, inwrd, peak_counts


def calc_updated_circ(lmp, tracker, threshold, df, fulldf, N_angular_bins=200):
    """
    Threshold + decay scheme.
    Returns: circumference_updated, xc, yc, r, inwrd, coverage_mask, deform
    """
    inwrd, inwrd_dT              = _base_step(df, fulldf, N_angular_bins)
    tracker.update(inwrd_dT)
    deform, coverage_mask, coords = tracker.apply_threshold(threshold)
    xc, yc, r                    = fit_circle(coords)
    return 2 * np.pi * r, xc, yc, r, inwrd, coverage_mask, deform


# ── Tracker: symmetric (circumferentially averaged, self-consistent) ──────────

class CoverageTrackerSymmetric:
    """
    Drives constriction from the peak of the circumferentially averaged
    y-profile — guaranteed to be self-consistent with post-analysis histograms.

    Physical interpretation
    -----------------------
    _cumulative[i, j] = total strand particles in x-bin i, y-bin j across
                        all calls so far.

    At each call:
        profile[j] = _cumulative.mean(axis=0)   # average over x-bins → (N_Y_BINS,)
        peak       = profile.max()               # peak y-bin occupancy
        r_new      = r_initial - peak * strandwidth_su

    This gives a single uniform radius for all x-bins, matching exactly what
    H_total.mean(axis=0).max() * strand_width gives in post-analysis
    (up to unit conversion: strandwidth_su = strandwidth_nm / NM_PER_SIM_UNIT).

    No circle fitting needed — circumference is set directly from r_new.
    Assumes circumferential symmetry, which is appropriate when you want to
    boil constriction down to a single radius value.

    Uses module globals y_edges, strand_thickness_width, NM_PER_SIM_UNIT.
    """

    def __init__(self, N, circumference):
        self._cumulative   = None
        self.N             = N
        self.r_initial     = circumference / (2 * np.pi)  # scalar — uniform radius
        self.r             = self.r_initial                # current radius (scalar)

    def update(self, inwrd_dT):
        """Accumulate new histogram frame. inwrd_dT shape: (N, N_Y_BINS)."""
        if self._cumulative is None:
            self._cumulative = np.zeros_like(inwrd_dT)
        self._cumulative += inwrd_dT

    def apply_symmetric(self):
        """
        Update radius from peak of circumferentially averaged y-profile.

        Returns
        -------
        r : float
            Updated radius in simulation units.
        profile : np.ndarray, shape (N_Y_BINS,)
            Mean cumulative counts per y-bin across all x-bins.
        peak : float
            Peak value of profile — drives radial displacement.
        """
        strandwidth_su = strand_thickness_width / NM_PER_SIM_UNIT

        # average over x-bins → (N_Y_BINS,)
        profile = self._cumulative.mean(axis=0)
        peak    = profile.max()

        # uniform radius for all x-bins
        self.r  = max(0.0, self.r_initial - peak * strandwidth_su)

        print("─" * 15,
              f"profile peak: {peak:.2f}  |  "
              f"r: {self.r:.3f} su  |  "
              f"inward: {self.r_initial - self.r:.3f} su",
              "─" * 15)
        return self.r, profile, peak

    def reset(self, circumference):
        self._cumulative = None
        self.r_initial   = circumference / (2 * np.pi)
        self.r           = self.r_initial


def calc_updated_circ_symmetric(lmp, tracker, df, fulldf, N_angular_bins=200):
    """
    Symmetric constriction scheme — self-consistent with post-analysis histograms.

    Radius is set from the peak of the circumferentially averaged y-profile:
        r = r_initial - cumulative.mean(axis=0).max() * strandwidth_su

    No circle fitting — circumference updated directly from scalar r.
    Assumes circumferential symmetry.

    Returns
    -------
    circumference_updated : float
    r : float
        Updated radius in simulation units.
    inwrd : np.ndarray, shape (timesteps, N, N_Y_BINS)
    profile : np.ndarray, shape (N_Y_BINS,)
        Circumferentially averaged cumulative y-profile.
    peak : float
        Peak of profile (drives displacement).
    """
    inwrd, inwrd_dT    = _base_step(df, fulldf, N_angular_bins)
    tracker.update(inwrd_dT)
    r, profile, peak   = tracker.apply_symmetric()
    circumference_updated = 2 * np.pi * r
    return circumference_updated, r, inwrd, profile, peak