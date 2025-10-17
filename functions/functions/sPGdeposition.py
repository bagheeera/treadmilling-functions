import numpy as np
from tqdm.notebook import tqdm

import numpy as np
from scipy.optimize import least_squares

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




def calc_inward_deformations(df, N, yconsider=50, Nbins_y=50):
    """
    Compute inward deformations using 2D histograms of (x, y) positions 
    for selected molecule types over time intervals.
    Inward deformation is the maximum over all bins along y for each x-bin.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns ['time', 'type', 'x', 'y'].
    N : int
        Number of bins along the x-axis.
    yconsider : float
        Half-range along the y-axis to consider (histogram spans [-yconsider, yconsider]).
    
    Returns
    -------
    np.ndarray
        Array of inward deformation values with shape (timesteps, N).
    """
    # Determine time range and interval size
    #T = df["time"].iloc[-1]
    t_steps = df["time"].unique() #np.linspace(0, T, timesteps)
    #delta_t = t_steps[1] - t_steps[0]
    delta_t = float(np.mean(np.diff(df["time"].unique())))

    # Precompute histogram bin edges
    x_edges = np.linspace(df["x"].min(), df["x"].max(), N + 1)
    y_edges = np.linspace(-yconsider, yconsider, Nbins_y)

    inward_deformations = {}

    for t in tqdm(t_steps, desc="Calculating inward deformations", leave=False):
        t = round(t, 5)
        # Select molecules of type 5 or 9 within the current time window
        df_t = df.loc[
            df["type"].isin([5, 9]) &
            (df["time"] >= t - delta_t) &
            (df["time"] < t)
        ]

        if df_t.empty:
            inward_deformations[t] = np.zeros(N)
            continue

        # Compute 2D histogram and take the maximum along y-axis for each x-bin
        H, _, _ = np.histogram2d(df_t["x"], df_t["y"], bins=[x_edges, y_edges])
        max_along_long_axis = H.max(axis=1)
        inward_deformations[t] = max_along_long_axis

    return inward_deformations


import numpy as np

def calc_radii(inward_deformations, N, timesteps, #df,
                circumference,
               strandwidth=4.5 # in nm
               ):
    """
    Compute the evolving (x, y) coordinates of the inward-deformed boundary 
    over time, given inward deformation magnitudes at each frame.
    
    Parameters
    ----------
    inward_deformations : np.ndarray
        Array of shape (timesteps, N) containing inward deformation magnitudes per angle bin.
    N : int
        Number of angular bins (points around the circumference).
    timesteps : int
        Number of time steps corresponding to `inward_deformations`.
    df : pandas.DataFrame
        DataFrame used to determine the spatial extent in x-direction.
        Must contain columns ['x'].
    
    Returns
    -------
    list of np.ndarray
        List of arrays, each of shape (N, 2), giving (x, y) coordinates for each timestep.
    """
    # --- Parameters ---
    deposition_factor = strandwidth
    D0 = 1200  # nm (not currently used, but retained for context)

    # --- Compute initial radius ---
    #circumference = (df["x"].max() - df["x"].min()) ## in units of sigma
    radius = circumference / (2 * np.pi)

    # --- Initialize ---
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.full(N, radius)
    stored_coordinates = []

    # --- Main loop ---
    for frame in timesteps:
        delta_r = inward_deformations[frame] * deposition_factor
        radii = np.maximum(0, radii - delta_r)  # ensure radius doesn’t go negative

        # Convert polar to Cartesian coordinates
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        stored_coordinates.append(np.column_stack((x, y)))

    return np.array(stored_coordinates)

import numpy as np
from skimage.measure import EllipseModel

def fit_ellipse_to_coordinates(circle_coords):
    """
    Fits an ellipse to the given (x, y) coordinates for each timestep.
    
    Parameters:
        circle_coords (np.ndarray): Shape (timesteps, N, 2), where N is the number of points.
        
    Returns:
        np.ndarray: Shape (timesteps, 5), storing (xc, yc, a, b, theta) for each timestep.
    """
    timesteps, N, _ = circle_coords.shape
    ellipse_params = np.zeros((timesteps, 5))  # Store ellipse (xc, yc, a, b, theta)

    for t in range(timesteps):
        points = circle_coords[t]  # Get (x, y) points at timestep t
        
        # Fit an ellipse
        ellipse = EllipseModel()
        if ellipse.estimate(points):
            xc, yc, a, b, theta = ellipse.params
        else:
            xc, yc, a, b, theta = np.nan, np.nan, np.nan, np.nan, np.nan  # If fitting fails
        
        ellipse_params[t] = [xc, yc, a, b, theta]

    return ellipse_params