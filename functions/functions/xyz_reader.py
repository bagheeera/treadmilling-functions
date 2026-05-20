import os, json
import pandas as pd
from tqdm.notebook import tqdm


def stream_xyz(tdir="./", batch_size=None, filename="output.xyz"):
    """
    Stream a LAMMPS-style .xyz trajectory.

    Yields one DataFrame per timestep (default),
    or larger DataFrames containing `batch_size` timesteps
    if batch_size is given.
    """
    # --- read parameters ---
    paramfile = os.path.join(tdir, "parameters.json")
    if os.path.exists(paramfile):
        with open(paramfile, "r") as f:
            config_data = json.load(f)
        tscale = config_data.get("tscale", 1)
        tstep = config_data.get("tstep", 1)
    else:
        print("no parameter file found, defaulting to standard timescale parameters")
        tstep = tscale = 1

    filepath = os.path.join(tdir, filename)
    filesize = os.path.getsize(filepath)

    # accumulator for batch mode
    batch = []

    with open(filepath, "r") as f, tqdm(total=filesize, unit="B", unit_scale=True, desc="Streaming XYZ") as pbar:
        timestep = None
        reading_atoms = False
        column_names = None
        chunk = []

        for line in f:
            pbar.update(len(line))
            line = line.strip()

            if line.startswith("ITEM: TIMESTEP"):
                # flush previous chunk
                if chunk and column_names:
                    df = pd.DataFrame(chunk, columns=column_names)
                    df["time"] = df["time"] * tstep / tscale
                    for col in df.columns:
                        if col in {"id", "mol", "type"}:
                            df[col] = df[col].astype("int32")
                        elif col != "time":
                            df[col] = df[col].astype("float32")

                    if batch_size:
                        batch.append(df)
                        if len(batch) >= batch_size:
                            yield pd.concat(batch, ignore_index=True)
                            batch = []
                    else:
                        yield df

                # new timestep
                timestep = int(next(f).strip())
                pbar.update(len(str(timestep)) + 1)  # account for that line
                chunk = []
                reading_atoms = False

            elif line.startswith("ITEM: ATOMS") or line.startswith("ITEM: ENTRIES"): ## "entries" for bonds output files
                column_names = ["time"] + line.split()[2:]
                reading_atoms = True

            elif reading_atoms and column_names:
                values = line.split()
                if len(values) == len(column_names) - 1:
                    chunk.append([timestep] + list(map(float, values)))

        # flush last timestep
        if chunk and column_names:
            df = pd.DataFrame(chunk, columns=column_names)
            df["time"] = df["time"] * tstep / tscale
            for col in df.columns:
                if col in {"id", "mol", "type"}:
                    df[col] = df[col].astype("int32")
                elif col != "time":
                    df[col] = df[col].astype("float32")

            if batch_size:
                batch.append(df)
            else:
                yield df

    # flush leftover batch
    if batch_size and batch:
        yield pd.concat(batch, ignore_index=True)

def read_xyz(tdir="./", batch_size=100, filename="output.xyz"):
    """
    Read entire trajectory into a single DataFrame.
    """
    df_all = pd.concat(stream_xyz(tdir, 
                                  batch_size=batch_size,
                                  filename=filename), ignore_index=True)
    return df_all

## write out xyz
from multiprocessing import Pool

# ---------------------------------------------------------------------
# Header template (without hardcoded box bounds)
# ---------------------------------------------------------------------
HEADER_TEMPLATE = """ITEM: TIMESTEP
YYY
ITEM: NUMBER OF ATOMS
XXX
ITEM: BOX BOUNDS pp pp pp
{x_min} {x_max}
{y_min} {y_max}
-4.2500000000000000e+00 4.2500000000000000e+00
ITEM: ATOMS v_vStep id mol type x y
"""

# ---------------------------------------------------------------------
# Row formatter
# ---------------------------------------------------------------------
def format_row(row):
    return (
        ' ' * 5 +
        ' '.join(
            str(int(x)) if i < 4 else f'{x:.2f}'
            for i, x in enumerate(row)
        )
    )

# ---------------------------------------------------------------------
# Chunk processor (pure function)
# ---------------------------------------------------------------------
def process_chunk(args):
    t, df_chunk = args
    lines = []

    # Determine current columns
    cols = list(df_chunk.columns)

    # Box bounds are computed as before (requires x/y columns)
    x_min = df_chunk['x'].min()
    x_max = df_chunk['x'].max()
    y_min = df_chunk['y'].min()
    y_max = df_chunk['y'].max()

    # Dynamic header text
    header = f"""ITEM: TIMESTEP
{t}
ITEM: NUMBER OF ATOMS
{len(df_chunk)}
ITEM: BOX BOUNDS pp pp pp
{x_min:.16e} {x_max:.16e}
{y_min:.16e} {y_max:.16e}
-4.2500000000000000e+00 4.2500000000000000e+00
ITEM: ATOMS {' '.join(cols)}
"""
    lines.append(header)

    # Format each row
    for _, row in df_chunk.iterrows():
        formatted = []
        for c in cols:
            v = row[c]
            if c in ("id", "mol", "type"):
                # integer formatting for atom id and molecule id
                formatted.append(f"{int(v)}")
            elif isinstance(v, (int, float)):
                # float formatting for everything else numeric
                formatted.append(f"{v:.2f}")
            else:
                # fallback for strings or other data types
                formatted.append(str(v))
        lines.append(" ".join(formatted) + "\n")

    return "".join(lines)

# ---------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------
def write_xyz(
    df,
    output_file,
    skip=1,
    nprocesses=1,
    time_col="time"
):
    """
    Write a LAMMPS-style XYZ trajectory from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    output_file : str
        Output .xyz filename
    skip : int, optional
        Write every `skip`-th timestep
    nprocesses : int, optional
        Number of worker processes (1 = serial)
    time_col : str, optional
        Name of the time column
    """
    sel_columns = [c for c in df.columns if c != time_col]
    time_chunks = [
        (t, df[df[time_col] == t][sel_columns])
        for t in df[time_col].unique()[::skip]
    ]
    
    if nprocesses == 1:
        blocks = map(process_chunk, time_chunks)
    else:
        with Pool(processes=nprocesses) as pool:
            blocks = pool.map(process_chunk, time_chunks)
    
    with open(output_file, "w") as f:
        f.writelines(blocks)