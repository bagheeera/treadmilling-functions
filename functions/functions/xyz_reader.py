import os, json
import pandas as pd
from tqdm.notebook import tqdm


def stream_xyz(tdir="./", batch_size=None):
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

    filepath = os.path.join(tdir, "output.xyz")
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

            elif line.startswith("ITEM: ATOMS"):
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

def read_xyz(tdir="./", batch_size=100):
    """
    Read entire trajectory into a single DataFrame.
    """
    df_all = pd.concat(stream_xyz(tdir, batch_size=batch_size), ignore_index=True)
    return df_all