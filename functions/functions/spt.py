import pandas as pd
import numpy as np
def read_xml(c):
    from lxml import etree
    import pandas as pd

    tree = etree.parse(c)

    rows = []
    for pid, particle in enumerate(tree.xpath(".//particle")):
        for det in particle.xpath(".//detection"):
            rows.append({
                "id": pid,
                "time": int(det.get("t")),
                "x": float(det.get("x")),
                "y": float(det.get("y")),
                "z": float(det.get("z")),
            })

    df = pd.DataFrame(rows)
    return df


def df_scatter(df, ax, s=0.5):
    df = df.copy()
    df.loc[:, "t_zeroed"] = df["t"] - df.groupby("id")["t"].transform("min")
    ax.scatter(*df[["x", "y"]].values.T,
                          c=df["t_zeroed"],
                          cmap="jet",
                          s=s)
    ax.set_aspect("equal")