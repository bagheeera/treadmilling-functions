## from http://127.0.0.1:7777/notebooks/0__treadmilling/6__balance_out_epsilon/process_synthases.ipynb

def correct_PBC_jumps(x, jumpcut=20, sidelength=200):
    x = x.copy()  # Avoid modifying the input array directly
    for i in range(1, len(x)):
        if x[i] > x[i - 1]:
            if x[i] - x[i - 1] > jumpcut:
                print(f"Jump detected at index {i}: {x[i]} -> adjusting by -{sidelength}")
                x[i:] -= sidelength
        elif x[i - 1] - x[i] > jumpcut:
            print(f"Jump detected at index {i}: {x[i]} -> adjusting by +{sidelength}")
            x[i:] += sidelength
    return x


def correct_PBC_jumps_dataframe(df, jumpcut=20, sidelength=200):
    df = df.copy()  # Work with a copy to avoid modifying the original DataFrame
    for col in ["x", "y"]:  # Apply to specific columns
        if col in df:
            df[col] = correct_PBC_jumps(df[col].values, jumpcut, sidelength)
        else:
            print(f"Column {col} not found in DataFrame")
    return df


from IPython.display import Video; 
## assumed mp4 written to rundir
def show_video(skey):
    tag = D[key]["rundir"].split("/")[-3] + "_.mp4"
    return Video(f"{D[key]["rundir"] + tag}", embed=True)