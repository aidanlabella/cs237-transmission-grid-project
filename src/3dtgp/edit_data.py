import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('/Users/aidan/sandbox/cs237-transmission-grid-project/data/WECC_sim_data/trip_branch_MESA CAL    -2408-MESA CAL    -2438-i_360/branch_current_real.csv')


    print(df.head())
    for col in df.columns:
        n = len(df)
        t = np.linspace(0, 2*np.pi, n)
        df[col] = 30 + (180 - 30) * (np.sin(t) + 1) / 2
    print(df.head())

    df.to_csv('/Users/aidan/sandbox/cs237-transmission-grid-project/data/WECC_sim_data/trip_branch_MESA CAL    -2408-MESA CAL    -2438-i_360/branch_current_real_sin.csv')

if __name__ == "__main__":
    main()
