import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('../../data/WECC_sim_data/trip_branch_MESA CAL    -2408-MESA CAL    -2438-i_360/gen_freqs.csv')

    print(df.head())
    sns.lineplot(data=df, x="time", y="generator-3933-CG")

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Generator 3933 CG")
    plt.title("Generator Output Over Time")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
