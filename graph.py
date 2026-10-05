"""DriveSense: cleaned original analysis workflow. See README.md for data setup."""
from features import resolve_path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    file_path = 'combined/highway.csv'
    df = pd.read_csv(resolve_path(file_path))
    sampling_rate = 836.0
    last_column = df.columns[-1]
    data = df[last_column].values
    time = np.arange(len(data)) / sampling_rate
    plt.figure(figsize=(10, 5))
    plt.plot(time, data, label=last_column)
    plt.title(f'Raw Data from Last Column ({last_column})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()
