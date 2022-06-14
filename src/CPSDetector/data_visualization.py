import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# check for the location of your csv file using this code.
# import os

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))
# csv_file = '/home/designa/Desktop/SpecDet/datasets/cpu_processes/cpu_states.csv' # change this to a the full path to your dataset using "pwd"
csv_file = "/Users/Gabriel/Projects/SPECDET/src/CPSDetector/cpu_states.csv"

dataset=pd.read_csv(csv_file)
print(dataset['class'].value_counts(dropna=False))

spectre_attack_df = dataset.loc[dataset['class']=="M"]
no_spectre_attack_df = dataset.loc[dataset['class']=="B"]

M = len(spectre_attack_df)
B = len(no_spectre_attack_df)


import pandas as pd
import matplotlib.pyplot as plt

# Bring some raw data.
frequencies = [M,B]
# In my original code I create a series and run on that,
# so for consistency I create a series from the list.
freq_series = pd.Series(frequencies)

x_labels = ['M','B']

# Plot the figure.
plt.figure(figsize=(12, 8))
ax = freq_series.plot(kind="bar")
ax.set_title("Amount Frequency")
ax.set_xlabel("Data point counts")
ax.set_ylabel("Size")
ax.set_xticklabels(x_labels)

rects = ax.patches

# Make some labels.

# Make some labels.
labels = ['', '']

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
    )
plt.show()