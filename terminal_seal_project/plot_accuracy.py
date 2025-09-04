import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Excel file path
excel_path = 'static/detected/detection_summary.xlsx'

# Read data
try:
    df = pd.read_excel(excel_path)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit(1)

# Count Available vs Not Available
counts = df['seal_available'].value_counts()
labels = counts.index.tolist()
values = counts.values

# Calculate accuracy (Available / Total)
available_count = counts.get('Available', 0)
total_count = counts.sum()
accuracy = available_count / total_count * 100 if total_count > 0 else 0

# 3D Bar Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# X positions
xpos = np.arange(len(labels))
ypos = np.zeros(len(labels))
zpos = np.zeros(len(labels))

dx = np.ones(len(labels))
dy = np.ones(len(labels))
dz = values

colors = ['green' if label == 'Available' else 'red' for label in labels]

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8)

# Set labels
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_xlabel('Seal Status')
ax.set_ylabel('Y')
ax.set_zlabel('Number of Images')
ax.set_title('Seal Detection Summary (3D)')

# Annotate bars
for i, v in enumerate(values):
    ax.text(xpos[i], ypos[i], dz[i]+0.5, str(v), color='black', ha='center', va='bottom', fontsize=12)

# Show accuracy label
ax.text2D(0.5, 0.95, f'Accuracy: {accuracy:.2f}%', transform=fig.transFigure, fontsize=14, color='blue', ha='center')

plt.tight_layout()
plt.show() 