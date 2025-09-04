import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Directory containing images and YOLO label files
img_dir = 'static/uploads'
label_dir = 'data/labels'  # Updated label directory path

# Get list of images
img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for idx, img_name in enumerate(img_files):
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # YOLO format: class x_center y_center width height (normalized)
            _, x_c, y_c, bw, bh = map(float, parts[:5])
            x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h
            x1 = x_c - bw / 2
            y1 = y_c - bh / 2
            x2 = x_c + bw / 2
            y2 = y_c + bh / 2

            # 3D rectangle: (x1, y1, idx), (x2, y1, idx), (x2, y2, idx), (x1, y2, idx)
            verts = [
                [(x1, y1, idx), (x2, y1, idx), (x2, y2, idx), (x1, y2, idx)]
            ]
            box = Poly3DCollection(verts, alpha=0.4)
            box.set_facecolor(np.random.rand(3,))
            ax.add_collection3d(box)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Image Index')
ax.set_title('3D Visualization of YOLO Bounding Boxes')
plt.tight_layout()
plt.show() 