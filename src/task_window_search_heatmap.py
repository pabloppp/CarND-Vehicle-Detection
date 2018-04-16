import pickle

import cv2
import matplotlib.pyplot as plt
import time

from src.tools.hog_window_search import subsampling_window_search, draw_boxes, combined_window_search, generate_heatmap
from src.tools.prepare_data import load_model

model = load_model()
svc = model["svc"]
X_scaler = model["scaler"]

image = cv2.imread("../test_images/test3.jpg")

t = time.time()
rects_1, rects_2, rects_3, rects_4, rects_5, rects = combined_window_search(image, svc, X_scaler)
elapsed = time.time() - t
print("Search took {:1.3f}s".format(elapsed))

heatmap = generate_heatmap(rects)
heatmap[heatmap <= 5] = 0

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Vehicle', fontsize=10)
ax2.imshow(heatmap * 255, cmap="hot")
ax2.set_title('Heatmaps', fontsize=10)
plt.show()

print("Windows: ", len(rects_1), len(rects_2), len(rects_3), len(rects_1) + len(rects_2) + len(rects_3))
# cv2.imwrite('../output_images/window_search_debug_combined.jpg', cv2.cvtColor(image_rects, cv2.COLOR_RGB2BGR))
