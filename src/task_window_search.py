import pickle

import cv2
import matplotlib.pyplot as plt
import time

from src.tools.hog_window_search import subsampling_window_search, draw_boxes, combined_window_search
from src.tools.prepare_data import load_model

model = load_model()
svc = model["svc"]
X_scaler = model["scaler"]

image = cv2.imread("../test_images/test6.jpg")

t = time.time()
rects_1, rects_2, rects_3, rects_4, rects_5, rects = combined_window_search(image, svc, X_scaler)
elapsed = time.time() - t
print("Search took {:1.3f}s".format(elapsed))

image_rects = draw_boxes(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), rects_1, color=(255, 0, 0))
image_rects = draw_boxes(image_rects, rects_2, color=(0, 255, 0))
image_rects = draw_boxes(image_rects, rects_3, color=(0, 0, 255))
image_rects = draw_boxes(image_rects, rects_4, color=(255, 255, 0))
image_rects = draw_boxes(image_rects, rects_5, color=(0, 255, 255))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Vehicle', fontsize=10)
ax2.imshow(image_rects)
ax2.set_title('Window search', fontsize=10)
plt.show()

print("Windows: ", len(rects_1), len(rects_2), len(rects_3), len(rects_1) + len(rects_2) + len(rects_3))
# cv2.imwrite('../output_images/window_search_debug_combined.jpg', cv2.cvtColor(image_rects, cv2.COLOR_RGB2BGR))
