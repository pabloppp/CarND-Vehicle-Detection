import pickle

import cv2
import matplotlib.pyplot as plt
import time

from scipy.ndimage import label

from src.tools.hog_window_search import subsampling_window_search, draw_boxes, combined_window_search, generate_heatmap, \
    draw_labeled_bboxes

model = pickle.load(open("svc_2.p", "rb"))
svc = model["svc"]
X_scaler = model["scaler"]

image = cv2.imread("../test_images/test6.jpg")

t = time.time()
rects_1, rects_2, rects_3, rects_4, rects_5, rects = combined_window_search(image, svc, X_scaler)

heatmap = generate_heatmap(rects)
heatmap[heatmap <= 10] = 0

labels = label(heatmap)

labeled_img = draw_labeled_bboxes(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), labels)
elapsed = time.time() - t
print("Full pipeline took {:1.3f}s".format(elapsed))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Vehicle', fontsize=10)
ax2.imshow(labeled_img)
ax2.set_title('Labeled', fontsize=10)
plt.show()

print("Windows: ", len(rects_1), len(rects_2), len(rects_3), len(rects_1) + len(rects_2) + len(rects_3))
# cv2.imwrite('../output_images/window_search_debug_combined.jpg', cv2.cvtColor(image_rects, cv2.COLOR_RGB2BGR))
