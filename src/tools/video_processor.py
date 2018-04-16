import cv2
import numpy as np
from scipy.ndimage import label

from src.tools.hog_window_search import generate_heatmap, combined_window_search, draw_labeled_bboxes


class VideoProcessor:
    def __init__(self, svc, X_scaler):
        self.heatmap_history = []
        self.svc = svc
        self.scaler = X_scaler

    def heatmap(self, image):
        image = np.copy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rects_1, rects_2, rects_3, rects_4, rects_5, rects = combined_window_search(image, self.svc, self.scaler)

        heatmap = generate_heatmap(rects)
        heatmap[heatmap <= 4] = 0

        # print(heatmap.shape)
        self.heatmap_history.append(heatmap)
        m = 255 / np.max(heatmap)
        return cv2.cvtColor(np.uint8(heatmap.reshape(720, 1280, 1) * m), cv2.COLOR_GRAY2RGB)

    def labeled(self, image):
        image = np.copy(image)
        labels = label(self.heatmap_history_combined())
        labeled_img = draw_labeled_bboxes(image, labels)
        return labeled_img

    def heatmap_history_combined(self, window=10, required=6):
        # trim everyting except the last N values to avoid a memory overload
        self.heatmap_history = self.heatmap_history[-window:]
        combined = np.zeros(self.heatmap_history[0].shape)
        for heatmap in self.heatmap_history:
            combined[heatmap == 0] -= 1
            combined[combined < 0] = 0
            combined[heatmap > 0] += 1
        combined[combined <= required] = 0
        return combined
