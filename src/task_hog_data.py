import cv2

from src.tools.general_settings import feat_settings
from src.tools.prepare_data import load_images, image_hog
import matplotlib.pyplot as plt

vehicles = load_images('../../data/vehicles/GTI_*/image*.png')
non_vehicles = load_images('../../data/non-vehicles/GTI/image*.png')

orient = feat_settings["orient"]
pix_per_cell = feat_settings["pix_per_cell"]
cell_per_block = feat_settings["cell_per_block"]

vehicle = vehicles[666]
vehicle_c1 = cv2.cvtColor(vehicle, cv2.COLOR_RGB2YCrCb)[:, :, 0]
vehicle_c2 = cv2.cvtColor(vehicle, cv2.COLOR_RGB2YCrCb)[:, :, 1]
vehicle_c3 = cv2.cvtColor(vehicle, cv2.COLOR_RGB2YCrCb)[:, :, 2]
_, vehicle_hog_c1 = image_hog(vehicle_c1, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
_, vehicle__hog_c2 = image_hog(vehicle_c2, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
_, vehicle__hog_c3 = image_hog(vehicle_c3, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
ax1.imshow(vehicle)
ax1.set_title('Vehicle', fontsize=10)
ax2.imshow(vehicle_hog_c1, cmap="gray")
ax2.set_title('Vehicle Hog Y', fontsize=10)
ax3.imshow(vehicle__hog_c2, cmap="gray")
ax3.set_title('Vehicle Hog Cr', fontsize=10)
ax4.imshow(vehicle__hog_c3, cmap="gray")
ax4.set_title('Vehicle Hog Cb', fontsize=10)
plt.show()

# features = combined_hog_features(vehicle)
# print(features.shape)
