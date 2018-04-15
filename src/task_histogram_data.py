import cv2

from src.tools.prepare_data import load_images, image_bin_spatial, color_hist
import matplotlib.pyplot as plt

vehicles = load_images('../../data/vehicles/GTI_*/image*.png')
non_vehicles = load_images('../../data/non-vehicles/GTI/image*.png')

orient = 18
pix_per_cell = 6
cell_per_block = 3

vehicle = vehicles[123]
vehicle_ycrcb_y = cv2.cvtColor(vehicle, cv2.COLOR_RGB2YCrCb)[:, :, 0]
c_hist, bin_centers, hist_features = color_hist(vehicle_ycrcb_y)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(vehicle)
ax1.set_title('Vehicle', fontsize=10)

ax2.bar(bin_centers, hist_features)
plt.xlim(0, 256)
ax2.set_title('Histogram YCrCb -> Y')
plt.tight_layout()

plt.show()

print(hist_features.shape)
