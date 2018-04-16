import cv2

from src.tools.prepare_data import load_images, image_bin_spatial
import matplotlib.pyplot as plt

vehicles = load_images('../../data/vehicles/GTI_*/image*.png')
non_vehicles = load_images('../../data/non-vehicles/GTI/image*.png')

vehicle = vehicles[1996]
vehicle_ycrcb_y = cv2.cvtColor(vehicle, cv2.COLOR_RGB2YCrCb)[:, :, 0]
vehicle_bin = image_bin_spatial(vehicle_ycrcb_y)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(vehicle)
ax1.set_title('Vehicle', fontsize=10)
ax2.imshow(vehicle_bin.reshape(16, 16), cmap="gray")
ax2.set_title('Vehicle Bin YCrCb -> Y', fontsize=10)
plt.show()

print(vehicle_bin.shape)
