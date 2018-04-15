import random

import cv2

from src.tools.prepare_data import load_images
import matplotlib.pyplot as plt

vehicles = load_images('../../data/vehicles/GTI_*/image*.png')
non_vehicles = load_images('../../data/non-vehicles/GTI/image*.png')

print(len(vehicles), "({:3.1f}%)".format(100 * len(vehicles) / (len(vehicles) + len(non_vehicles))), "vehicles")
print(len(non_vehicles), "({:3.1f}%)".format(100 * len(non_vehicles) / (len(vehicles) + len(non_vehicles))),
      "non vehicles")

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(vehicles[random.randint(0, len(vehicles))])
ax1.set_title('Vehicle', fontsize=10)
ax2.imshow(non_vehicles[random.randint(0, len(non_vehicles))])
ax2.set_title('Non Vehicle', fontsize=10)
plt.show()

# cv2.imwrite('../output_images/vehicle_sample.jpg', vehicles[0])
# cv2.imwrite('../output_images/non_vehicle_sample.jpg', non_vehicles[0])
