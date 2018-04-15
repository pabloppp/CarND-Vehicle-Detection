import pickle

import cv2

from src.tools.prepare_data import combined_features

model = pickle.load(open("svc.p", "rb"))
svc = model["svc"]
X_scaler = model["scaler"]

image_car = cv2.imread("../test_images/test_car1.jpg")
image_car = cv2.cvtColor(image_car, cv2.COLOR_BGR2RGB)

image_non_car = cv2.imread("../test_images/test_non_car1.jpg")
image_non_car = cv2.cvtColor(image_non_car, cv2.COLOR_BGR2RGB)

features_car = combined_features(image_car)
features_car = X_scaler.transform(features_car.reshape(1, -1))

features_non_car = combined_features(image_non_car)
features_non_car = X_scaler.transform(features_non_car.reshape(1, -1))

prediction_car = svc.predict(features_car)
prediction_non_car = svc.predict(features_non_car)
print("Is a car:", prediction_car)
print("Is not a car:", prediction_non_car)
