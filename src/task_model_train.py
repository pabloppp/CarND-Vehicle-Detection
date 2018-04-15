import time
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from src.tools.prepare_data import load_images, combined_features

vehicles = load_images('../../data/vehicles/**/*.png')
non_vehicles = load_images('../../data/non-vehicles/**/*.png')

print("vehicles", len(vehicles))
print("non vehicles", len(non_vehicles))

train_valid_split = 0.1

# Feature extraction
features_vehicle = []
features_non_vehicle = []
t = time.time()
for image in vehicles:
    feat = combined_features(image)
    features_vehicle.append(feat)
for image in non_vehicles:
    feat = combined_features(image)
    features_non_vehicle.append(feat)
elapsed = time.time() - t

features_vehicle = np.array(features_vehicle)
features_non_vehicle = np.array(features_non_vehicle)

train_size_vehicle = int(len(features_vehicle) * (1 - train_valid_split))
valid_size_vehicle = int(len(features_vehicle) * train_valid_split)
train_size_non_vehicle = int(len(features_non_vehicle) * (1 - train_valid_split))
valid_size_non_vehicle = int(len(features_non_vehicle) * train_valid_split)

print("Feature extraction time: {:1.3f}s per image (total: {:3.1f}s)".format(
    elapsed / (len(features_vehicle) + len(features_non_vehicle)), elapsed))

# Trainig/Validation data setyp
X_train = np.vstack((
    features_vehicle[:train_size_vehicle],
    features_non_vehicle[:train_size_non_vehicle]
)).astype(np.float64)
y_train = np.array([1] * train_size_vehicle + [0] * train_size_non_vehicle)

X_valid = np.vstack((
    features_vehicle[train_size_vehicle:train_size_vehicle + valid_size_vehicle],
    features_non_vehicle[train_size_non_vehicle:train_size_non_vehicle + valid_size_non_vehicle])
).astype(np.float64)
y_valid = np.array([1] * valid_size_vehicle + [0] * valid_size_non_vehicle)

# Scale data
X_scaler = StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_valid = X_scaler.transform(X_valid)

X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)

print(X_train.shape, X_valid.shape)

print("Training started")
svc = SVC(kernel='rbf', C=15.0, gamma='auto')
t = time.time()
svc.fit(X_train, y_train)
elapsed = time.time() - t
print("Training took {:2.3f}s".format(elapsed))

# Persist trained classifier
svc_model = {"svc": svc, "scaler": X_scaler}
pickle.dump(svc_model, open("svc_2.p", "wb"))

print('Train Accuracy {:1.4f}'.format(svc.score(X_train, y_train)))
print('Validation Accuracy {:1.4f}'.format(svc.score(X_valid, y_valid)))

t = time.time()
print('Sample prediction {}'.format(svc.predict(X_valid[:10])))
print('Sample expected   {}'.format(y_valid[:10]))
elapsed = time.time() - t
print("Single predictions take {:1.6f}s".format(elapsed / 10))
