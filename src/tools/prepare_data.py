import glob
import pickle

import cv2
import numpy as np
from skimage.feature import hog

from src.tools.general_settings import feat_settings


def load_images(path):
    images = []
    for image_path in glob.glob(path, recursive=True):
        image = cv2.imread(image_path)
        images.append(image)
    return images


def load_model():
    return pickle.load(open("svc_linear.p", "rb"))


def image_hog(img, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True):
    return hog(img,
               orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               visualise=vis,
               feature_vector=feature_vec,
               block_norm="L2-Hys")


def image_bin_spatial(image, size=feat_settings["spatial_size"]):
    return cv2.resize(image, size).ravel()


def color_hist(img, nbins=feat_settings["n_bins"], bins_range=(0, 256)):
    c1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    c2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    c3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    c_hist = np.vstack((c1_hist[0], c2_hist[0], c3_hist[0]))
    # Generating bin centers
    bin_edges = c1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    return bin_centers, c_hist


def combined_features(img):
    orient = feat_settings["orient"]
    pix_per_cell = feat_settings["pix_per_cell"]
    cell_per_block = feat_settings["cell_per_block"]
    spatial_size = feat_settings["spatial_size"]
    n_bins = feat_settings["n_bins"]

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_c1 = img_ycrcb[:, :, 0]
    img_c2 = img_ycrcb[:, :, 1]
    img_c3 = img_ycrcb[:, :, 2]

    features_hog_c1 = image_hog(img_c1, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                vis=False)

    features_hog_c2 = image_hog(img_c2, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                vis=False)

    features_hog_c3 = image_hog(img_c3, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                vis=False)

    features_spatial = image_bin_spatial(img_ycrcb, size=spatial_size)
    _, features_hist = color_hist(img_ycrcb, nbins=n_bins)

    return np.hstack((
        features_spatial,
        features_hist.ravel(),
        features_hog_c1,
        features_hog_c2,
        features_hog_c3
    ))
