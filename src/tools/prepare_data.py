import glob

import cv2
import numpy as np
from skimage.feature import hog


def load_images(path):
    images = []
    for image_path in glob.glob(path, recursive=True):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


def image_hog(img, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True):
    return hog(img,
               orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               visualise=vis,
               feature_vector=feature_vec,
               block_norm="L2-Hys")


def image_bin_spatial(image, size=(32, 32)):
    return cv2.resize(image, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    c_hist = np.histogram(img, bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = c_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = c_hist[0]
    # Return the individual histograms, bin_centers and feature vector
    return c_hist, bin_centers, hist_features


def combined_features(img):
    orient = 9  # 12
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (16, 16)
    n_bins = 16

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_c1 = img_ycrcb[:, :, 0]
    img_c2 = img_hsv[:, :, 1]
    img_c3 = img_hsv[:, :, 2]

    features_hog_c1 = image_hog(img_c1, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                vis=False)
    features_hog_c2 = image_hog(img_c2, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                vis=False)
    features_hog_c3 = image_hog(img_c3, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                vis=False)

    features_spatial_1 = image_bin_spatial(img_c1, size=spatial_size)
    features_spatial_2 = image_bin_spatial(img_c2, size=spatial_size)
    features_spatial_3 = image_bin_spatial(img_c3, size=spatial_size)

    _, _, features_hist_1 = color_hist(img_c1, nbins=n_bins)
    _, _, features_hist_2 = color_hist(img_c2, nbins=n_bins)
    _, _, features_hist_3 = color_hist(img_c3, nbins=n_bins)

    return np.concatenate((
        features_hog_c1,
        features_hog_c2,
        features_hog_c3,
        features_spatial_1,
        features_spatial_2,
        features_spatial_3,
        features_hist_1,
        features_hist_2,
        features_hist_3
    ))
