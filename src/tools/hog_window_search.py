import cv2
import numpy as np
import time

from src.tools.general_settings import feat_settings
from src.tools.prepare_data import image_hog, image_bin_spatial, color_hist


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def subsampling_window_search(img, svc, X_scaler, y_start, y_stop, x_start, x_stop, scale=1.0,
                              orient=feat_settings["orient"],
                              pix_per_cell=feat_settings["pix_per_cell"],
                              cell_per_block=feat_settings["cell_per_block"],
                              spatial_size=feat_settings["spatial_size"],
                              n_bins=feat_settings["n_bins"],
                              debug=False):
    rectangles = []
    # img = img.astype(np.float32)  # / 255

    s_image = img[y_start:y_stop, x_start:x_stop, :]
    ycrcb_s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2YCrCb)

    if scale != 1:
        imshape = ycrcb_s_image.shape
        ycrcb_s_image = cv2.resize(ycrcb_s_image, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    img_c1 = ycrcb_s_image[:, :, 0]
    img_c2 = ycrcb_s_image[:, :, 1]
    img_c3 = ycrcb_s_image[:, :, 2]

    # Define blocks and steps as above
    nx_blocks = (img_c1.shape[1] // pix_per_cell) - cell_per_block + 1
    ny_blocks = (img_c1.shape[0] // pix_per_cell) - cell_per_block + 1
    # n_feat_per_block = orient * cell_per_block ** 2

    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nx_steps = (nx_blocks - nblocks_per_window) // cells_per_step + 1
    ny_steps = (ny_blocks - nblocks_per_window) // cells_per_step + 1

    hog_c1 = image_hog(img_c1, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                       feature_vec=False, vis=False)
    hog_c2 = image_hog(img_c2, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                       feature_vec=False, vis=False)
    hog_c3 = image_hog(img_c3, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                       feature_vec=False, vis=False)

    for xb in range(nx_steps):
        for yb in range(ny_steps):

            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step
            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            features_hog_c1 = hog_c1[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()
            features_hog_c2 = hog_c2[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()
            features_hog_c3 = hog_c3[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()

            subimg_ycrcb = cv2.resize(ycrcb_s_image[y_top:y_top + window, x_left:x_left + window], (64, 64))

            # cv2.imwrite('../output_images/test{}{}.jpg'.format(xb, yb), cv2.cvtColor(subimg_ycrcb, cv2.COLOR_YCrCb2BGR))

            features_spatial = image_bin_spatial(subimg_ycrcb, size=spatial_size)
            _, features_hist = color_hist(subimg_ycrcb, nbins=n_bins)

            test_features = X_scaler.transform(np.concatenate((
                features_spatial,
                features_hist.ravel(),
                features_hog_c1,
                features_hog_c2,
                features_hog_c3
            )).reshape(1, -1))

            test_prediction = svc.predict(test_features)

            if (test_prediction == 1) | debug:
                # cv2.imwrite('../output_images/test{}{}.jpg'.format(xb, yb), cv2.cvtColor(subimg_ycrcb, cv2.COLOR_YCrCb2BGR))
                x_box_left = np.int(x_left * scale)
                y_top_draw = np.int(y_top * scale)
                win_draw = np.int(window * scale)
                rectangles.append(
                    ((x_box_left + x_start, y_top_draw + y_start),
                     (x_box_left + win_draw + x_start, y_top_draw + win_draw + y_start)))

    return rectangles


def combined_window_search(image, svc, X_scaler):
    rects_1 = subsampling_window_search(image, svc, X_scaler,
                                        y_start=380, y_stop=480,
                                        x_start=0, x_stop=1280,
                                        scale=1,
                                        debug=False)

    rects_2 = subsampling_window_search(image, svc, X_scaler,
                                        y_start=380, y_stop=560,
                                        x_start=0, x_stop=1280,
                                        scale=1.5,
                                        debug=False)

    rects_3 = subsampling_window_search(image, svc, X_scaler,
                                        y_start=380, y_stop=620,
                                        x_start=0, x_stop=1280,
                                        scale=2,
                                        debug=False)

    rects_4 = subsampling_window_search(image, svc, X_scaler,
                                        y_start=380, y_stop=660,
                                        x_start=0, x_stop=1280,
                                        scale=2.5,
                                        debug=False)

    rects_5 = subsampling_window_search(image, svc, X_scaler,
                                        y_start=380, y_stop=700,
                                        x_start=0, x_stop=1280,
                                        scale=4,
                                        debug=False)

    rects = [rects_1, rects_2, rects_3, rects_4, rects_5]

    return rects_1, rects_2, rects_3, rects_4, rects_5, [item for sublist in rects for item in sublist]


def generate_heatmap(rects, dims=(720, 1280)):
    heatmap = np.zeros(dims)
    for rect in rects:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] += 1

    return heatmap


def draw_labeled_bboxes(img, labels):
    img = img.copy()
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img
