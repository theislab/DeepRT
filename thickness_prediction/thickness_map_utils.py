import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import gc
from scipy import ndimage
import matplotlib.colors as colors


class ThicknessMapUtils():

    def __init__(self, label_path, image_path, prediction_path):
        self.label_path = None
        self.image_path = None
        self.prediction_path = None

    def percentual_deviance(self, label, prediction):
        return np.round(np.mean(np.abs(label - prediction)) / np.mean(label), 2)

    def load_images(self, record_name):
        label = np.load(os.path.join(self.label_path, record_name))
        prediction = np.load(os.path.join(self.prediction_path, record_name))
        image = cv2.imread(os.path.join(self.image_path, record_name.replace(".npy", ".jpeg")))

        label_mu = self.pixel_to_mu_meter(label)
        prediction_mu = self.pixel_to_mu_meter(prediction)

        # resize prediciton
        prediction_mu = self.resize_prediction(prediction_mu)

        # remove three channel to stop normalization
        label_lr = self.get_low_res_depth_grid(label_mu)[:, :, 0]
        prediction_lr = self.get_low_res_depth_grid(prediction_mu)[:, :, 0]
        return (label_mu, prediction_mu, label_lr, prediction_lr, image)

    def createCircularMask(self, h, w, center=None, radius=None):
        if center is None:  # use the middle of the image
            center = [int(w / 2), int(h / 2)]
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    def get_zone(self, record_th_z, zone):
        zone_value = record_th_z[record_th_z.Name == zone].AvgThickness.iloc[0]
        # mean of all zone thickness
        zone_avg = np.nanmean(np.array(record_th_z.AvgThickness, dtype = np.float32))

        if zone_value is None:
            zone_value = zone_avg

        return (float(zone_value))

    def extract_values(self, record_th_z):
        C0_value = self.get_zone(record_th_z, "C0")
        S2_value = self.get_zone(record_th_z, "S2")
        S1_value = self.get_zone(record_th_z, "S1")
        N1_value = self.get_zone(record_th_z, "N1")
        N2_value = self.get_zone(record_th_z, "N2")
        I1_value = self.get_zone(record_th_z, "I1")
        I2_value = self.get_zone(record_th_z, "I2")
        T1_value = self.get_zone(record_th_z, "T1")
        T2_value = self.get_zone(record_th_z, "T2")
        return (C0_value, S2_value, S1_value, N1_value, N2_value, I1_value, I2_value, T1_value, T2_value)

    def set_low_res_depth_grid(self, C0_value, S2_value, S1_value, N1_value, N2_value, I1_value, I2_value, T1_value,
                               T2_value,
                               C0, S2, S1, N1, N2, I1, I2, T1, T2, img):
        img[C0] = C0_value
        img[S1] = S1_value
        img[S2] = S2_value
        img[I1] = I1_value
        img[I2] = I2_value
        img[T1] = T1_value
        img[T2] = T2_value
        img[N1] = N1_value
        img[N2] = N2_value

        return img

    def rescale_oct_height(self, depth_map):
        scaling_factor = np.divide(496, 160, dtype = np.float32)
        rescaled_depth_map = depth_map * scaling_factor
        return rescaled_depth_map

    def center_img(self, img):
        global centered_img
        coords = np.argwhere(img > 0)
        x_min, y_min = coords.min(axis = 0)[0:2]
        x_max, y_max = coords.max(axis = 0)[0:2]
        cropped_img = img[x_min:x_max - 1, y_min:y_max - 1]
        if len(cropped_img.shape) == 2:
            square_cropped_img = cropped_img[0:min(cropped_img.shape), 0:min(cropped_img.shape)]

            centered_img = np.zeros((768, 768))

            nb = centered_img.shape[0]
            na = square_cropped_img.shape[0]
            lower = (nb) // 2 - (na // 2)
            upper = (nb // 2) + (na // 2)

            difference = np.abs(lower - upper) - square_cropped_img.shape[0]
            upper = upper - difference

            centered_img[lower:upper, lower:upper] = square_cropped_img

        if len(cropped_img.shape) == 3:
            square_cropped_img = cropped_img[0:min(cropped_img.shape[0:2]), 0:min(cropped_img.shape[0:2]), :]
            centered_img = np.zeros((768, 768, 3)).astype(np.uint8)

            nb = centered_img.shape[0]
            na = square_cropped_img.shape[0]
            lower = (nb) // 2 - (na // 2)
            upper = (nb // 2) + (na // 2)

            difference = np.abs(lower - upper) - square_cropped_img.shape[0]
            upper = upper - difference

            centered_img[lower:upper, lower:upper, :] = square_cropped_img

        return (centered_img)

    def get_low_res_grid(self, img):
        # scale of LOCALIZER
        outer_ring_radius = int(6 / 0.0118) / 2
        middle_ring_radius = int(3 / 0.0118) / 2
        inner_ring_radius = int(1 / 0.0118) / 2

        min_ = min(img.nonzero()[0]), min(img.nonzero()[1])
        max_ = max(img.nonzero()[0]), max(img.nonzero()[1])
        image_span = np.subtract(max_, min_)

        measure_area = np.zeros(image_span)

        nrows = img.shape[0]
        ncols = img.shape[1]
        cnt_row = image_span[1] / 2 + min_[1]
        cnt_col = image_span[0] / 2 + min_[0]

        max_diam = min(image_span)

        # init empty LOCALIZER sized grid
        img_mask = np.zeros((nrows, ncols), np.float32)

        # create base boolean masks
        inner_ring_mask = self.createCircularMask(nrows, ncols, center = (cnt_row, cnt_col), radius = inner_ring_radius)
        middle_ring_mask = self.createCircularMask(nrows, ncols, center = (cnt_row, cnt_col),
                                                   radius = middle_ring_radius)

        # fit low res grid to measurement area
        if outer_ring_radius * 2 > max_diam:
            outer_ring_radius = max_diam / 2

        outer_ring_mask = self.createCircularMask(nrows, ncols, center = (cnt_row, cnt_col), radius = outer_ring_radius)

        inner_disk = inner_ring_mask
        middle_disk = (middle_ring_mask.astype(int) - inner_ring_mask.astype(int)).astype(bool)
        outer_disk = (outer_ring_mask.astype(int) - middle_ring_mask.astype(int)).astype(bool)

        # create label specific diagonal masks
        upper_triangel_right_mask = np.arange(0, img.shape[1])[:, None] <= np.arange(img.shape[1])
        lower_triangel_left_mask = np.arange(0, img.shape[1])[:, None] > np.arange(img.shape[1])
        upper_triangel_left_mask = lower_triangel_left_mask[::-1]
        lower_triangel_right_mask = upper_triangel_right_mask[::-1]
        ''''
        #pad the shortened arrays
        im_utr = np.zeros((768,768))
        im_ltl = np.zeros((768,768))
        im_utl = np.zeros((768,768))
        im_ltr = np.zeros((768,768))
    
        #pad the diagonal masks
        im_utr[0:upper_triangel_right_mask.shape[0],:] = upper_triangel_right_mask
        im_ltl[768-upper_triangel_right_mask.shape[0]:,:] = lower_triangel_left_mask
        im_utl[0:upper_triangel_left_mask.shape[0], :] = upper_triangel_left_mask
        im_ltr[768-lower_triangel_right_mask.shape[0]:, :] = lower_triangel_right_mask
        #conversion
        im_utr = im_utr.astype(np.bool)
        im_ltl = im_ltl.astype(np.bool)
        im_utl = im_utl.astype(np.bool)
        im_ltr = im_ltr.astype(np.bool)
        '''
        # create 9 depth regions
        C0 = inner_ring_mask
        S2 = np.asarray(upper_triangel_left_mask & outer_disk & upper_triangel_right_mask)
        S1 = np.asarray(upper_triangel_left_mask & middle_disk & upper_triangel_right_mask)
        N1 = np.asarray(lower_triangel_right_mask & middle_disk & upper_triangel_right_mask)
        N2 = np.asarray(lower_triangel_right_mask & outer_disk & upper_triangel_right_mask)
        I1 = np.asarray(lower_triangel_right_mask & middle_disk & lower_triangel_left_mask)
        I2 = np.asarray(lower_triangel_right_mask & outer_disk & lower_triangel_left_mask)
        T1 = np.asarray(upper_triangel_left_mask & middle_disk & lower_triangel_left_mask)
        T2 = np.asarray(upper_triangel_left_mask & outer_disk & lower_triangel_left_mask)
        return C0, S2, S1, N1, N2, I1, I2, T1, T2

    def get_depth_grid_edges(self, area):
        struct = ndimage.generate_binary_structure(2, 2)
        erode = ndimage.binary_erosion(area, struct)
        edges = area ^ erode
        return np.stack((edges,) * 3, axis = -1)

    def get_low_res_grid_shape(self, img):
        C0, S2, S1, N1, N2, I1, I2, T1, T2 = self.get_low_res_grid(img)
        grid = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
        grid = grid + self.get_depth_grid_edges(C0)
        grid = grid + self.get_depth_grid_edges(S1)
        grid = grid + self.get_depth_grid_edges(S2)
        grid = grid + self.get_depth_grid_edges(I1)
        grid = grid + self.get_depth_grid_edges(I2)
        grid = grid + self.get_depth_grid_edges(T1)
        grid = grid + self.get_depth_grid_edges(T2)
        grid = grid + self.get_depth_grid_edges(N1)
        grid = grid + self.get_depth_grid_edges(N2)
        return grid

    def get_low_res_depth_grid(self, img):
        C0, S2, S1, N1, N2, I1, I2, T1, T2 = self.get_low_res_grid(img)
        grid = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
        grid[C0] = np.mean(img[C0])
        grid[S1] = np.mean(img[S1])
        grid[S2] = np.mean(img[S2])
        grid[I1] = np.mean(img[I1])
        grid[I2] = np.mean(img[I2])
        grid[T1] = np.mean(img[T1])
        grid[T2] = np.mean(img[T2])
        grid[N1] = np.mean(img[N1])
        grid[N2] = np.mean(img[N2])
        return grid

    def pixel_to_mu_meter(self, img):
        img_um = np.multiply(img, 0.0039 * 1000)
        return img_um

    def get_low_res_depth_grid_values(self, img):
        C0, S1, S2, N1, N2, I1, I2, T1, T2 = self.get_low_res_grid(img)
        # turn zero to nan
        img[img < 10] = 0
        img[img == 0] = np.nan
        # get mean values
        C0_value = np.nanmean(img[C0])
        S1_value = np.nanmean(img[S1])
        S2_value = np.nanmean(img[S2])
        I1_value = np.nanmean(img[I1])
        I2_value = np.nanmean(img[I2])
        T1_value = np.nanmean(img[T1])
        T2_value = np.nanmean(img[T2])
        N1_value = np.nanmean(img[N1])
        N2_value = np.nanmean(img[N2])

        # concert back nan values to zero
        img = np.nan_to_num(img)

        low_grid_values = [C0_value, S1_value, S2_value, N1_value, N2_value, I1_value, I2_value, T1_value, T2_value]
        return low_grid_values

    def get_low_res_depth_grid_maxvalues(self, img):
        C0, S1, S2, N1, N2, I1, I2, T1, T2 = self.get_low_res_grid(img)
        # turn zero to nan
        img[img < 10] = 0
        img[img == 0] = np.nan
        # get mean values
        C0_value = np.max(img[C0])
        S1_value = np.max(img[S1])
        S2_value = np.max(img[S2])
        I1_value = np.max(img[I1])
        I2_value = np.max(img[I2])
        T1_value = np.max(img[T1])
        T2_value = np.max(img[T2])
        N1_value = np.max(img[N1])
        N2_value = np.max(img[N2])

        # concert back nan values to zero
        img = np.nan_to_num(img)

        low_grid_values = [C0_value, S1_value, S2_value, N1_value, N2_value, I1_value, I2_value, T1_value, T2_value]
        return low_grid_values

    def get_text_coord(self, img):
        C0, S1, S2, N1, N2, I1, I2, T1, T2 = self.get_low_res_grid(img)

        S1_x_mc = np.median(np.where(S1 == True)[1])
        S1_y_mc = np.median(np.where(S1 == True)[0])

        S2_x_mc = np.median(np.where(S2 == True)[1])
        S2_y_mc = np.median(np.where(S2 == True)[0])

        N1_x_mc = np.median(np.where(N1 == True)[1])
        N1_y_mc = np.median(np.where(N1 == True)[0])

        N2_x_mc = np.median(np.where(N2 == True)[1])
        N2_y_mc = np.median(np.where(N2 == True)[0])

        I1_x_mc = np.median(np.where(I1 == True)[1])
        I1_y_mc = np.median(np.where(I1 == True)[0])

        I2_x_mc = np.median(np.where(I2 == True)[1])
        I2_y_mc = np.median(np.where(I2 == True)[0])

        T1_x_mc = np.median(np.where(T1 == True)[1])
        T1_y_mc = np.median(np.where(T1 == True)[0])

        T2_x_mc = np.median(np.where(T2 == True)[1])
        T2_y_mc = np.median(np.where(T2 == True)[0])

        C0_x_mc = S1_x_mc
        C0_y_mc = N2_y_mc

        coord_list = [C0_x_mc, C0_y_mc, S1_x_mc, S1_y_mc, S2_x_mc, S2_y_mc, \
                      N1_x_mc, N1_y_mc, N2_x_mc, N2_y_mc, I1_x_mc, I1_y_mc, I2_x_mc, I2_y_mc, \
                      T1_x_mc, T1_y_mc, T2_x_mc, T2_y_mc]
        return coord_list

    def pixel_to_mu_meter(self, img):
        img_um = np.multiply(img, 0.0039 * 1000)
        return img_um

    def resize_prediction(self, img):
        prediction_resized = cv2.resize(img, (768, 768))
        return prediction_resized

    def write_depthgrid_values(self, coord_list, value_list, text_size):
        for i in range(0, int(len(coord_list) / 2)):
            plt.text(coord_list[i * 2], coord_list[(i + 1) * 2 - 1], str(int(value_list[i])), ha = 'center',
                     va = 'center',
                     bbox = dict(facecolor = 'white'), size = text_size)

    def plot_fundus(self, label_path, image_path, save_name, save_path, laterality):
        # center image
        label_mu = cv2.resize(np.load(label_path), (768, 768))
        label_mu[label_mu < 25] = 0
        label_mu = self.center_img(label_mu)
        # load image and set margin to zero and center
        fundus_image = cv2.resize(cv2.imread(image_path), (768, 768))
        fundus_image[label_mu == 0] = 0
        fundus_image = self.center_img(fundus_image)

        plt.figure(figsize = (10, 10))
        plt.subplot(1, 1, 1)
        plt.imshow(fundus_image)
        plt.title("fundus: record:{}, laterality: {}".format(save_name, laterality))

        plt.savefig(os.path.join(save_path, str(save_name)))
        plt.close()

    def plot_fundus_label_and_prediction(self, label_path, prediction_path, image_path,
                                         save_path, save_name, laterality, full_abt, answers):
        cm_heidelberg = self.heidelberg_colormap()

        save_name = str(save_name) + ".png"

        prediction_mu = cv2.resize(np.load(os.path.join(prediction_path,
                                                        save_name.replace(".png", ".npy"))).reshape(1, 256, 256, 1)[0,
                                   :, :,
                                   0], (768, 768)) * 500.

        label_mu = cv2.resize(np.load(os.path.join(label_path,
                                                   save_name.replace(".png", ".npy"))), (768, 768))

        # center image
        label_mu[label_mu < 25] = 0
        label_mu = self.center_img(label_mu)
        # center image
        prediction_mu[prediction_mu < 25] = 0
        prediction_mu = self.center_img(prediction_mu)

        percentual_dev = self.percentual_deviance(label_mu, prediction_mu)
        # load image and set margin to zero and center
        fundus_image = cv2.resize(cv2.imread(os.path.join(image_path,
                                                          save_name)), (768, 768))
        fundus_image[label_mu == 0] = 0
        fundus_image = self.center_img(fundus_image)

        # get values for low res grid and coordinates
        label_mu = np.nan_to_num(label_mu)
        low_grid_values_label = self.get_low_res_depth_grid_values(label_mu)

        # get values for low res grid and coordinates
        prediction_mu = np.nan_to_num(prediction_mu)
        low_grid_values_prediction = self.get_low_res_depth_grid_values(prediction_mu)

        # overlay nine area grid
        prediction_mu = np.nan_to_num(prediction_mu)
        prediction = np.copy(prediction_mu)
        low_res_grid_prediction = self.get_low_res_grid_shape(prediction_mu)
        prediction[low_res_grid_prediction.astype(np.bool)[:, :, 0]] = 0

        # overlay nine area grid
        label_mu = np.nan_to_num(label_mu)
        label = np.copy(label_mu)
        low_res_grid = self.get_low_res_grid_shape(label_mu)
        label[low_res_grid.astype(np.bool)[:, :, 0]] = 0

        coord_list = self.get_text_coord(label_mu)

        title_text = "Fundus: Refferal answer: {}, a/e/n answer: {} \n " \
                     "Fundus + prediction: Refferal answer: {}, a/e/n answer: {}\n" \
                     "Gold standard referral: {}, Gold standard a/e/n answer: {}"

        plt.figure(figsize = (40, 20))
        sup_text_size = 35
        text_size = 30
        plt.suptitle(title_text.format(answers["referral_answer_f"], answers["a_e_n_answer_f"],
                                       answers["referral_answer_fp"], answers["a_e_n_answer_fp"],
                                       answers["gold_standard_referral"], answers["gold_standard_a_e_n"]),
                     size = sup_text_size)

        # PLOT FUNDUS
        plt.subplot(2, 3, 1)
        plt.imshow(fundus_image)
        plt.title("Record with percentual deviance of: {}".format(str(percentual_dev)), size = text_size)

        # PLOT LABEL
        plt.subplot(2, 3, 2)
        label_mu = np.ma.masked_where(label_mu < 100, label_mu)
        cmap = cm_heidelberg
        cmap.set_bad(color = 'black')
        plt.imshow(label_mu, cmap = cmap, vmin = 100, vmax = 750)
        plt.title("laterality: {}".format(laterality), size = text_size)
        plt.colorbar(fraction = 0.046, pad = 0.04).ax.tick_params(labelsize = text_size * 0.8)
        plt.title("Groundtruth thickness map", size = text_size)

        # PLOT LABEL WITH LOW RES GRID
        plt.subplot(2, 3, 3)
        label = np.ma.masked_where(label < 100, label)
        cmap = cm_heidelberg
        cmap.set_bad(color = 'black')
        plt.imshow(label, cmap = cmap, vmin = 100, vmax = 750)
        # plt.title("low res:{}, laterality: {}".format(save_name,laterality))
        self.write_depthgrid_values(coord_list, low_grid_values_label, text_size - 10)
        plt.colorbar(fraction = 0.046, pad = 0.04).ax.tick_params(labelsize = text_size * 0.8)

        # PLOT PREDICTION
        plt.subplot(2, 3, 4)
        prediction_mu = np.ma.masked_where(prediction_mu < 100, prediction_mu)
        cmap = cm_heidelberg
        cmap.set_bad(color = 'black')
        plt.imshow(prediction_mu, cmap = cmap, vmin = 100, vmax = 750)
        # plt.title("high res:{}, laterality: {}".format(save_name, laterality))
        plt.colorbar(fraction = 0.046, pad = 0.04).ax.tick_params(labelsize = text_size * 0.8)
        plt.title("Predicted thickness map", size = text_size)

        # PLOT PREDICTION WITH LOW RED GRID
        plt.subplot(2, 3, 5)
        prediction = np.ma.masked_where(prediction < 100, prediction)
        cmap = cm_heidelberg
        cmap.set_bad(color = 'black')
        plt.imshow(prediction, cmap = cmap, vmin = 100, vmax = 750)
        plt.title("laterality: {}".format(laterality), size = text_size)
        self.write_depthgrid_values(coord_list, low_grid_values_prediction, text_size - 10)
        plt.colorbar(fraction = 0.046, pad = 0.04).ax.tick_params(labelsize = text_size * 0.8)

        plt.savefig(os.path.join(save_path, str(save_name)))
        plt.close()

    def heidelberg_colormap(self):
        from matplotlib.colors import LinearSegmentedColormap
        plt.figsize = (40, 40)

        plt.close('all')

        cdict = {

            'blue': ((0.0, 0.0, 0.0),  # black
                     (0.1, 1.0, 1.0),  # purple
                     (0.2, 1.0, 1.0),  # blue
                     (0.3, 0.0, 0.0),  # green
                     (0.4, 0.0, 0.0),  # yellow
                     (0.55, 0.0, 0.0),  # red
                     (0.65, 1.0, 1.0),  # white
                     (1.0, 1.0, 1.0)),  # white

            'green': ((0.0, 0.0, 0.0),  # black
                      (0.1, 0.0, 0.0),  # purple
                      (0.2, 0.0, 0.0),  # blue
                      (0.3, 1.0, 1.0),  # green
                      (0.4, 1.0, 1.0),  # yellow
                      (0.55, 0.0, 0.0),  # red
                      (0.65, 1.0, 1.0),  # white
                      (1.0, 1.0, 1.0)),

            'red': ((0.0, 0.0, 0.0),  # black
                    (0.1, 1.0, 1.0),  # purple
                    (0.2, 0.0, 0.0),  # blue
                    (0.3, 0.0, 0.0),  # green
                    (0.4, 1.0, 1.0),  # yellow
                    (0.55, 1.0, 1.0),  # red
                    (0.65, 1.0, 1.0),  # white
                    (1.0, 1.0, 1.0)),
        }

        cm_heidelberg = LinearSegmentedColormap('bgr', cdict)

        return cm_heidelberg

    def plot_fundus_label_or_prediction_heidelberg_cs(self, record_path, image_path, save_path, save_name, laterality,
                                                      prediction=True):
        if prediction:
            label_mu = cv2.resize(np.load(record_path).reshape(1, 256, 256, 1)[0, :, :, 0], (768, 768)) * 500.
        else:
            label_mu = cv2.resize(np.load(record_path), (768, 768))

        cm_heidelberg = self.heidelberg_colormap()
        # center image
        label_mu[label_mu < 25] = 0
        label_mu = self.center_img(label_mu)

        # load image and set margin to zero and center
        fundus_image = cv2.resize(cv2.imread(image_path), (768, 768))
        fundus_image[label_mu == 0] = 0
        fundus_image = self.center_img(fundus_image)

        # get values for low res grid and coordinates
        label_mu = np.nan_to_num(label_mu)
        low_grid_values_label = self.get_low_res_depth_grid_values(label_mu)

        # overlay nine area grid
        label_mu = np.nan_to_num(label_mu)
        label = np.copy(label_mu)
        low_res_grid = self.get_low_res_grid_shape(label_mu)
        label[low_res_grid.astype(np.bool)[:, :, 0]] = 0

        coord_list = self.get_text_coord(label_mu)
        plt.figure(figsize = (20, 20))
        plt.subplot(1, 3, 1)
        plt.imshow(fundus_image)
        plt.title("fundus")
        plt.subplot(1, 3, 2)
        label_mu = np.ma.masked_where(label_mu < 100, label_mu)
        label_mu[label_mu > 500.0] = 1000
        cmap = cm_heidelberg
        cmap.set_bad(color = 'black')
        plt.imshow(label_mu, cmap = cmap, vmin = 100, vmax = 750)
        plt.title("high res:{}, laterality: {}".format(save_name, laterality))
        plt.colorbar(fraction = 0.046, pad = 0.04)
        plt.subplot(1, 3, 3)
        label = np.ma.masked_where(label < 100, label)
        cmap = cm_heidelberg
        cmap.set_bad(color = 'black')

        plt.imshow(label, cmap = cmap, vmin = 100, vmax = 750)
        plt.title("low res:{}, laterality: {}".format(save_name, laterality))
        self.write_depthgrid_values(coord_list, low_grid_values_label)
        plt.colorbar(fraction = 0.046, pad = 0.04)

        plt.savefig(os.path.join(save_path, str(save_name)))
        plt.close()
