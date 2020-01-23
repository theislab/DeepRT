import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2


class Map():
    def __init__(self, dicom, segmentations, dicom_path):
        self.dicom_path = dicom_path
        self.dicom = dicom
        self.segmentations = segmentations
        self.thickness_map = None

    def crop_image(self, img, tol=0):
        # img is 2D image data
        # tol  is tolerance

        # if 3 dim fundus appears
        if len( img.shape ) > 2:
            mask = img[:, :, 0] > tol
        else:
            mask = img > tol
        return img[np.ix_( mask.any( 1 ), mask.any( 0 ) )]

    def heidelberg_colormap(self):
        from matplotlib.colors import LinearSegmentedColormap
        plt.figsize = (40, 40)

        plt.close( 'all' )

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

        cm_heidelberg = LinearSegmentedColormap( 'bgr', cdict )

        return (cm_heidelberg)

    def plot_thickness_map(self, save_path):
        '''
        :param save_path:
        :return:
        '''
        cm_heidelberg = self.heidelberg_colormap()

        plt.figure( figsize = (5, 5) )

        # crop black margin
        thickness_map = self.crop_image(self.thickness_map, tol = 0)

        thickness_map = np.ma.masked_where(thickness_map < 100,
                                           thickness_map)

        thickness_map[thickness_map > 500.0] = 1000
        cmap = cm_heidelberg
        cmap.set_bad( color = 'black' )
        plt.imshow(thickness_map, cmap = cmap, vmin = 100, vmax = 750)
        plt.axis("off")

        plt.savefig(save_path)
        plt.close()

    def get_iterable_dimension(self):
        y_iter = None
        x_iter = None

        # determine y or x iterable
        y = np.unique( self.dicom.record_lookup.y_starts ).shape[0]
        x = np.unique( self.dicom.record_lookup.x_starts ).shape[0]
        if y > x:
            y_iter = "iterable"
        else:
            x_iter = "iterable"
        return y_iter, x_iter

    def oct_pixel_to_mu_m(self, depth_vector, iter):
        y_scale = float( self.dicom.record_lookup.y_scales.iloc[iter] )
        return (np.multiply( depth_vector, y_scale * 1000 ))

    def get_position_series(self):
        startx_pos = self.dicom.record_lookup.x_starts.reset_index( drop = True ).fillna( 0 )
        endx_pos = self.dicom.record_lookup.x_ends.reset_index( drop = True ).fillna( 0 )
        starty_pos = self.dicom.record_lookup.y_starts.reset_index( drop = True ).fillna( 0 )
        endy_pos = self.dicom.record_lookup.y_ends.reset_index( drop = True ).fillna( 0 )
        return startx_pos, endx_pos, starty_pos, endy_pos

    def get_depth_vector(self, img):
        def find_nearest_min(array, value):
            idx = (np.abs( array - value )).argmin()
            return idx

        # def find_nearest_max(array, value):
        #     idx = (np.abs(array - value)).argmax()
        #     return idx

        def get_zero_patches(idx_zero):

            def split(arr, cond):
                return [arr[cond], arr[~cond]]

            diff_vector = np.zeros( 1 + idx_zero.shape[0] )
            diff_vector[1:] = idx_zero
            differences = np.subtract( idx_zero, diff_vector[0:-1] )[1:]
            indices = np.where( differences > 1 )
            # print(indices, differences)
            zero_patches = []

            if indices[0].size != 0:
                # extract first zero patch
                zero_patches.append( idx_zero[np.where( idx_zero <= idx_zero[indices[0][0]] )] )
                # extract the following zero patches
                for i in range( indices[0][:].shape[0], 0, -1 ):
                    zero_patches.append( idx_zero[np.where( idx_zero > idx_zero[indices[0][i - 1]] )] )
            if indices[0].size == 0:
                zero_patches.append( idx_zero )
            return (zero_patches)

        depth_vector = np.zeros( img.shape[1] )
        for i in range( 0, img.shape[1] ):
            layer = np.argwhere( img[:, i] )
            if layer.size != 0:
                depth_vector[i] = max( layer ) - min( layer )
        # get all indices
        idx_nonzero = np.argwhere( depth_vector )
        idx_zero = np.where( depth_vector == 0 )[0]
        # if no non zero, return zero depth vector
        if idx_nonzero.size == 0:
            return (np.zeros( img.shape[1] ))
        # check if list is empty = no zero patches
        if len( idx_zero ) != 0:
            # get list with seperate zero patches
            zero_patches = get_zero_patches( idx_zero )
            # find interpolation value
            for patch in zero_patches:
                # print(patch)
                closest_min = find_nearest_min( idx_nonzero, min( patch ) )
                closest_max = find_nearest_min( idx_nonzero, max( patch ) )
                # print(closest_min, closest_max)
                interpolation = (depth_vector[idx_nonzero[closest_min]] + depth_vector[idx_nonzero[closest_max]]) / 2
                # interpolate
                depth_vector[patch] = interpolation

        return depth_vector

    def depth_grid(self):
        # get fundus dimension
        global grid_pd_int
        depth_grid_dim = (768, 768)
        y_cord, x_cord = self.get_iterable_dimension()
        grid = np.zeros( depth_grid_dim )
        for i in range( 0, len( self.segmentations ) ):
            d_v = self.get_depth_vector( self.segmentations[i] )
            # scale dv to mm
            d_v = self.oct_pixel_to_mu_m( d_v, i )
            # get starting and ending x,y series with new indices
            startx_pos, endx_pos, starty_pos, endy_pos = self.get_position_series()
            try:
                if y_cord == "iterable":
                    # set assert indices are ints
                    x_start = int( startx_pos[i] )
                    x_end = int( endx_pos[i] )
                    y_start = int( starty_pos[i] )

                    # assert d_v has same width as x_end -x_start
                    if d_v.shape[0] > (x_end - x_start):
                        d_v = d_v[0:x_end - x_start]
                    if d_v.shape[0] < (x_end - x_start):
                        difference = (x_end - x_start) - d_v.shape[0]
                        d_v = np.append( d_v, np.zeros( int( difference ) ) )
                    # shift indices when laterilty changes to "L"
                    # in case x_start is negative in xml it is set to zero and x_end
                    # reduced correspondingly to fit the depth vector
                    # scale depth vector to mm

                    grid[y_start, x_start:x_end] = d_v
                if x_cord == "iterable":
                    # assert indices are ints
                    # set assert indices are ints
                    y_start = int( starty_pos[i] )
                    y_end = int( endy_pos[i] )
                    x_start = int( startx_pos[i] )

                    # assert d_v has same width as x_end -x_start
                    if d_v.shape[0] > (y_start - y_end):
                        d_v = d_v[0:y_start - y_end]
                    if d_v.shape[0] < (y_start - y_end):
                        difference = (y_start - y_end) - d_v.shape[0]
                        d_v = np.append( d_v, np.zeros( difference ) )
                    # shift indices when laterilty changes to "L"
                    grid[y_end:y_start, x_start] = d_v
            except:
                print( "COULD NOT CALCULATE GRID" )
                print( self.dicom.record_lookup.record_id )

        # linearly interpolate missing values
        grid[grid == 0] = np.nan
        # interpolate depending on which axis the depth vector is filled in
        if x_cord == "iterable":
            grid_pd_int = pd.DataFrame( grid ).interpolate( limit_direction = 'both',
                                                            axis = 1 )
            # set all areas outside of measurements to 0
            min_startx = int( min( startx_pos.iloc[1:] ) )
            max_startx = int( max( startx_pos.iloc[1:] ) )

            grid_pd_int.loc[:, max_startx:grid_pd_int.shape[1]] = 0
            grid_pd_int.loc[:, 0:min_startx] = 0

        if y_cord == "iterable":
            grid_pd_int = pd.DataFrame( grid ).interpolate( limit_direction = 'both',
                                                            axis = 0 )
            # set all areas outside of measurements to 0
            min_starty = int( min( starty_pos.iloc[1:] ) )
            max_starty = int( max( starty_pos.iloc[1:] ) )

            grid_pd_int[max_starty:grid_pd_int.shape[0]] = 0
            grid_pd_int[0:min_starty] = 0

            grid_pd_int = grid_pd_int.fillna( 0 )

            self.thickness_map = np.array(grid_pd_int)
