#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: VSCode
@author: Filipe Bento
@contact: dev.bento@gmail.com
@created: jun-16 of 2021
"""

# Imports for defining models
from tensorflow import keras

# Imports for general purposes
import tensorflow as tf
import numpy as np
import math

class HCMaskLayer(keras.layers.Layer):

    def __init__(self, _input_shape, mask, group_by='mean', **_kwargs):
        _kwargs['input_shape'] = _input_shape
        super(HCMaskLayer, self).__init__(**_kwargs)
        self.shape = _input_shape
        self.mask_length = int(_input_shape[0])
        self._i = 1
        self.mask = mask
        self.warm()

    def build(self, input_shape):
        self.warm()
        super(HCMaskLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self._output_shape[0], self._output_shape[1]

    def compute_input_shape(self, input_shape):
        """
        Compute input shape

        Function to compute end input shape result after max pooling.
        Adapted from https://github.com/keras-team/keras/blob/f899d0fb336cce4061093a575a573c4f897106e1/keras/layers/pooling.py#L180

        :param input_shape: Shape of input
        :type input_shape: Tensor, dimension [batch size x height x width x channels x instantiation parameters]
        """
        return input_shape[0], self._output_shape[0], self._output_shape[1]

    def get_config(self):
        config = super(HCMaskLayer, self).get_config()
        return config

    def call(self, input_image):
        self._i += 1

        applied_mask = self.apply_mask(input_image)
        tensor_hc = tf.convert_to_tensor(applied_mask, np.float32)

        # hilbert_x, hilbert_y = self.hilbert_curve_coordinates(
        #     self.shape[1:], self.depth)
        # # Transpose the input to put the batch axis as the last one
        # x_transpose = tf.transpose(x, perm=[1, 2, 3, 0])
        # # Apply the indexing to get the data
        # response_transposed = tf.gather_nd(
        #     x_transpose, np.c_[hilbert_x, hilbert_y])
        # # Transpose again the input to put the batch axis as the first one
        # response = tf.transpose(response_transposed, perm=[2, 0, 1])

        return tensor_hc

    def xy2d(self, n, x, y):
        d = 0
        s = int(n/2)
        while s > 0:
            rx = (x & s) > 0
            ry = (y & s) > 0
            d += s*s*((3*rx) ^ ry)
            s, x, y, rx, ry = self.rot(s, x, y, rx, ry)
            s = int(s / 2)
        return d

    def d2xy(self, n, d):
        t = d
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & int(t/2)
            ry = 1 & (t ^ rx)
            s, x, y, rx, ry = self.rot(s, x, y, rx, ry)
            x += s*rx
            y += s*ry
            t = int(t/4)
            s *= 2
        return x, y

    def rot(self, n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n-1-x
                y = n-1-y
            x, y = y, x

        return n, x, y, rx, ry
    
    def get_mask_oriented(self):
        '''
            Each filter object has the { 'coordX:coordY': filter_size} format.

            The coordinates refers to the bottom left corner of the filter.

            This method calculates the other 3 corners and gets each of them is the origin of the curve in that section.

            At the end, the filters_oriented object will contain the index of the filter in the hilbert curve as key, and the coordinates, center point and length of the filter as value.

            Ex.:
            {
                256: {
                    'origin': '16:0', # the filter in the mask that starts at row 16 ans col 0
                    'center': () # that has the center point of
                    'length': 16 # and length of 16 width and 16 height
                }
            }
        '''

        # Maps for each key in filters dictionary.
        for filter_key in iter(self.mask):
            # Splits the key. Ex: '0:0' -> ['0', '0']
            coords = filter_key.split(':')

            # Converts to integers and deconstruct into coordX and coordY.
            coord_x, coord_y = int(coords[0]), int(coords[1])

            # Gets the filter length, decreased by one, because iteration starts at 0.
            filter_length = self.mask[filter_key] - 1

            # Gets all the four corners coordinates of the filter.
            corners = [
                (coord_x, coord_y),  # bottom left
                (coord_x + filter_length, coord_y),  # top left
                (coord_x + filter_length, coord_y + filter_length),  # top right
                (coord_x, coord_y + filter_length)  # bottom right
            ]

            # List all the corresponding indexes of each corner mapped in the hilbert curve.
            corners_indexes = list(map(lambda coord: (self.xy2d(
                self.mask_length, coord[0], coord[1]), coord), corners))

            # Order the indexes, so the lowest value corresponds to the origing cordnate of the curve in that resolution.
            corners_indexes.sort()

            # Gets the origin coordinate.
            hc_origin = corners_indexes[0][0]

            # Calculate the center point of the filter.
            # Get the mid X coord
            centerX = coord_x + int(filter_length/2)
            # Get the mid Y coord
            centerY = coord_y + int(filter_length/2)
            center = (centerX, centerY)

            final_corners = list(map(lambda coord: coord[1], corners_indexes))

            # Saves the parameters for the filter
            self.mask_oriented[hc_origin] = {
                'coords': filter_key,
                'corners': final_corners,
                'center': center,
                'length': self.mask[filter_key]
            }

        # Sort filters by order of appearance in hilbert curve indexes.
        self.mask_oriented = dict(
            sorted(self.mask_oriented.items(), key=lambda item: item[0]))

    def apply_mask(self, input_image, group_by="mean"):
        final_hc = []
        self.multi_level_hc_centers = []

        converted_image = np.array(input_image)

        for index in iter(self.mask_oriented):
            corner = self.mask_oriented[index]['corners']
            origin_x, origin_y = corner[0]
            diagonal_x, diagonal_y = corner[2]
            section = converted_image[origin_x: origin_x + diagonal_x, origin_y: origin_y + diagonal_y]
            mean_y = np.mean(section, 1);
            # print(mean_y.shape)
            mean = np.mean(mean_y, 0);
            # print(origin_x, origin_y, diagonal_x, diagonal_y, section)
            final_hc.append(mean)
        
        return np.array(final_hc)

    def warm(self):
        self.get_mask_oriented()