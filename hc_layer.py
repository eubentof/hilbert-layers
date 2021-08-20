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

class HCLayer(keras.layers.Layer):

    def __init__(self, _depth, _input_shape, **_kwargs):
        _kwargs['input_shape'] = _input_shape
        super(HCLayer, self).__init__(**_kwargs)
        self.depth = _depth
        self.shape = _input_shape
        self._output_shape = (4 ** _depth, 1)
        self._i = 1
        self.warm()

    def build(self, input_shape):
        self.warm()
        super(HCLayer, self).build(input_shape)

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
        config = super(HCLayer, self).get_config()
        return config

    def call(self, x, mask=None):
        self._i += 1
        hilbert_x, hilbert_y = self.hilbert_curve_coordinates(
            self.shape[1:], self.depth)
        # Transpose the input to put the batch axis as the last one
        x_transpose = tf.transpose(x, perm=[1, 2, 3, 0])
        # Apply the indexing to get the data
        response_transposed = tf.gather_nd(
            x_transpose, np.c_[hilbert_x, hilbert_y])
        # Transpose again the input to put the batch axis as the first one
        response = tf.transpose(response_transposed, perm=[2, 0, 1])

        return response

    hilbert_curves = {}

    def _create_hilbert_curve(self, depth_):

        try:
            return self.hilbert_curves[depth_]
        except KeyError:
            depth = 13
            curve = np.zeros(shape=(4 ** depth, 2)).astype(np.int32)
            # curve[0:4, :] = [[0, 0, 1, 1], [0, 1, 1, 0]]
            curve[0:4, :] = [[0, 0], [0, 1], [1, 1], [1, 0]]

            self.hilbert_curves[1] = curve[0:4, :]

            current_step = 1
            current_hilbert_curve_size = 1

            for i in range(2, depth + 1):
                current_step *= 2
                current_hilbert_curve_size *= 4

                figure_x = np.copy(curve[0:current_hilbert_curve_size, 0])
                figure_y = np.copy(curve[0:current_hilbert_curve_size, 1])

                # Left up
                curve[0:current_hilbert_curve_size, 0] = figure_y
                curve[0:current_hilbert_curve_size, 1] = figure_x

                # Left bottom
                curve[current_hilbert_curve_size:current_hilbert_curve_size * 2, 0] = figure_x
                curve[current_hilbert_curve_size:current_hilbert_curve_size *
                      2, 1] = figure_y + current_step

                # Right bottom
                curve[current_hilbert_curve_size *
                      2:current_hilbert_curve_size * 3, 0] = figure_x + current_step
                curve[current_hilbert_curve_size *
                      2:current_hilbert_curve_size * 3, 1] = figure_y + current_step

                # Right up
                curve[current_hilbert_curve_size * 3:current_hilbert_curve_size *
                      4, 0] = current_step * 2 - 1 - figure_y
                curve[current_hilbert_curve_size * 3:current_hilbert_curve_size *
                      4, 1] = current_step - 1 - figure_x

                # cls.hilbert_curves[i] = np.flip(curve[0:current_hilbert_curve_size * 4, :], 1)
                self.hilbert_curves[i] = curve[0:current_hilbert_curve_size * 4, :]

            return self.hilbert_curves[depth_]

    def _get_coordinates_from_image(self, input_shape_, depth_):
        def get_info_from_an_axis(input_shape__, depth__, axis_):
            step_size = input_shape__[axis_] / 2 ** depth__
            ceil = np.ceil(step_size)
            floor = np.floor(step_size)

            if np.abs(step_size - int(step_size)) > 0.001:
                if np.abs(step_size - int(step_size) - 0.5) < 0.001:
                    def step_size_add_function(x): return [ceil, floor][x % 2]
                elif np.abs(step_size - int(step_size)) > 0.7:
                    def step_size_add_function(x): return [
                        ceil, ceil, ceil, floor][x % 4]
                else:
                    def step_size_add_function(x): return ceil
            else:
                def step_size_add_function(x): return ceil
            begin = max(floor - np.ceil(ceil / 2), 0)
            return begin, step_size_add_function

        begin_x, step_size_add_function_x = get_info_from_an_axis(
            input_shape_, depth_, 0)
        begin_y, step_size_add_function_y = get_info_from_an_axis(
            input_shape_, depth_, 1)
        response_x, response_y = [], []
        i = 0
        while begin_x < input_shape_[0]:
            response_x.append(int(begin_x))
            begin_x += step_size_add_function_x(i)

            response_y.append(int(begin_y))
            begin_y += step_size_add_function_y(i)
            i += 1

        return response_x, response_y

    def _closest_power(self, number_):
        possible_results = math.floor(
            math.log(number_, 2)), math.ceil(math.log(number_, 2))
        return min(possible_results, key=lambda z: abs(number_ - 2 ** z))

    def hilbert_curve_coordinates(self, input_shape_, depth_):
        max_depth = self._closest_power(input_shape_[0])
        max_depth = min(max_depth if 2 ** max_depth <=
                        input_shape_[0] else max_depth - 1, depth_)

        simple_hilbert_curve = self._create_hilbert_curve(max_depth)
        coordinates_x, coordinates_y = self._get_coordinates_from_image(
            input_shape_, max_depth)

        hilbert_x = np.take(coordinates_x, simple_hilbert_curve[:, 0])
        hilbert_y = np.take(coordinates_y, simple_hilbert_curve[:, 1])

        return hilbert_x, hilbert_y

    def warm(self):
        self._create_hilbert_curve(1)