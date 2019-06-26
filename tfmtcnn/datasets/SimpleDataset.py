# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tfmtcnn.datasets.AbstractDataset import AbstractDataset
from tfmtcnn.datasets.LandmarkDataset import LandmarkDataset
from tfmtcnn.datasets.SimpleFaceDataset import SimpleFaceDataset
from tfmtcnn.datasets.TensorFlowDataset import TensorFlowDataset

from tfmtcnn.networks.NetworkFactory import NetworkFactory

# generates simple dataset from input landmark and bounding box datasets.
# this is used for training PNet.
class SimpleDataset(AbstractDataset):
    def __init__(self, network_name='PNet'):
        AbstractDataset.__init__(self, network_name)

    def _generate_landmark_samples(self, landmark_image_dir,
                                   landmark_file_name, base_number_of_images,
                                   target_root_dir):
        landmark_dataset = LandmarkDataset()
        return (landmark_dataset.generate(
            landmark_image_dir, landmark_file_name, base_number_of_images,
            NetworkFactory.network_size(self.network_name()), target_root_dir))

    def _generate_image_samples(self, annotation_image_dir,
                                annotation_file_name, base_number_of_images,
                                target_root_dir):
        face_dataset = SimpleFaceDataset()
        return (face_dataset.generate_samples(
            annotation_image_dir, annotation_file_name, base_number_of_images,
            NetworkFactory.network_size(self.network_name()), target_root_dir))

    def _generate_image_list(self, target_root_dir):
        positive_file = open(
            SimpleFaceDataset.positive_file_name(target_root_dir), 'r')
        positive_data = positive_file.readlines()

        part_file = open(
            SimpleFaceDataset.part_file_name(target_root_dir), 'r')
        part_data = part_file.readlines()

        negative_file = open(
            SimpleFaceDataset.negative_file_name(target_root_dir), 'r')
        negative_data = negative_file.readlines()

        landmark_file = open(
            LandmarkDataset.landmark_file_name(target_root_dir), 'r')
        landmark_data = landmark_file.readlines()

        image_list_file = open(
            self._image_list_file_name(target_root_dir), 'w')

        for i in np.arange(len(positive_data)):
            image_list_file.write(positive_data[i])

        for i in np.arange(len(negative_data)):
            image_list_file.write(negative_data[i])

        for i in np.arange(len(part_data)):
            image_list_file.write(part_data[i])

        for i in np.arange(len(landmark_data)):
            image_list_file.write(landmark_data[i])

        return (True)

    def _generate_dataset(self, target_root_dir):
        tensorflow_dataset = TensorFlowDataset()
        if (not tensorflow_dataset.generate(
                self._image_list_file_name(target_root_dir), target_root_dir,
                'image_list')):
            return (False)

        return (True)
    
    # generates image and landmark samples from the input data from WIDER Face and CelebA datasets.
    # it also generates Tensorflow compatible dataset
    def generate(self, annotation_image_dir, annotation_file_name,
                 landmark_image_dir, landmark_file_name, base_number_of_images,
                 target_root_dir):

        if (not os.path.isfile(annotation_file_name)):
            return (False)
        if (not os.path.exists(annotation_image_dir)):
            return (False)

        if (not os.path.isfile(landmark_file_name)):
            return (False)
        if (not os.path.exists(landmark_image_dir)):
            return (False)

        target_root_dir = os.path.expanduser(target_root_dir)
        target_root_dir = os.path.join(target_root_dir, self.network_name())
        if (not os.path.exists(target_root_dir)):
            os.makedirs(target_root_dir)

        print('Generating image samples.')
        if (not self._generate_image_samples(
                annotation_image_dir, annotation_file_name,
                base_number_of_images, target_root_dir)):
            print('Error generating image samples.')
            return (False)
        print('Generated image samples.')

        print('Generating landmark samples.')
        if (not self._generate_landmark_samples(
                landmark_image_dir, landmark_file_name, base_number_of_images,
                target_root_dir)):
            print('Error generating landmark samples.')
            return (False)
        print('Generated landmark samples.')

        if (not self._generate_image_list(target_root_dir)):
            return (False)

        print('Generating TensorFlow dataset.')
        if (not self._generate_dataset(target_root_dir)):
            print('Error generating TensorFlow dataset.')
            return (False)
        print('Generated TensorFlow dataset.')

        return (True)
