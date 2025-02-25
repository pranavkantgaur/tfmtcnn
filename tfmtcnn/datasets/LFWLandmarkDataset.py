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
import cv2
import numpy as np

import tfmtcnn.datasets.constants as datasets_constants
from tfmtcnn.utils.BBox import BBox


class LFWLandmarkDataset(object):

    __name = 'LFWLandmark'
    __minimum_face_size = datasets_constants.minimum_dataset_face_size

    @classmethod
    def name(cls):
        return (LFWLandmarkDataset.__name)

    @classmethod
    def minimum_face_size(cls):
        return (LFWLandmarkDataset.__minimum_face_size)

    def __init__(self):
        self._clear()

    def _clear(self):
        self._is_valid = False
        self._data = dict()
        self._number_of_faces = 0

    def is_valid(self):
        return (self._is_valid)

    def data(self):
        return (self._data)

    def read(self, landmark_image_dir, landmark_file_name):

        self._clear()

        if (not os.path.isfile(landmark_file_name)):
            return (False)

        images = []
        bounding_boxes = []
        landmarks = []
        landmark_file = open(landmark_file_name, 'r')
        while (True):
            line = landmark_file.readline().strip()
            landmark_data = line.split(' ')

            image_path = landmark_data[0]
            if (not image_path):
                break
            else:
                image_path = os.path.join(landmark_image_dir, landmark_data[0])

            bounding_box = (landmark_data[1], landmark_data[3],
                            landmark_data[2], landmark_data[4])
            bounding_box = map(int, bounding_box)
            image_width = bounding_box[2] - bounding_box[0]
            image_height = bounding_box[3] - bounding_box[1]

            landmark = np.zeros((5, 2))
            for index in range(0, 5):
                point = (float(landmark_data[5 + 2 * index]),
                         float(landmark_data[5 + 2 * index + 1]))
                landmark[index] = point

            if ((max(image_width, image_height) >=
                 LFWLandmarkDataset.minimum_face_size()) and (image_width > 0)
                    and (image_height > 0)):
                images.append(image_path)
                bounding_boxes.append(BBox(bounding_box))
                landmarks.append(landmark)
                self._number_of_faces += 1

        if (len(images)):
            self._data['images'] = images
            self._data['bboxes'] = bounding_boxes
            self._data['landmarks'] = landmarks
            self._data['number_of_faces'] = self._number_of_faces

            self._is_valid = True
            print(self._number_of_faces, 'faces in ', len(images),
                  'number of images for LFWLandmark dataset')
        else:
            self._clear()

        return (self.is_valid())
