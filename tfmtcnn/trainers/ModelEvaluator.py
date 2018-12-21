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
import cv2

from tfmtcnn.networks.FaceDetector import FaceDetector

from tfmtcnn.datasets.DatasetFactory import DatasetFactory
from tfmtcnn.datasets.InferenceBatch import InferenceBatch
import tfmtcnn.datasets.constants as datasets_constants

from tfmtcnn.utils.convert_to_square import convert_to_square
from tfmtcnn.utils.IoU import IoU


class ModelEvaluator(object):
    def __init__(self):
        self._test_dataset = None
        self._face_detector = None

    def load(self, dataset_name, annotation_image_dir, annotation_file_name):
        face_dataset = DatasetFactory.face_dataset(dataset_name)
        if (not face_dataset):
            return (False)

        if (not face_dataset.read(annotation_image_dir, annotation_file_name)):
            return (False)

        self._test_dataset = face_dataset.data()
        return (True)

    def create_detector(self, last_network, model_root_dir):
        self._face_detector = FaceDetector(last_network, model_root_dir)

        minimum_face_size = datasets_constants.minimum_face_size
        self._face_detector.set_min_face_size(minimum_face_size)
        return (True)

    def evaluate(self, print_result=False):
        if (not self._test_dataset):
            return (False)

        if (not self._face_detector):
            return (False)

        test_data = InferenceBatch(self._test_dataset['images'])
        detected_boxes, landmarks = self._face_detector.detect_face(test_data)

        image_file_names = self._test_dataset['images']
        ground_truth_boxes = self._test_dataset['bboxes']
        number_of_images = len(image_file_names)

        if (not (len(detected_boxes) == number_of_images)):
            return (False)

        number_of_positive_faces = 0
        number_of_part_faces = 0
        number_of_ground_truth_faces = 0
        accuracy = 0.0
        for image_file_path, detected_box, ground_truth_box in zip(
                image_file_names, detected_boxes, ground_truth_boxes):
            ground_truth_box = np.array(
                ground_truth_box, dtype=np.float32).reshape(-1, 4)
            number_of_ground_truth_faces = number_of_ground_truth_faces + len(
                ground_truth_box)
            if (detected_box.shape[0] == 0):
                continue

            detected_box = convert_to_square(detected_box)
            detected_box[:, 0:4] = np.round(detected_box[:, 0:4])

            current_image = cv2.imread(image_file_path)

            for box in ground_truth_box:

                #x_left, y_top, x_right, y_bottom, _ = box.astype(int)
                #width = x_right - x_left + 1
                #height = y_bottom - y_top + 1

                #if( (x_left < 0) or (y_top < 0) or (x_right > (current_image.shape[1] - 1) ) or (y_bottom > (current_image.shape[0] - 1 ) ) ):
                #	continue

                current_IoU = IoU(box, detected_box)
                maximum_IoU = np.max(current_IoU)

                accuracy = accuracy + maximum_IoU
                if (maximum_IoU > datasets_constants.positive_IoU):
                    number_of_positive_faces = number_of_positive_faces + 1
                elif (maximum_IoU > datasets_constants.part_IoU):
                    number_of_part_faces = number_of_part_faces + 1

        if (print_result):
            print('Positive faces       - ', number_of_positive_faces)
            print('Partial faces        - ', number_of_part_faces)
            print('Total detected faces - ',
                  (number_of_positive_faces + number_of_part_faces))
            print('Ground truth faces   - ', number_of_ground_truth_faces)
            print('Positive accuracy    - ',
                  number_of_positive_faces / number_of_ground_truth_faces)
            print('Detection accuracy   - ',
                  (number_of_positive_faces + number_of_part_faces) /
                  number_of_ground_truth_faces)
            print('Accuracy             - ',
                  accuracy / number_of_ground_truth_faces)

        return (True)
