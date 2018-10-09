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

r"""Evaluate the MTCNN model for the given dataset.

Usage:
```shell

$ python tfmtcnn/tfmtcnn/evaluate_model.py \
	--model_root_dir tfmtcnn/models/mtcnn/wider-celeba \
	--annotation_image_dir ../data/WIDER_Face/WIDER_val/images \ 
	--annotation_file_name ../data/WIDER_Face/WIDER_val/wider_face_val_bbx_gt.txt 
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import cv2

import tfmtcnn.datasets.constants as datasets_constants
from tfmtcnn.datasets.DatasetFactory import DatasetFactory
from tfmtcnn.datasets.InferenceBatch import InferenceBatch

from tfmtcnn.networks.FaceDetector import FaceDetector
from tfmtcnn.networks.NetworkFactory import NetworkFactory

from tfmtcnn.utils.convert_to_square import convert_to_square
from tfmtcnn.utils.IoU import IoU

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_root_dir', type=str, help='Input model root directory where model weights are saved.', default=None)

	parser.add_argument('--annotation_file_name', type=str, help='Input WIDER face dataset annotations file.', default=None)
	parser.add_argument('--annotation_image_dir', type=str, help='Input WIDER face dataset training image directory.', default=None)

	return(parser.parse_args(argv))

def main(args):

	if(not args.annotation_file_name):
		raise ValueError('You must supply input WIDER face dataset annotations file with --annotation_file_name.')
	if(not args.annotation_image_dir):
		raise ValueError('You must supply input WIDER face dataset training image directory with --annotation_image_dir.')		

	if(args.model_root_dir):
		model_root_dir = args.model_root_dir
	else:
		model_root_dir = NetworkFactory.model_deploy_dir()

	last_network='ONet'
	face_detector = FaceDetector(last_network, model_root_dir)
	minimum_face_size = datasets_constants.minimum_face_size

	dataset = None
	status = False
	face_dataset_name='WIDERFaceDataset'
	face_dataset = DatasetFactory.face_dataset(face_dataset_name)
	if(face_dataset.read(args.annotation_image_dir, args.annotation_file_name)):			
		dataset = face_dataset.data()
		status = True

	if(not status):
		return

	test_data = InferenceBatch(dataset['images'])
	detected_boxes, landmarks = face_detector.detect_face(test_data)

	image_file_names = dataset['images']
	ground_truth_boxes = dataset['bboxes']
	number_of_images = len(image_file_names)

	if(not (len(detected_boxes) == number_of_images)):
		return

	accuracy = 0.0 
	number_of_faces = 0
	for image_file_path, detected_box, ground_truth_box in zip(image_file_names, detected_boxes, ground_truth_boxes):
       		ground_truth_box = np.array(ground_truth_box, dtype=np.float32).reshape(-1, 4)
		number_of_faces = number_of_faces + len(ground_truth_box)
       		if( detected_box.shape[0] == 0 ):
            			continue

       		detected_box = convert_to_square(detected_box)
       		detected_box[:, 0:4] = np.round(detected_box[:, 0:4])

		current_image = cv2.imread(image_file_path)

       		for box in detected_box:
       			x_left, y_top, x_right, y_bottom, _ = box.astype(int)
       			width = x_right - x_left + 1
       			height = y_bottom - y_top + 1

       			if( (width < minimum_face_size) or (height < minimum_face_size) or (x_left < 0) or (y_top < 0) or (x_right > (current_image.shape[1] - 1) ) or (y_bottom > (current_image.shape[0] - 1 ) ) ):
               			continue

			current_IoU = IoU(box, ground_truth_box)
			maximum_IoU = np.max(current_IoU)
			print('maximum_IoU', maximum_IoU)
			if(maximum_IoU > 0.01):
				accuracy = accuracy + 1

	print('Accuracy - ', accuracy, 'number_of_faces', number_of_faces)

if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))


