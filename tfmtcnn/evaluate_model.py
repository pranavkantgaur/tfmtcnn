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
	--model_root_dir tfmtcnn/tfmtcnn/models/mtcnn/train \
	--annotation_image_dir /datasets/WIDER_Face/WIDER_val/images \ 
	--annotation_file_name /datasets/WIDER_Face/WIDER_val/wider_face_val_bbx_gt.txt

$ python tfmtcnn/tfmtcnn/evaluate_model.py \
	--model_root_dir tfmtcnn/tfmtcnn/models/mtcnn/train \
	--dataset_name FDDBDataset \
	--annotation_image_dir /datasets/FDDB/ \ 
	--annotation_file_name /datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt  
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

import tfmtcnn.datasets.constants as datasets_constants

from tfmtcnn.networks.FaceDetector import FaceDetector

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_root_dir', type=str, help='Input model root directory where model weights are saved.', default=None)

	parser.add_argument('--dataset_name', type=str, help='Input dataset name.', choices=['WIDERFaceDataset', 'CelebADataset', 'FDDBDataset'], default='WIDERFaceDataset')
	parser.add_argument('--annotation_file_name', type=str, help='Input face dataset annotations file.', default=None)
	parser.add_argument('--annotation_image_dir', type=str, help='Input face dataset image directory.', default=None)

	return(parser.parse_args(argv))

def main(args):

	if(not args.annotation_file_name):
		raise ValueError('You must supply input face dataset annotations file with --annotation_file_name.')
	if(not args.annotation_image_dir):
		raise ValueError('You must supply input face dataset training image directory with --annotation_image_dir.')		

	if(args.model_root_dir):
		model_root_dir = args.model_root_dir
	else:
		model_root_dir = NetworkFactory.model_deploy_dir()

	last_network='ONet'
	face_detector = FaceDetector(last_network, model_root_dir)

	minimum_face_size = datasets_constants.minimum_face_size
	face_detector.set_min_face_size(minimum_face_size)

	status = face_detector.evaluate(args.dataset_name, args.annotation_image_dir, args.annotation_file_name, True)
	if(not status):
		print('Error evaluating the model')

if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))


