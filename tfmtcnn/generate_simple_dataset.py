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

r"""Generates a basic dataset i.e. PNet dataset.

Usage:
```shell

$ python generate_simple_dataset.py \
	--annotation_image_dir=./data/WIDER_Face/WIDER_train/images \ 
	--annotation_file_name=./data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt \
	--landmark_image_dir=./data/CelebA/images \
	--landmark_file_name=./data/CelebA/CelebA.txt \
	--target_root_dir=./data/datasets/mtcnn 

$ python generate_simple_dataset.py \
	--annotation_image_dir=./data/WIDER_Face/WIDER_train/images \ 
	--annotation_file_name=./data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt \
	--landmark_image_dir=./data/CelebA/images \
	--landmark_file_name=./data/CelebA/CelebA.txt \
	--sample_multiplier_factor=20 \
	--target_root_dir=./data/datasets/mtcnn 
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

from datasets.SimpleDataset import SimpleDataset

default_multiplier_factor = 10

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--annotation_image_dir', type=str, help='Input WIDER face dataset training image directory.', default=None)
	parser.add_argument('--annotation_file_name', type=str, help='Input WIDER face dataset annotation file.', default=None)

	parser.add_argument('--landmark_image_dir', type=str, help='Input landmark dataset training image directory.', default=None)
	parser.add_argument('--landmark_file_name', type=str, help='Input landmark dataset annotation file.', default=None)

	parser.add_argument('--sample_multiplier_factor', type=int, help='Number of samples generated from one input sample.', default=default_multiplier_factor)

	parser.add_argument('--target_root_dir', type=str, help='Output directory where output images and TensorFlow data files are saved.', default=None)
	return(parser.parse_args(argv))

def main(args):

	if(not args.annotation_image_dir):
		raise ValueError('You must supply input WIDER face dataset training image directory with --annotation_image_dir.')
	if(not args.annotation_file_name):
		raise ValueError('You must supply input WIDER face dataset annotation file with --annotation_file_name.')

	if(not args.landmark_image_dir):
		raise ValueError('You must supply input landmark dataset training image directory with --landmark_image_dir.')		
	if(not args.landmark_file_name):
		raise ValueError('You must supply input landmark dataset annotation file with --landmark_file_name.')				

	if(not args.target_root_dir):
		raise ValueError('You must supply output directory for storing output images and TensorFlow data files with --target_root_dir.')

	if(args.sample_multiplier_factor < 1):
		sample_multiplier_factor = default_multiplier_factor
	else:
		sample_multiplier_factor = args.sample_multiplier_factor

	simple_dataset = SimpleDataset()
	status = simple_dataset.generate(args.annotation_image_dir, args.annotation_file_name, args.landmark_image_dir, args.landmark_file_name, sample_multiplier_factor, args.target_root_dir)
	if(status):
		print('PNet dataset is generated at ' + args.target_root_dir)
	else:
		print('Error generating basic dataset.')

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))


