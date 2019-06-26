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

$ python tfmtcnn/tfmtcnn/generate_simple_dataset.py \
	--annotation_image_dir ../data/WIDER_Face/WIDER_train/images \ 
	--annotation_file_name ../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt \
	--landmark_image_dir ../data/CelebA/images \
	--landmark_file_name ../data/CelebA/CelebA.txt \
	--base_number_of_images 700000 \
	--target_root_dir ../data/datasets/mtcnn 
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

from tfmtcnn.datasets.SimpleDataset import SimpleDataset
import tfmtcnn.datasets.constants as datasets_constants

default_base_number_of_images = datasets_constants.default_base_number_of_images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--annotation_image_dir',
        type=str,
        help='Input WIDER face dataset training image directory.',
        default=None)

    parser.add_argument(
        '--annotation_file_name',
        type=str,
        help='Input WIDER face dataset annotation file.',
        default=None)

    parser.add_argument(
        '--landmark_image_dir',
        type=str,
        help='Input landmark dataset training image directory.',
        default=None)

    parser.add_argument(
        '--landmark_file_name',
        type=str,
        help='Input landmark dataset annotation file.',
        default=None)

    parser.add_argument(
        '--base_number_of_images',
        type=int,
        help='Input base number of images.',
        default=default_base_number_of_images)

    parser.add_argument(
        '--target_root_dir',
        type=str,
        help=
        'Output directory where output images and TensorFlow data files are saved.',
        default=None)

    return (parser.parse_args(argv))

# reads bounding box training data from WIDER dataset and landmark training data from CelebA dataset.
# converts these datasets to the input format assumed by P-Net.
def main(args):

    if (not args.annotation_image_dir):
        raise ValueError(
            'You must supply input WIDER face dataset training image directory with --annotation_image_dir.'
        )

    if (not args.annotation_file_name):
        raise ValueError(
            'You must supply input WIDER face dataset annotation file with --annotation_file_name.'
        )

    if (not args.landmark_image_dir):
        raise ValueError(
            'You must supply input landmark dataset training image directory with --landmark_image_dir.'
        )

    if (not args.landmark_file_name):
        raise ValueError(
            'You must supply input landmark dataset annotation file with --landmark_file_name.'
        )

    if (not args.target_root_dir):
        raise ValueError(
            'You must supply output directory for storing output images and TensorFlow data files with --target_root_dir.'
        )

    if (args.base_number_of_images < 1):
        base_number_of_images = default_base_number_of_images
    else:
        base_number_of_images = args.base_number_of_images

    simple_dataset = SimpleDataset()
    status = simple_dataset.generate(
        args.annotation_image_dir, args.annotation_file_name,
        args.landmark_image_dir, args.landmark_file_name,
        base_number_of_images, args.target_root_dir)

    if (status):
        print('PNet dataset is generated at ' + args.target_root_dir)
    else:
        print('Error generating PNet dataset.')


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
