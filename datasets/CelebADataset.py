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

from utils.BBox import BBox

class CelebADataset(object):

	def __init__(self, name='CelebA'):
		self._name = name
		self._is_valid = False
		self._landmark_data = []

	def is_valid(self):
		return(self._is_valid)

	def data(self):
		return(self._landmark_data)

	def read(self, landmark_image_dir, landmark_file_name):

		self._is_valid = False
    		self._landmark_data = []

		with open(landmark_file_name, 'r') as landmark_file:
        		lines = landmark_file.readlines()

    		for line in lines:
        			line = line.strip()
        			components = line.split(' ')
        			image_path = os.path.join(landmark_image_dir, components[0])

				image_left = float(components[1])
				image_top = float(components[2])
				image_with = float(components[3])
				image_height = float(components[4])
				image_right = image_left + image_with
				image_bottom = image_top  + image_height
        			bounding_box = (image_left, image_top, image_right, image_bottom)        
        			bounding_box = map(int,bounding_box)

        			landmark = np.zeros((5, 2))
        			for index in range(0, 5):
            				rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            				landmark[index] = rv

        			self._landmark_data.append((image_path, BBox(bounding_box), landmark))

		if(len(self._landmark_data)):
			self._is_valid = True

    		return(self._is_valid)

