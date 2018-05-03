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
		self._clear()

	def _clear(self):
		self._is_valid = False
		self._data = dict()

	def is_valid(self):
		return(self._is_valid)

	def data(self):
		return(self._data)

	def read(self, landmark_image_dir, landmark_file_name):
		self._clear()

		if(not os.path.isfile(landmark_file_name)):
			return(False)

		images = []
		bounding_boxes = []
		landmarks = []
		landmark_file = open(landmark_file_name, 'r')
		while( True ):
       			line = landmark_file.readline().strip()
			landmark_data = line.split(' ')

			image_path = landmark_data[0]
       			if( not image_path):
				break
			else:
				image_path = os.path.join(landmark_image_dir, landmark_data[0])

			image_left = float(landmark_data[1])
			image_top = float(landmark_data[2])
			image_with = float(landmark_data[3])
			image_height = float(landmark_data[4])
			image_right = image_left + image_with
			image_bottom = image_top  + image_height

       			bounding_box = (image_left, image_top, image_right, image_bottom)        
       			bounding_box = map(int,bounding_box)

       			landmark = np.zeros((5, 2))
       			for index in range(0, 5):
       				point = (float(landmark_data[5+2*index]), float(landmark_data[5+2*index+1]))
       				landmark[index] = point

			images.append(image_path)
			bounding_boxes.append(BBox(bounding_box))
			landmarks.append(landmark)

		if(len(images)):			
			self._data['images'] = images
			self._data['bboxes'] = bounding_boxes
			self._data['landmarks'] = landmarks
			self._is_valid = True
		else:
			self._clear()

		return(self.is_valid())

