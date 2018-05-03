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
import numpy.random as npr
from utils.IoU import IoU

from datasets.WIDERFaceDataset import WIDERFaceDataset

class SimpleFaceDataset(object):

	__positive_IoU = 0.65
	__part_IoU = 0.4
	__negative_IoU = 0.3

	def __init__(self, name='SimpleFaceDataset'):
		self._name = name
		self._clear()

	def _clear(self):
		self._is_valid = False
		self._data = dict()

	@classmethod
	def positive_IoU(cls):
		return(SimpleFaceDataset.__positive_IoU)

	@classmethod
	def minimum_face_size(cls):
		return(40)

	@classmethod
	def part_IoU(cls):
		return(SimpleFaceDataset.__part_IoU)

	@classmethod
	def negative_IoU(cls):
		return(SimpleFaceDataset.__negative_IoU)

	@classmethod
	def positive_file_name(cls, target_root_dir):
		positive_file_name = os.path.join(target_root_dir, 'positive.txt')
		return(positive_file_name)

	@classmethod
	def part_file_name(cls, target_root_dir):
		part_file_name = os.path.join(target_root_dir, 'part.txt')
		return(part_file_name)

	@classmethod
	def negative_file_name(cls, target_root_dir):
		negative_file_name = os.path.join(target_root_dir, 'negative.txt')
		return(negative_file_name)

	def is_valid(self):
		return(self._is_valid)

	def data(self):
		return(self._data)

	def _read(self, annotation_image_dir, annotation_file_name):
		
		self._clear()

		face_dataset = WIDERFaceDataset()
		if(face_dataset.read(annotation_image_dir, annotation_file_name)):
			self._is_valid = True
			self._data = face_dataset.data()		

		return(self._is_valid)

	def generate_samples(self, annotation_image_dir, annotation_file_name, face_size, target_root_dir):

		if(not self._read(annotation_image_dir, annotation_file_name)):
			return(False)

		image_file_names = self._data['images']
		ground_truth_boxes = self._data['bboxes']

		positive_dir = os.path.join(target_root_dir, 'positive')
		part_dir = os.path.join(target_root_dir, 'part')
		negative_dir = os.path.join(target_root_dir, 'negative')

		if(not os.path.exists(positive_dir)):
    			os.makedirs(positive_dir)
		if(not os.path.exists(part_dir)):
    			os.makedirs(part_dir)
		if(not os.path.exists(negative_dir)):
    			os.makedirs(negative_dir)

		positive_file = open(SimpleFaceDataset.positive_file_name(target_root_dir), 'w')
		part_file = open(os.path.join(target_root_dir, 'part.txt'), 'w')
		negative_file = open(os.path.join(target_root_dir, 'negative.txt'), 'w')

		positive_images = 0
		part_images = 0
		negative_images = 0
		annotation_number = 0

    		for image_file_path, ground_truth_box in zip(image_file_names, ground_truth_boxes):
        		bounding_boxes = np.array(ground_truth_box, dtype=np.float32).reshape(-1, 4)			

			current_image = cv2.imread(image_file_path)
    			height, width, channel = current_image.shape

			neg_num = 0
			while(neg_num < 50):
				size = npr.randint(face_size, min(width, height) / 2)
			        nx = npr.randint(0, width - size)
        			ny = npr.randint(0, height - size)
        
        			crop_box = np.array([nx, ny, nx + size, ny + size])
        
        			current_IoU = IoU(crop_box, bounding_boxes)
        
        			cropped_image = current_image[ny : ny + size, nx : nx + size, :]
        			resized_image = cv2.resize(cropped_image, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
				if( np.max(current_IoU) < SimpleFaceDataset.negative_IoU() ):
					file_path = os.path.join(negative_dir, "%s.jpg"%negative_images)
					negative_file.write(file_path + ' 0\n')
					cv2.imwrite(file_path, resized_image)
            				negative_images += 1
            				neg_num += 1

			for bounding_box in bounding_boxes:
				x1, y1, x2, y2 = bounding_box
				w = x2 - x1 + 1
				h = y2 - y1 + 1

				if( ( max(w, h) < SimpleFaceDataset.minimum_face_size() ) or (x1 < 0) or (y1 < 0) or (w < 0) or (h < 0) ):
            				continue

				for i in range(5):
			            	size = npr.randint(face_size, min(width, height) / 2)
            				delta_x = npr.randint(max(-size, -x1), w)
            				delta_y = npr.randint(max(-size, -y1), h)
            				nx1 = int(max(0, x1 + delta_x))
            				ny1 = int(max(0, y1 + delta_y))
            				if ( (nx1 + size) > width ) or ( (ny1 + size) > height ):
                				continue

            				crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            				current_IoU = IoU(crop_box, bounding_boxes)
    
            				cropped_image = current_image[ny1: ny1 + size, nx1: nx1 + size, :]
            				resized_image = cv2.resize(cropped_image, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
    
            				if( np.max(current_IoU) < SimpleFaceDataset.negative_IoU() ):
                				file_path = os.path.join(negative_dir, "%s.jpg" % negative_images)
                				negative_file.write(file_path + ' 0\n')
                				cv2.imwrite(file_path, resized_image)
                				negative_images += 1 

				for i in range(20):
            				size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            				delta_x = npr.randint(-w * 0.2, w * 0.2)
            				delta_y = npr.randint(-h * 0.2, h * 0.2)

            				nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            				ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            				nx2 = nx1 + size
            				ny2 = ny1 + size

            				if nx2 > width or ny2 > height:
                				continue 
            				crop_box = np.array([nx1, ny1, nx2, ny2])
            				offset_x1 = (x1 - nx1) / float(size)
					offset_y1 = (y1 - ny1) / float(size)
            				offset_x2 = (x2 - nx2) / float(size)
            				offset_y2 = (y2 - ny2) / float(size)

            				cropped_image = current_image[ny1 : ny2, nx1 : nx2, :]
            				resized_image = cv2.resize(cropped_image, (face_size, face_size), interpolation=cv2.INTER_LINEAR)

            				box_ = bounding_box.reshape(1, -1)
            				if( IoU(crop_box, box_) >= SimpleFaceDataset.positive_IoU() ):
                				file_path = os.path.join(positive_dir, "%s.jpg"%positive_images)
                				positive_file.write(file_path + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                				cv2.imwrite(file_path, resized_image)
                				positive_images += 1
            				elif( IoU(crop_box, box_) >= SimpleFaceDataset.part_IoU() ):
                				file_path = os.path.join(part_dir, "%s.jpg"%part_images)
                				part_file.write(file_path + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                				cv2.imwrite(file_path, resized_image)
                				part_images += 1

				annotation_number += 1        
				if(annotation_number % 1000 == 0 ):
					print('%s number of images are done - positive - %s,  part - %s, negative - %s' % (annotation_number, positive_images, part_images, negative_images))

		negative_file.close()
		part_file.close()
		positive_file.close()

		return(True)

