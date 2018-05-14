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
from os.path import join, exists
import random
import cv2
import numpy as np
import numpy.random as npr

from utils.BBox import BBox
from utils.IoU import IoU

from datasets.DatasetFactory import DatasetFactory

from datasets.Landmark import rotate
from datasets.Landmark import flip
from datasets.Landmark import randomShift
from datasets.Landmark import randomShiftWithArgument

class LandmarkDataset(object):

	__landmark_ratio = 1.0

	def __init__(self, name='Landmark'):
		self._name = name
		self._clear()

	def _clear(self):
		self._is_valid = False
		self._data = []

	@classmethod
	def landmark_file_name(cls, target_root_dir):
		landmark_file_name = os.path.join(target_root_dir, 'landmark.txt')
		return(landmark_file_name)

	def is_valid(self):
		return(self._is_valid)

	def data(self):
		return(self._data)

	def _read(self, landmark_image_dir, landmark_file_name):
		
		self._clear()
	
		#landmark_dataset = DatasetFactory.landmark_dataset('LFWLandmark')
		landmark_dataset = DatasetFactory.landmark_dataset('CelebADataset')
		if(landmark_dataset.read(landmark_image_dir, landmark_file_name)):
			self._is_valid = True
			self._data = landmark_dataset.data()		

		return(self._is_valid)

	def _can_generate_sample(self):
		return( random.choice([0,1,2,3]) > 1 )		

	def generate(self, landmark_image_dir, landmark_file_name, base_number_of_images, minimum_face, target_root_dir):

		if(not self._read(landmark_image_dir, landmark_file_name)):
			return(False)
		
		image_file_names = self._data['images']
		ground_truth_boxes = self._data['bboxes']
		ground_truth_landmarks = self._data['landmarks']

		landmark_dir = os.path.join(target_root_dir, 'landmark')
		if(not os.path.exists(landmark_dir)):
    			os.makedirs(landmark_dir)

		landmark_file = open(LandmarkDataset.landmark_file_name(target_root_dir), 'w')

		size = minimum_face

		generated_landmark_images = 0
    		processed_input_images = 0
		total_number_of_input_images = len(image_file_names)

		needed_landmark_images = int( ( 1.0 * base_number_of_images * LandmarkDataset.__landmark_ratio ) / total_number_of_input_images )
		needed_landmark_images = max(1, needed_landmark_images)
		base_number_of_attempts = 200
		maximum_attempts = base_number_of_attempts * needed_landmark_images

		for image_path, ground_truth_bounding_box, ground_truth_landmark in zip(image_file_names, ground_truth_boxes, ground_truth_landmarks):

        		current_face_images = []
        		current_face_landmarks = []  

			image_path = image_path.replace("\\", '/')
        		image = cv2.imread(image_path)
			if( image is None):
				continue     
   		
        		image_height, image_width, image_channels = image.shape
        		gt_box = np.array([ground_truth_bounding_box.left, ground_truth_bounding_box.top, ground_truth_bounding_box.right, ground_truth_bounding_box.bottom])
        		f_face = image[ground_truth_bounding_box.top:ground_truth_bounding_box.bottom+1, ground_truth_bounding_box.left:ground_truth_bounding_box.right+1]
        		f_face = cv2.resize(f_face,(size,size))
        		landmark = np.zeros((5, 2))

        		for index, one in enumerate(ground_truth_landmark):
            			landmark_point = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            			landmark[index] = landmark_point

        		current_face_images.append(f_face)
        		current_face_landmarks.append(landmark.reshape(10))
        		landmark = np.zeros((5, 2))  

			current_landmark_images = 0
			number_of_attempts = 0
			while(	(current_landmark_images < needed_landmark_images) and (number_of_attempts < maximum_attempts) ):

				number_of_attempts += 1

            			x1, y1, x2, y2 = gt_box
            			ground_truth_width = x2 - x1 + 1
            			ground_truth_height = y2 - y1 + 1        
            			if (x1 < 0 )or (y1 < 0):
                			continue

               			bounding_box_size = npr.randint(int(min(ground_truth_width, ground_truth_height) * 0.8), np.ceil(1.25 * max(ground_truth_width, ground_truth_height)))
               			delta_x = npr.randint(-ground_truth_width, ground_truth_width) * 0.2
               			delta_y = npr.randint(-ground_truth_height, ground_truth_height) * 0.2
               			nx1 = int(max(x1+ground_truth_width/2-bounding_box_size/2+delta_x, 0))
               			ny1 = int(max(y1+ground_truth_height/2-bounding_box_size/2+delta_y, 0))
                
               			nx2 = nx1 + int(bounding_box_size)
               			ny2 = ny1 + int(bounding_box_size)
               			if( ( nx2 > image_width ) or ( ny2 > image_height ) ):
               				continue

               			crop_box = np.array([nx1,ny1,nx2,ny2])
               			cropped_im = image[ny1:ny2+1,nx1:nx2+1,:]
               			resized_im = cv2.resize(cropped_im, (size, size))

               			current_IoU = IoU(crop_box, np.expand_dims(gt_box,0))

				if( current_IoU >= DatasetFactory.positive_IoU() ):               				

               				for index, one in enumerate(ground_truth_landmark):
                       				landmark_point = ((one[0]-nx1)/bounding_box_size, (one[1]-ny1)/bounding_box_size)
                       				landmark[index] = landmark_point

					current_face_images.append(resized_im)
               				current_face_landmarks.append(landmark.reshape(10))
               				landmark = np.zeros((5, 2))
               				landmark_ = current_face_landmarks[-1].reshape(-1,2)
               				bounding_box = BBox([nx1,ny1,nx2,ny2])   

               				#mirror                    
               				if( self._can_generate_sample() ):
                      				face_flipped, landmark_flipped = flip(resized_im, landmark_)
                       				face_flipped = cv2.resize(face_flipped, (size, size))
                       				#c*h*w
                       				current_face_images.append(face_flipped)
                       				current_face_landmarks.append(landmark_flipped.reshape(10))

               				#rotate
               				if( self._can_generate_sample() ):
                       				face_rotated_by_alpha, landmark_rotated = rotate(image, bounding_box, bounding_box.reprojectLandmark(landmark_), 5)
                       				#landmark_offset
                       				landmark_rotated = bounding_box.projectLandmark(landmark_rotated)
                       				face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                       				current_face_images.append(face_rotated_by_alpha)
                       				current_face_landmarks.append(landmark_rotated.reshape(10))
                
                       				#flip
                       				face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                       				face_flipped = cv2.resize(face_flipped, (size, size))
                       				current_face_images.append(face_flipped)
               					current_face_landmarks.append(landmark_flipped.reshape(10))       							
                    
               				#inverse clockwise rotation
               				if( self._can_generate_sample() ):
                      				face_rotated_by_alpha, landmark_rotated = rotate(image, bounding_box, bounding_box.reprojectLandmark(landmark_), -5)
                       				landmark_rotated = bounding_box.projectLandmark(landmark_rotated)
                       				face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                       				current_face_images.append(face_rotated_by_alpha)
                       				current_face_landmarks.append(landmark_rotated.reshape(10))
                
                       				face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                       				face_flipped = cv2.resize(face_flipped, (size, size))
                       				current_face_images.append(face_flipped)
                       				current_face_landmarks.append(landmark_flipped.reshape(10)) 

				current_image_array, current_landmark_array = np.asarray(current_face_images), np.asarray(current_face_landmarks)

            			for i in range(len(current_image_array)):
                			if np.sum(np.where(current_landmark_array[i] <= 0, 1, 0)) > 0:
                    				continue

                			if np.sum(np.where(current_landmark_array[i] >= 1, 1, 0)) > 0:
                    				continue

					if(current_landmark_images < needed_landmark_images):
                				cv2.imwrite(join(landmark_dir,"%d.jpg" %(generated_landmark_images)), current_image_array[i])
                				landmarks = map(str, list(current_landmark_array[i]))
                				landmark_file.write(join(landmark_dir,"%d.jpg" %(generated_landmark_images))+" -2 "+" ".join(landmarks)+"\n")
                				generated_landmark_images += 1
						current_landmark_images += 1
					else:
						break

       			processed_input_images = processed_input_images + 1
       			if( processed_input_images % 1000 == 0 ):
               			print( '( %s / %s ) number of input images are processed.' % ( processed_input_images, total_number_of_input_images) )

    		landmark_file.close()

		return(True)

