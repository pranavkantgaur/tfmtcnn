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

from tfmtcnn.utils.IoU import IoU

import tfmtcnn.datasets.constants as datasets_constants
from tfmtcnn.datasets.DatasetFactory import DatasetFactory

class SimpleFaceDataset(object):

	__positive_ratio = datasets_constants.positive_ratio
	__part_ratio = datasets_constants.part_ratio
	__negative_ratio = datasets_constants.negative_ratio

	def __init__(self, name='SimpleFaceDataset'):
		self._name = name

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

	def _read(self, annotation_image_dir, annotation_file_name, face_dataset_name='WIDERFaceDataset'):
		
		dataset = None
		status = False
		face_dataset = DatasetFactory.face_dataset(face_dataset_name)
		if(face_dataset.read(annotation_image_dir, annotation_file_name)):			
			dataset = face_dataset.data()
			status = True

		return(status, dataset)

	def _generated_samples(self, target_root_dir):
		positive_file = open(SimpleFaceDataset.positive_file_name(target_root_dir), 'a+')
		generated_positive_samples = len(positive_file.readlines())

		part_file = open(SimpleFaceDataset.part_file_name(target_root_dir), 'a+')
		generated_part_samples = len(part_file.readlines())

		negative_file = open(SimpleFaceDataset.negative_file_name(target_root_dir), 'a+')
		generated_negative_samples = len(negative_file.readlines())

		negative_file.close()
		part_file.close()
		positive_file.close()

		return( generated_positive_samples, generated_part_samples, generated_negative_samples)

	def generate_samples(self, annotation_image_dir, annotation_file_name, base_number_of_images, target_face_size, target_root_dir):

		status, dataset = self._read(annotation_image_dir, annotation_file_name)
		if(not status):
			return(False)

		image_file_names = dataset['images']
		ground_truth_boxes = dataset['bboxes']
		number_of_faces = dataset['number_of_faces']

		positive_dir = os.path.join(target_root_dir, 'positive')
		part_dir = os.path.join(target_root_dir, 'part')
		negative_dir = os.path.join(target_root_dir, 'negative')

		if(not os.path.exists(positive_dir)):
    			os.makedirs(positive_dir)
		if(not os.path.exists(part_dir)):
    			os.makedirs(part_dir)
		if(not os.path.exists(negative_dir)):
    			os.makedirs(negative_dir)

		generated_positive_samples, generated_part_samples, generated_negative_samples = self._generated_samples(target_root_dir)
		#print('previous_positive_samples',generated_positive_samples, 'previous_part_samples',generated_part_samples, 'previous_negative_samples',  generated_negative_samples)

		positive_file = open(SimpleFaceDataset.positive_file_name(target_root_dir), 'a+')
		part_file = open(SimpleFaceDataset.part_file_name(target_root_dir), 'a+')
		negative_file = open(SimpleFaceDataset.negative_file_name(target_root_dir), 'a+')

		current_image_number = 0		
		
		negative_samples_per_image_ratio = (SimpleFaceDataset.__negative_ratio - 1)
		needed_negative_samples = base_number_of_images - ( generated_negative_samples / (1.0 * SimpleFaceDataset.__negative_ratio) )
		needed_base_negative_samples = ( 1.0 * needed_negative_samples ) / number_of_faces

		needed_negative_samples_per_image = int( 1.0 * negative_samples_per_image_ratio * needed_base_negative_samples * ( 1.0 * number_of_faces / len(image_file_names) ) )
		needed_negative_samples_per_image = max(0, needed_negative_samples_per_image)

		needed_negative_samples_per_bounding_box = np.ceil(1.0 * (SimpleFaceDataset.__negative_ratio - negative_samples_per_image_ratio) * needed_base_negative_samples )
		needed_negative_samples_per_bounding_box = max(0, needed_negative_samples_per_bounding_box)

		needed_positive_samples = ( base_number_of_images * SimpleFaceDataset.__positive_ratio ) - generated_positive_samples
		needed_positive_samples_per_bounding_box = np.ceil( 1.0 * needed_positive_samples / number_of_faces )
		needed_positive_samples_per_bounding_box = max(0, needed_positive_samples_per_bounding_box)

		needed_part_samples = ( base_number_of_images * SimpleFaceDataset.__part_ratio ) - generated_part_samples
		needed_part_samples_per_bounding_box = np.ceil( 1.0 *  needed_part_samples / number_of_faces )
		needed_part_samples_per_bounding_box = max(0, needed_part_samples_per_bounding_box)

		base_number_of_attempts = 5000

    		for image_file_path, ground_truth_box in zip(image_file_names, ground_truth_boxes):
        		bounding_boxes = np.array(ground_truth_box, dtype=np.float32).reshape(-1, 4)			

			current_image = cv2.imread(image_file_path)
    			input_image_height, input_image_width, input_image_channels = current_image.shape

			needed_negative_samples = needed_negative_samples_per_image

			negative_images = 0
			maximum_attempts = base_number_of_attempts * needed_negative_samples

			number_of_attempts = 0
			while(	(negative_images < needed_negative_samples) and (number_of_attempts < maximum_attempts) ):
				number_of_attempts += 1

				crop_box_size = npr.randint(target_face_size, min(input_image_width, input_image_height)/2 )
			        nx = npr.randint(0, (input_image_width - crop_box_size) )
        			ny = npr.randint(0, (input_image_height - crop_box_size) )
        
        			crop_box = np.array([nx, ny, nx + crop_box_size, ny + crop_box_size])        
        			current_IoU = IoU(crop_box, bounding_boxes)        			
        			
				if( np.max(current_IoU) < DatasetFactory.negative_IoU() ):
					cropped_image = current_image[ny : ny + crop_box_size, nx : nx + crop_box_size, :]
					resized_image = cv2.resize(cropped_image, (target_face_size, target_face_size), interpolation=cv2.INTER_LINEAR)

					file_path = os.path.join(negative_dir, "simple-negative-%s.jpg"%generated_negative_samples)
					negative_file.write(file_path + ' 0' + os.linesep)
					cv2.imwrite(file_path, resized_image)

            				generated_negative_samples += 1
            				negative_images += 1					

			needed_negative_samples = needed_negative_samples_per_bounding_box
			needed_positive_samples = needed_positive_samples_per_bounding_box
			needed_part_samples = needed_part_samples_per_bounding_box

			for bounding_box in bounding_boxes:

				x1, y1, x2, y2 = bounding_box
				bounding_box_width = x2 - x1 + 1
				bounding_box_height = y2 - y1 + 1

				if( (x1 < 0) or (y1 < 0) ):				
            				continue				

				negative_images = 0
				maximum_attempts = base_number_of_attempts * needed_negative_samples
				number_of_attempts = 0
				while( (negative_images < needed_negative_samples) and (number_of_attempts < maximum_attempts) ):

					number_of_attempts += 1

			            	crop_box_size = npr.randint(target_face_size, min(input_image_width, input_image_height)/2 )

            				#delta_x = npr.randint(max(-crop_box_size, -x1), bounding_box_width)
            				#delta_y = npr.randint(max(-crop_box_size, -y1), bounding_box_height)

					delta_x = npr.randint(-1 * crop_box_size, +1 * crop_box_size + 1) * 0.2
					delta_y = npr.randint(-1 * crop_box_size, +1 * crop_box_size + 1) * 0.2

            				nx1 = int(max(0, x1 + delta_x))
            				ny1 = int(max(0, y1 + delta_y))
            				if( ( (nx1 + crop_box_size) > input_image_width ) or ( (ny1 + crop_box_size) > input_image_height ) ):
                				continue

            				crop_box = np.array([nx1, ny1, nx1 + crop_box_size, ny1 + crop_box_size])
            				current_IoU = IoU(crop_box, bounding_boxes) 				
            				    
            				if( np.max(current_IoU) < DatasetFactory.negative_IoU() ):
						cropped_image = current_image[ny1: ny1 + crop_box_size, nx1: nx1 + crop_box_size, :]
						resized_image = cv2.resize(cropped_image, (target_face_size, target_face_size), interpolation=cv2.INTER_LINEAR)

                				file_path = os.path.join(negative_dir, "simple-negative-%s.jpg" % generated_negative_samples)
                				negative_file.write(file_path + ' 0' + os.linesep)
                				cv2.imwrite(file_path, resized_image)

                				generated_negative_samples += 1 
						negative_images += 1
				
				positive_images = 0							
				part_images = 0

				maximum_attempts = base_number_of_attempts * (needed_positive_samples + needed_part_samples)
				number_of_attempts = 0
				while( (number_of_attempts < maximum_attempts) and ( (positive_images < needed_positive_samples) or (part_images < needed_part_samples) ) ):

					number_of_attempts += 1

            				crop_box_size = npr.randint(int(min(bounding_box_width, bounding_box_height) * 0.8), np.ceil(1.25 * max(bounding_box_width, bounding_box_height)))
            				delta_x = npr.randint(-1.0 * bounding_box_width, +1.0 * bounding_box_width + 1) * 0.2			
            				delta_y = npr.randint(-1.0 * bounding_box_height, +1.0 * bounding_box_height + 1) * 0.2

            				nx1 = int(max( (x1 + bounding_box_width/2.0 + delta_x - crop_box_size/2.0), 0))
            				ny1 = int(max( (y1 + bounding_box_height/2.0 + delta_y - crop_box_size/2.0), 0))
            				nx2 = nx1 + crop_box_size
            				ny2 = ny1 + crop_box_size

            				if( (nx2 > input_image_width) or (ny2 > input_image_height) ):
                				continue 

            				crop_box = np.array([nx1, ny1, nx2, ny2])
            				offset_x1 = (x1 - nx1) / float(crop_box_size)
					offset_y1 = (y1 - ny1) / float(crop_box_size)
            				offset_x2 = (x2 - nx2) / float(crop_box_size)
            				offset_y2 = (y2 - ny2) / float(crop_box_size)

            				cropped_image = current_image[ny1 : ny2, nx1 : nx2, :]
            				resized_image = cv2.resize(cropped_image, (target_face_size, target_face_size), interpolation=cv2.INTER_LINEAR)

            				normalized_box = bounding_box.reshape(1, -1)
            				if( ( IoU(crop_box, normalized_box) >= DatasetFactory.positive_IoU() ) and (positive_images < needed_positive_samples) ):
                				file_path = os.path.join(positive_dir, "simple-positive-%s.jpg"%generated_positive_samples)
                				positive_file.write(file_path + ' 1 %.2f %.2f %.2f %.2f'%(offset_x1, offset_y1, offset_x2, offset_y2) + os.linesep)
                				cv2.imwrite(file_path, resized_image)
                				generated_positive_samples += 1
						positive_images += 1

            				elif( ( IoU(crop_box, normalized_box) >= DatasetFactory.part_IoU() ) and (part_images < needed_part_samples) ):
                				file_path = os.path.join(part_dir, "simple-part-%s.jpg"%generated_part_samples)
                				part_file.write(file_path + ' -1 %.2f %.2f %.2f %.2f'%(offset_x1, offset_y1, offset_x2, offset_y2) + os.linesep)
                				cv2.imwrite(file_path, resized_image)
                				generated_part_samples += 1
						part_images += 1

        			current_image_number += 1        
        			if(current_image_number % 1000 == 0 ):
					print('(%s/%s) number of images are done - positive - %s,  part - %s, negative - %s' % (current_image_number, len(image_file_names), generated_positive_samples, generated_part_samples, generated_negative_samples))

		negative_file.close()
		part_file.close()
		positive_file.close()

		return(True)


