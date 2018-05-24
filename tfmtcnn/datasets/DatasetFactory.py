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

import datasets.constants as datasets_constants

from datasets.WIDERFaceDataset import WIDERFaceDataset
from datasets.CelebADataset import CelebADataset
from datasets.LFWLandmarkDataset import LFWLandmarkDataset

class DatasetFactory(object):

	__positive_IoU = datasets_constants.positive_IoU
	__part_IoU = datasets_constants.part_IoU
	__negative_IoU = datasets_constants.negative_IoU

	def __init__(self):
		pass

	@classmethod
	def positive_IoU(cls):
		return(DatasetFactory.__positive_IoU)

	@classmethod
	def part_IoU(cls):
		return(DatasetFactory.__part_IoU)

	@classmethod
	def negative_IoU(cls):
		return(DatasetFactory.__negative_IoU)

	@classmethod
	def face_dataset(cls, name):
		if( name == CelebADataset.name() ):
			return(CelebADataset())
		elif ( name == WIDERFaceDataset.name() ):
			return(WIDERFaceDataset())	

	@classmethod
	def landmark_dataset(cls, name):
		if( name == CelebADataset.name() ):
			return(CelebADataset())
		elif ( name == LFWLandmarkDataset.name() ):
			return(LFWLandmarkDataset())
		





