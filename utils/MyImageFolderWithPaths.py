import torch
import os
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
	def __getitem__(self, index):
		original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
		path = self.imgs[index][0]
		tuple_with_path = (original_tuple + (path,)) 
		return tuple_with_path
	
	def index2classlist(self):
		return self._find_classes_(self.root)

	def _find_classes_(self, dir):
		# return -- list : index of list coresponding to classname
		classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
		classes.sort()
		#class_to_idx = {classes[i]: i for i in range(len(classes))}
		#return classes, class_to_idx
		return classes