from PIL import Image, ImageOps
from torchvision import datasets, transforms, utils
import torch

def scale_width_keep_ar(img, target_width):
	"""
	resize image keeping aspect ratio
	"""
	ow, oh = img.size
	
	target_height = int(target_width * oh // ow)
	
	return img.resize((target_width, target_height), Image.BICUBIC)

def scale_keep_ar_min_fixed(img, fixed_min):
    ow, oh = img.size

    if ow < oh:
        
        nw = fixed_min

        nh = nw * oh // ow
    
    else:
        
        nh = fixed_min 

        nw = nh * ow // oh
    return img.resize((nw, nh), Image.BICUBIC)

def get_transform_for_train():

    transform_list = []
    
    transform_list.append(transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, 448)))
    
    transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
    
    transform_list.append(transforms.RandomCrop((448, 448)))
    
    transform_list.append(transforms.ToTensor())
    
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    
    return transforms.Compose(transform_list)

def get_transform_for_test():

    transform_list = []
    
    transform_list.append(transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, 560)))
   
    transform_list.append(transforms.TenCrop(448)) 
    
    transform_list.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))((transforms.ToTensor())(crop)) for crop in crops])) )
    
    #transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    
    return transforms.Compose(transform_list)

def get_transform_for_test_simple():

    transform_list = []
    
    transform_list.append(transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, 448)))
    
    transform_list.append(transforms.CenterCrop((448, 448)))
    
    transform_list.append(transforms.ToTensor())
    
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    
    return transforms.Compose(transform_list)











