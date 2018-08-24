from PIL import Image, ImageOps
from torchvision import datasets, transforms, utils

def scale_width_keep_ar(img, target_width):
	ow, oh = img.size
	target_height = int(target_width * oh // ow)
	return img.resize((target_width, target_height), Image.BICUBIC)

def get_transform():
    transform_list = []
    
    #transform_list.append(transforms.Lambda(lambda img:scale_width_keep_ar(img, 448)))
    
    transform_list.append(transforms.Resize(448))
    #transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    #transform_list.append(transforms.RandomRotation((-10, 10)))
    
    #transform_list.append(transforms.ColorJitter(0.5, 0.5, 0.5, 0.25))
    
    transform_list.append(transforms.CenterCrop((448, 448)))
    
    # gray
    #transform_list.append(transforms.RandomGrayscale(1))
    
    transform_list.append(transforms.ToTensor())
    
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    
    return transforms.Compose(transform_list)





##########################################################################################################################
"""
def compose_transform_on_PIL():
	transform_list = []
	transform_list.append(transforms.Lambda(lambda img:scale_width_keep_ar(img, opt.scale_width_keep_ar)))
	transform_list.append(transforms.CenterCrop(opt.random_crop))
	transform_list.append(transforms.RandomRotation((-10, 10)))
	transform_list.append(transforms.RandomGrayscale(1))
	return transforms.Compose(transform_list)

def compose_transform_on_PIL_to_tuple():
	transform_list = []
	transform_list.append(transforms.FiveCrop(48, 256))
	return transforms.Compose(transform_list)
	
def transform_on_Tuple_to_Tensor(x_t):
	transform_list =[]
	assert isinstance(x_t, tuple), 'x_t is not a tuple, we may not ues five crop'
	transform_list.append(transforms.Lambda(lambda x_t:torch.stack[(ToTensor()(x) for x in x_t)]))
	transform = transform.Compose(transform_list)
	return transform(x_t)

def create_testdata_withfivecrops(path):
	transform_PIL = compose_transform_on_PIL()
	transform_PIL2tuple = compose_transform_on_PIL_to_tuple()
	transform_norm = transforms.Compose([transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
	img = Image.open(path)
	img_PIL = transform_PIL(img)
	img_tuple = transform_PIL2tuple(img_PIL)
	img_tensor = transform_on_Tuple_to_Tensor(img_tuple)
	l = []
	for i in img_tensor.size(0):
		img = img_tensor(i)
		l.append(transform_norm(img))
	torch.stack(l)




def transform_PIL
def transform_PIL_(img):
	assert len(img.size) == 3 and img.size(0) == 3, 'transform_PIL input is not PIL'
	c, h, w =  img.size
"""





