import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision
vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features
conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-11])
k = 10
m = 200
conv6 = torch.nn.Conv2d(512, k*m, kernel_size=1, stride=1, padding=0)
pool6 = torch.nn.MaxPool2d((56, 56), stride=(56, 56), return_indices = True)
conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-11:])

class DFL_VGG(nn.Module):
	def __init__(self, k=10, nclass=200):
		super(DFL_VGG, self).__init__()
		# Feature extraction
		self.conv1_conv4 = conv1_conv4

		# G-Stream
		self.conv5 = conv5
		self.cls5 = nn.Sequential(
			nn.Conv2d(512, 200, kernel_size=1, stride = 1, padding = 0),
			nn.BatchNorm2d(200),
			nn.ReLU(True),
			nn.AdaptiveAvgPool2d((1,1)),
			)

		# P-Stream
		self.conv6 = conv6
		self.pool6 = pool6
		self.cls6 = nn.Linear(200*10, 200)

		# Side-branch
		self.crosspool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)


	def forward(self, x):
		# Stem: Feature extraction
		inter4 = self.conv1_conv4(x)

        # G-stream
		x = self.conv5(inter4)
		out1 = self.cls5(x)
		out1 = out1.view(out1.size(0), -1)

		# P-stream
		# indices is for visualization
		x = self.conv6(inter4)
		x, indices = self.pool6(x)
		inter6 = x
		x = x.view(x.size(0), -1)
		out2 = self.cls6(x)
		
		# Side-branch
		N1 = inter6.size(0)
		inter6 = inter6.view(N1, -1, 200*10)
		x = self.crosspool(inter6)
		out3 = x.view(N1, -1)
	
		return out1, out2, out3, indices



