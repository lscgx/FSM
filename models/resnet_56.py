import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import os
import numpy as np

norm_mean, norm_var = 0.0, 1.0

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)

class ResBasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, compress_rate=[0.]):
		super(ResBasicBlock, self).__init__()
		self.stride = stride
		self.inplanes = inplanes
		self.midplanes = math.ceil(planes * (1-compress_rate[0]))
		self.planes = math.ceil(planes * (1-compress_rate[1]))

		self.conv1 = conv3x3(inplanes, self.midplanes , stride)
		self.bn1 = nn.BatchNorm2d(self.midplanes)
		self.relu1 = nn.ReLU(inplace=True)

		self.conv2 = conv3x3(self.midplanes, self.planes)
		self.bn2 = nn.BatchNorm2d(self.planes )
		self.relu2 = nn.ReLU(inplace=True)
		self.stride = stride
		self.shortcut = nn.Sequential()

		self.last_block_rank = None

		if stride != 1 or self.inplanes != self.planes:
			if stride != 1:
				self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (self.planes-self.inplanes)//2 , self.planes-self.inplanes-(self.planes-self.inplanes)//2), "constant", 0))
			else :
				self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, :, :], (0, 0, 0, 0, (self.planes-self.inplanes)//2 , self.planes-self.inplanes-(self.planes-self.inplanes)//2), "constant", 0))

	def forward(self, x):

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.last_block_rank is None:
			out += self.shortcut(x)
		else :
			for i,j in enumerate(self.last_block_rank):
				out[:,j,:,:] += x[:,i,:,:]

		out = self.relu2(out)
		return out

	def set_last_block_rank(self,x):
		print('set_last_block_rank...')
		self.last_block_rank = x

class ResNet(nn.Module):
	def __init__(self, block, num_layers, compress_rate, num_classes=10):
		super(ResNet, self).__init__()
		assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
		n = (num_layers - 2) // 6
		self.compress_rate = compress_rate
		self.num_layers = num_layers

		self.inplanes = math.ceil(16 * (1-compress_rate[0]))
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

		self.bn1 = nn.BatchNorm2d(self.inplanes)
		self.relu = nn.ReLU(inplace=True)

		self.layer1 = self._make_layer(block, 16, blocks=n, stride=1,compress_rate=compress_rate[1:2 * n + 1])
		self.layer2 = self._make_layer(block, 32, blocks=n, stride=2,compress_rate=compress_rate[2 * n + 1:4 * n + 1])
		self.layer3 = self._make_layer(block, 64, blocks=n, stride=2,compress_rate=compress_rate[4 * n + 1:6 * n + 1])
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))

		if num_layers == 110:
			self.linear = nn.Linear(math.ceil(64 * (1-compress_rate[-1])), num_classes)
		else:
			self.fc = nn.Linear(math.ceil(64 * (1-compress_rate[-1])) * block.expansion, num_classes)

	def _make_layer(self, block, planes, blocks, stride, compress_rate):
		layers = []

		layers.append(block(self.inplanes, planes, stride, compress_rate=compress_rate[0:2]))

		self.inplanes = math.ceil(planes * block.expansion * (1 - compress_rate[1]))
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, compress_rate=compress_rate[2 * i:2 * i + 2]))
			self.inplanes = math.ceil(planes * (1 - compress_rate[2 * i + 1]))

		return nn.Sequential(*layers)

	def forward(self, x):

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)

		if self.num_layers == 110:
			x = self.linear(x)
		else:
			x = self.fc(x)

		return x

def resnet_56(compress_rate=None,oristate_dict = None,ranks = None):

	model = None 
	if oristate_dict is not None and ranks is not None and compress_rate is not None:
		if len(compress_rate) < 55:
			compress_rate = compress_rate + [0.] * (55 - len(compress_rate))
		if len(ranks) < 55:
			ranks = ranks + ['no_pruned'] * (55 - len(ranks))

		model = ResNet(ResBasicBlock, 56, compress_rate=compress_rate)
		print(model)
		state_dict = model.state_dict()
		cov_id = 0
		N = 55
		rank = None
		last_select_index = None

		for k,name in enumerate(oristate_dict):
			# print(k,name,rank)
			if k < 55 * 6:
				if k%6 == 0:
					rank = ranks[cov_id]
					if rank == 'no_pruned':
						rank = list(range(len(state_dict[name])))
					if last_select_index is not None:
						for _i,i in enumerate(rank):
							for _j,j in enumerate(last_select_index):
								state_dict[name][_i][_j] = oristate_dict[name][i][j]
					else :
						for _i,i in enumerate(rank):
							state_dict[name][_i] = oristate_dict[name][i]
					last_select_index = rank
					cov_id += 1
				elif k%6 == 5:
					state_dict[name] = oristate_dict[name]
				else:
					for _i,i in enumerate(rank):
						state_dict[name][_i] = oristate_dict[name][i]
			elif k == 55 * 6:
				rank = list(range(10))
				for _i,i in enumerate(rank):
					for _j,j in enumerate(last_select_index):
						state_dict[name][_i][_j] = oristate_dict[name][i][j]
			elif k == 55 * 6 + 1:
				state_dict[name] = oristate_dict[name]
		model.load_state_dict(state_dict)
	elif  compress_rate is not None:
		model = ResNet(ResBasicBlock, 56, compress_rate=compress_rate)
	else :
		compress_rate = [0] * 56
		model = ResNet(ResBasicBlock, 56, compress_rate=compress_rate)
	return model 
