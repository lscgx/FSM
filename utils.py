from torchvision import datasets, models, transforms
import torch
import torchsummary
from thop import profile
import logging
import pickle
import os
import sys
import time
import datetime
from pathlib import Path
from torch.utils.data import DataLoader

logger = None
time_format = "%Y-%m-%d %H:%M:%S"

def load_data(data_name = 'CIFAR10', data_dir = './data', batch_size = 128, num_workers = 1,pin_memory = True,scale_size = 224):
	if data_name == 'CIFAR10':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		trainset = datasets.CIFAR10(root=data_dir, train=True,download=True, transform=transform_train)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

		testset = datasets.CIFAR10(root=data_dir, train=False,download=True, transform=transform_test)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=num_workers)
		return trainloader,testloader
	elif data_name == 'ImageNet':
		traindir = os.path.join(data_dir, 'ILSVRC2012_img_train')
		valdir   = os.path.join(data_dir, 'ILSVRC2012_img_val')
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		trainset = datasets.ImageFolder(
			traindir,
			transforms.Compose([
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.Resize(scale_size),
				transforms.ToTensor(),
				normalize,
			]))

		train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)

		testset = datasets.ImageFolder(
			valdir,
			transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.Resize(scale_size),
				transforms.ToTensor(),
				normalize,
			]))
		test_loader = DataLoader(
			testset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_memory)
		return train_loader,test_loader

def model_size(model,input_size,device):
	model.to(device)
	input_image = torch.randn(1, 3, input_size, input_size).to(device)
	flops, params = profile(model, inputs=(input_image,))
	return flops,params

def get_logger(filename, verbosity=1, name=None):
	global logger 
	level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
	formatter = logging.Formatter(
		"[%(asctime)s] %(message)s",time_format
	)
	logger = logging.getLogger(name)
	logger.setLevel(level_dict[verbosity])

	sh = logging.StreamHandler()
	sh.setFormatter(formatter)
	logger.addHandler(sh)

	fh = logging.FileHandler(filename, "a")
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def init_logger(verbosity=1, name=None):
	global logger 
	filename = 'log/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	# if not os.path.isdir(filename):
	# 	os.makedirs(filename)
	logger = get_logger(filename, verbosity=1, name=None)
	return logger

def progress_bar(msg=None):
	time = datetime.datetime.now().strftime(time_format)
	sys.stdout.write('[' + time + '] ')
	sys.stdout.write(msg + '\n')
	sys.stdout.flush()

def format_compress_rate(compress_rate):
	if compress_rate is None : return compress_rate
	import re
	cprate_str=compress_rate
	cprate_str_list=cprate_str.split('+')
	pat_cprate = re.compile(r'\d+\.\d*')
	pat_num = re.compile(r'\*\d+')
	cprate=[]
	for x in cprate_str_list:
		num=1
		find_num=re.findall(pat_num,x)
		if find_num:
			assert len(find_num) == 1
			num=int(find_num[0].replace('*',''))
		find_cprate = re.findall(pat_cprate, x)
		assert len(find_cprate)>=1
		cprate+=[float(x) for x in find_cprate]*num
		
	return cprate

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

