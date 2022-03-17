import torch.nn as nn
import numpy as np
import torch
import math
import argparse
import torch.optim as optim
import time
import os
import copy
import pickle
import random
import bisect
from PIL import Image
from copy import deepcopy
from scipy.stats import norm
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sympy import *

from utils import *
from models import *
from mask import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 & ImageNet Pruning')

parser.add_argument(
	'--data_dir',    
	default='G:\\data',    
	type=str,   
	metavar='DIR',                 
	help='path to dataset')
parser.add_argument(
	'--dataset',     
	default='CIFAR10',   
	type=str,   
	choices=('CIFAR10','ImageNet'),
	help='dataset')
parser.add_argument(
	'--num_workers', 
	default=0,           
	type=int,   
	metavar='N',                   
	help='number of data loading workers (default: 0)')
parser.add_argument(
    '--epochs',
    type=int,
    default=15,
    help='The num of epochs to train.')
parser.add_argument(
	'--lr',         
	default=0.01,        
	type=float,                                
	help='initial learning rate')
parser.add_argument(
    '--lr_decay_step',
    default='5,10',
    type=str,
    metavar='LR',
    help='learning rate decay step')
parser.add_argument(
    '--resume',
    type=str,
    default=None,
    metavar='PATH',
    help='load the model from the specified checkpoint')
parser.add_argument(
	'--batch_size', 
	default=128, 
	type=int,
    metavar='N',
    help='mini-batch size')
parser.add_argument(
	'--momentum', 
	default=0.9, 
	type=float, 
	metavar='M',
    help='momentum')
parser.add_argument(
	'--weight_decay', 
	default=0., 
	type=float,
    metavar='W', 
    help='weight decay',
    dest='weight_decay')
parser.add_argument(
	'--gpu', 
	default='0', 
	type=str,
    help='GPU id to use.')
parser.add_argument(
    '--job_dir',
    type=str,
    default='',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--input_size',
    type=int,
    default=32,
    help='The num of input size')
parser.add_argument(
    '--save_id',
    type=int,
    default=0,
    help='save_id')
parser.add_argument(
    '--from_scratch',
    type=bool,
    default=False,
    help='train from_scratch')

args           = None
lr_decay_step  = None
logger         = None
compress_rate  = None
trainloader    = None
testloader     = None
criterion      = None
device         = None
model          = None
mask           = None
best_acc       = 0.
best_accs      = []

def init():
	global args,lr_decay_step,logger,compress_rate,trainloader,testloader,criterion,device,model,mask,best_acc,best_accs
	args = parser.parse_args()
	lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
	logger = get_logger(os.path.join(args.job_dir, 'log/log'))
	compress_rate = format_compress_rate(args.compress_rate)
	trainloader,testloader = load_data(data_name = args.dataset, data_dir = args.data_dir, batch_size = args.batch_size, num_workers = args.num_workers)
	criterion = nn.CrossEntropyLoss()
	device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	model = eval(args.arch)()
	mask = eval('mask_'+args.arch)(model=model, job_dir=args.job_dir, device=device)
	best_acc = 0.  # best test accuracy
	if len(args.job_dir) > 0  and args.job_dir[-1] != '\\':
		args.job_dir += '/'
	if len(args.gpu) > 1:
		gpus = args.gpu.split(',')
		device_id = [int(x) for x in gpus]
		print('device_ids:',device_id)
		model = nn.DataParallel(model, device_ids=device_id).cuda()
	else :
		model = model.to(device)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.info('args:{}'.format(args))

def train(epoch,model,cov_id,trainloader,optimizer,criterion,mask = None):
	losses = AverageMeter('Loss', ':.4f')
	top1   = AverageMeter('Acc@1', ':.2f')
	top5   = AverageMeter('Acc@5', ':.2f')

	model.train()
	since  = time.time()
	_since = time.time()
	for i, (inputs,labels) in enumerate(trainloader, 0):
		# if i > 1 : break
		inputs = inputs.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		if mask is not None : mask.grad_mask(cov_id)

		acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(acc1[0], inputs.size(0))
		top5.update(acc5[0], inputs.size(0))

		if i!=0 and i%2000 == 0:
			_end = time.time()
			logger.info('epoch[{}]({}/{}) Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(epoch,i,int(1280000/args.batch_size),losses.avg,top1.avg,top5.avg,_end - _since))
			_since = time.time()

	end = time.time()
	logger.info('train    Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(losses.avg,top1.avg,top5.avg,end - since))

def validate(epoch,model,cov_id,testloader,criterion,save = True):
	losses = AverageMeter('Loss', ':.4f')
	top1   = AverageMeter('Acc@1', ':.2f')
	top5   = AverageMeter('Acc@5', ':.2f')

	model.eval()
	with torch.no_grad():
		since = time.time()
		for i, data in enumerate(testloader, 0):
			# if i > 1 : break
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			loss = criterion(outputs, labels)

			acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
			losses.update(loss.item(), inputs.size(0))
			top1.update(acc1[0], inputs.size(0))
			top5.update(acc5[0], inputs.size(0))

		end = time.time()
		logger.info('validate Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(losses.avg,top1.avg,top5.avg,end - since))

	global best_acc
	if save and best_acc <= top1.avg:
		best_acc = top1.avg
		state = {
			'state_dict': model.state_dict(),
			'best_prec1': best_acc,
			'epoch': epoch,
		}
		if not os.path.isdir(args.job_dir + 'pruned_checkpoint'):
			os.makedirs(args.job_dir + 'pruned_checkpoint')
		cov_name = '_cov' + str(cov_id)
		if cov_id == -1: cov_name = ''
		torch.save(state,args.job_dir + 'pruned_checkpoint/'+args.arch+cov_name + '.pt')
		logger.info('storing checkpoint:'+'pruned_checkpoint/'+args.arch+cov_name + '.pt')

	return top1.avg,top5.avg

def iter_vgg16bn():

	cfg = [0,1,3,4,6,7,8,10,11,12,14,15,16]
	ranks = []
	best_accs = []
	optimizer = None 
	scheduler = None
	conv_names = get_conv_names(model)

	for layer_id in range(0,13):
		logger.info("===> pruning layer {}".format(layer_id))
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

		if layer_id == 0: 
			pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + args.resume)
			model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
		else :
			pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_id-1) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_id-1) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])

		_num = len(model.state_dict()[conv_names[layer_id]])
		if layer_id < 12: _nxt_num = len(model.state_dict()[conv_names[layer_id+1]])
		relu_expect = get_relu_expect(arch=args.arch,model=model,layer_id=layer_id)
		effects = []
		for i in range(_num):
			if layer_id == 12:
				effect = relu_expect[i]
			else:
				effect = get_effect_for_dstr(model,i,layer_id+1,relu_expect[i],type='mean')
			effects.append(effect)
		rank = list(np.argsort(effects))
		rank1 = rank[int(_num*compress_rate[layer_id]):_num] 
		ranks.append(rank1)

		mask.layer_mask(layer_id, param_per_cov=4, rank=rank1, type = 1, arch=args.arch)
		ori_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
		state_dict = model.state_dict()

		ori_rm = []
		ori_rv = []
		if layer_id < 12:
			for i in range(0,_nxt_num):
				expect2 = get_expect_after_relu(model.state_dict()[conv_names[layer_id+1]][i],relu_expect,saved=rank1)
				expect1 = get_expect_after_relu(model.state_dict()[conv_names[layer_id+1]][i],relu_expect,saved=rank)
				expect3 = state_dict['features.norm'+str(cfg[layer_id+1])+'.running_mean'][i]
				est_error = expect1/expect3
				if est_error > 0.:
					inc = expect2*(1./est_error) - state_dict['features.norm'+str(cfg[layer_id+1])+'.running_mean'][i]
					state_dict['features.norm'+str(cfg[layer_id+1])+'.running_mean'][i] += inc
					state_dict['features.norm'+str(cfg[layer_id+1])+'.running_var'][i]  *= (1.-compress_rate[layer_id]) 
					ori_rm.append(inc)
					ori_rv.append((1.-compress_rate[layer_id]))
				else :
					ori_rm.append(0.)
					ori_rv.append(1.)
		aft_acc1,_ = validate(0,model,0,testloader,criterion,save = False)

		if aft_acc1 < ori_acc1:
			logger.info("rollback...")
			for i in range(0,_nxt_num):
				state_dict['features.norm'+str(cfg[layer_id+1])+'.running_mean'][i] -= ori_rm[i]
				state_dict['features.norm'+str(cfg[layer_id+1])+'.running_var'][i]  /= ori_rv[i]
			aft_acc1,_ = validate(0,model,0,testloader,criterion,save = False)

		for epoch in range(0, args.epochs):
			logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
			train(epoch,model,layer_id,trainloader,optimizer,criterion,mask)
			scheduler.step()
			validate(epoch,model,layer_id,testloader,criterion)
		global best_acc 
		best_accs.append(round(best_acc.item(),4)) 
		logger.info("===> layer_id {} bestacc {:.4f}".format(layer_id,best_accs[-1]))
		best_acc=0.

	logger.info(best_accs)

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(12) + '.pt', map_location=device)
	rst_model = vgg_16_bn(_state_dict(final_state_dict['state_dict']),ranks)
	logger.info(rst_model)
	flops,params = model_size(rst_model,args.input_size,device)
	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))
	
def iter_resnet_56():
	ranks = []
	layers = 55
	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	ranks.append('no_pruned')

	for layer_id in range(1,4):
		for block_id in range(0,9):
			logger.info("===> pruning layer_id {} block_id {}".format(layer_id,block_id))

			if layer_id == 1 and block_id == 0: 
				pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
				logger.info('loading checkpoint:' + args.resume)
				model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
			else :
				pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])

			for cov_id in range(1,3):
				if cov_id == 1:
					_id         = (layer_id-1)*18 + block_id * 2 + cov_id 
					_num        = len(model.state_dict()[conv_names[_id]])
					_nxt_num    = len(model.state_dict()[conv_names[_id+1]])
					relu_expect = get_relu_expect(arch='resnet_56', model=model, layer_id=_id)
					effects     = []
					for i in range(_num):
						effect = get_effect_for_dstr(model,i,_id+1,relu_expect[i],type='mean')
						effects.append(effect)
					rank  = list(np.argsort(effects))
					rank1 = rank[int(_num*compress_rate[_id]):_num] 
					ranks.append(rank1)

					mask.layer_mask(_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)
					ori_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
					state_dict = model.state_dict()

					ori_rm = []
					ori_rv = []
					for i in range(0,_nxt_num):
						expect2 = get_expect_after_relu(model.state_dict()[conv_names[_id+1]][i],relu_expect,saved=rank1)
						expect1 = get_expect_after_relu(model.state_dict()[conv_names[_id+1]][i],relu_expect,saved=rank)
						expect3 = state_dict[bn_names[_id+1]+'.running_mean'][i]
						est_error = expect1/expect3
						if est_error > 0.:
							inc = expect2*(1./est_error) - expect3
							state_dict[bn_names[_id+1]+'.running_mean'][i] += inc
							state_dict[bn_names[_id+1]+'.running_var'][i]  *= (1.-compress_rate[_id]) 
							ori_rm.append(inc)
							ori_rv.append((1.-compress_rate[_id]))
						else :
							ori_rm.append(0.)
							ori_rv.append(1.)
					aft_acc1,_ = validate(0,model,0,testloader,criterion,save = False)

					if aft_acc1 < ori_acc1:
						logger.info("rollback...")
						for i in range(0,_nxt_num):
							state_dict[bn_names[_id+1]+'.running_mean'][i] -= ori_rm[i]
							state_dict[bn_names[_id+1]+'.running_var'][i]  /= ori_rv[i]
						aft_acc1,_ = validate(0,model,0,testloader,criterion,save = False)

					_train(_id)
				else :
					ranks.append('no_pruned')


	logger.info(best_accs)
	logger.info([len(x) for x in ranks])

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(53) + '.pt', map_location=device)
	rst_model = resnet_56(compress_rate = compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks=ranks)
	logger.info(rst_model)

	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}

	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_resnet_50():
	ranks = []
	layers = 49
	stage_repeat = [3, 4, 6, 3]
	layer_last_id = [0,10,23,42,52]

	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)
	ranks.append('no_pruned')

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	_id = 0
	for layer_id in range(1,5):
		if layer_id == 1:
			pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + args.resume)
			model.load_state_dict(pruned_checkpoint)
		else :
			pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id-1]) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id-1]) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])
		lid    = layer_last_id[layer_id]
		_num   = len(model.state_dict()[conv_names[lid]])
		rank   = get_rank_resnet_50(model,_num,lid,type='mean')
		l_rank = rank[int(_num*compress_rate[lid-layer_id]):_num] 
		cid    = 0
		for block_id in range(0,stage_repeat[layer_id-1]):
			if block_id == 0:
				mask.layer_mask(layer_last_id[layer_id-1]+3, param_per_cov=3, rank=l_rank, type = 1, arch=args.arch)
				mask.layer_mask(layer_last_id[layer_id-1]+4, param_per_cov=3, rank=l_rank, type = 1, arch=args.arch)
				cid = layer_last_id[layer_id-1]+4
			else :
				cid += 3
				mask.layer_mask(cid, param_per_cov=3, rank=l_rank, type = 1, arch=args.arch)
		acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
		_train(lid)

		for block_id in range(0,stage_repeat[layer_id-1]):
			logger.info("===> pruning layer_id {} block_id {}".format(layer_id,block_id))

			if block_id == 0:
				pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])
			else :
				pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])

			for cov_id in range(1,4):
				_id += 1
				cpid = _id - layer_id
				if block_id == 0 : cpid += 1
				if cov_id < 3:
					_num        = len(model.state_dict()[conv_names[_id]])
					_nxt_num    = len(model.state_dict()[conv_names[_id+1]])
					relu_expect = get_relu_expect_resnet_50(model,_id)
					rank        = get_rank_resnet_50(model,_num,_id,type='mean')
					rank1       = rank[int(_num*compress_rate[cpid]):_num] 
					ranks.append(rank1)
					mask.layer_mask(_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)
					ori_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
					
					state_dict = model.state_dict()

					ori_rm = []
					ori_rv = []
					for i in range(0,_nxt_num):
						expect2 = get_expect_after_relu(model.state_dict()[conv_names[_id+1]][i],relu_expect,saved=rank1)
						expect1 = get_expect_after_relu(model.state_dict()[conv_names[_id+1]][i],relu_expect,saved=rank)
						expect3 = state_dict[bn_names[_id+1]+'.running_mean'][i] 
						est_error = expect1/expect3

						if est_error > 0.5:
							inc = expect2*(1./est_error) - state_dict[bn_names[_id+1]+'.running_mean'][i]
							state_dict[bn_names[_id+1]+'.running_mean'][i] = expect2 * (1./est_error)
							state_dict[bn_names[_id+1]+'.running_var'][i] *= (1 - compress_rate[cpid])
							ori_rm.append(inc)
							ori_rv.append((1.-compress_rate[cpid]))
						else :
							ori_rm.append(0.)
							ori_rv.append(1.)
					aft_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
					if aft_acc1 < ori_acc1:
						logger.info("rollback...")
						for i in range(0,_nxt_num):
							state_dict[bn_names[_id+1]+'.running_mean'][i] -= ori_rm[i]
							state_dict[bn_names[_id+1]+'.running_var'][i]  /= ori_rv[i]
						aft_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
				else :
					ranks.append(l_rank)

				if block_id == 0 and cov_id == 3: 
					_id += 1
					ranks.append(l_rank)

			_train(_id)
			logger.info("===> layer_id {} block_id {} bestacc {:.4f}".format(layer_id,block_id,best_accs[-1]))

	logger.info(best_accs)
	logger.info(compress_rate)
	logger.info([len(x) for x in ranks])

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'_cov52.pt', map_location=device)
	rst_model = resnet_50(compress_rate=compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks = ranks)
	logger.info(rst_model)
	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_googlenet():
	ranks = []
	inceptions = ['a3','b3','a4','b4','c4','d4','e4','a5','b5']

	layer_0_id = [1]
	for i in range(8):
		layer_0_id.append(layer_0_id[-1]+7)
	ffmaps = []
	for x in model.filters:
		ffmaps.append([[x[0],sum(x[:2])],[sum(x[:2]),sum(x[:3])]]) 

	conv_names = get_conv_names(model)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
	relu_expect = get_relu_expect(arch = args.arch, model = model, layer_id = 0)
	#--------------------------get rank-----------------------------
	_num = len(model.state_dict()[conv_names[0]])
	effects = []
	for i in range(_num):
		effect = 0.
		for x in [0,1,3,6]:
			effect += get_effect_for_dstr(model,i,1+x,relu_expect[i],type='mean')
		effect /= 4
		effects.append(effect)
	rank = list(np.argsort(effects))
	rank1 = rank[int(_num*compress_rate[0]):_num]
	ranks.append([rank1])
	#---------------------------------------------------------------
	logger.info("===> pruning pre_layers")
	mask.layer_mask(0, param_per_cov=4, rank=rank1, type = 1, arch=args.arch)
	acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
	_train(0)

	for i,inception_id in enumerate(inceptions):
		i += 1
		logger.info("===> pruning inception_id {}".format(i))

		pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(i-1) + '.pt', map_location=device)
		logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(i-1) + '.pt')
		model.load_state_dict(pruned_checkpoint['state_dict'])
		i_offset = [2,4,5]
		_rank = []
		for j in range(3):
			layer_id = (i-1)*7 + 1 + i_offset[j]
			_num = len(model.state_dict()[conv_names[layer_id]])
			if compress_rate[(i-1)*3+j+1] == 0.:
				_rank.append(list(range(_num)))	 
				continue
			relu_expect = get_relu_expect(arch = args.arch, model = model, layer_id = layer_id)
			
			effects = []
			for k in range(_num):
				if j == 1:
					effect = get_effect_for_dstr(model,k,layer_id+1,relu_expect[k],type='mean')
				else :
					effect = 0.
					for x in [0,1,3,6]:
						effect += get_effect_for_dstr(model,k,layer_0_id[i]+x,relu_expect[k],type='mean',slices=ffmaps[i-1][1 if j > 1 else j])
					effect /= 4
				effects.append(effect)
			rank  = list(np.argsort(effects))
			rank1 = rank[int(_num*compress_rate[(i-1)*3+j+1]):_num]
			_rank.append(rank1)
			mask.layer_mask(layer_id, param_per_cov=4, rank=rank1, type = 1, arch=args.arch)
			acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
		ranks.append(_rank)
		_train(i)
		logger.info("===> inception_id {} best_acc {:.4f}".format(i,best_accs[-1]))

	logger.info(best_accs)
	logger.info(compress_rate)

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'_cov9.pt', map_location=device)
	rst_model =  googlenet(compress_rate = compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks = ranks).to(device)
	flops,params = model_size(rst_model,args.input_size,device)
	logger.info(rst_model)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def train_from_scratch():
	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)

	if args.compress_rate is None:
		compress_rate = pruned_checkpoint['compress_rate']
	else :
		compress_rate = format_compress_rate(args.compress_rate)

	model = eval(args.arch)(compress_rate=compress_rate).to(device)

	logger.info(model)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

	validate(0,model,0,testloader,criterion,save = False)

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	for epoch in range(0, args.epochs):
		if epoch in lr_decay_step and args.arch != 'resnet_50':
			resume = args.job_dir + 'pruned_checkpoint/'+args.arch+'.pt'
			pruned_checkpoint = torch.load(resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + 'pruned_checkpoint/'+args.arch+'.pt')
			model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		train(epoch,model,0,trainloader,optimizer,criterion) #,mask
		scheduler.step()
		validate(epoch,model,-1,testloader,criterion)

	flops,params = model_size(model,args.input_size,device)

	best_model = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'.pt', map_location=device)
	rst_model = eval(args.arch)(compress_rate=compress_rate).to(device)
	rst_model.load_state_dict(_state_dict(best_model['state_dict']))

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': round(best_acc.item(),4),
		'compress_rate': compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def _train(i):
	global best_acc,best_accs
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
	for epoch in range(0, args.epochs):
		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		train(epoch,model,i,trainloader,optimizer,criterion,mask) #,mask
		scheduler.step()
		validate(epoch,model,i,testloader,criterion)
	if args.epochs > 0 and best_acc > 0.:
		best_accs.append(round(best_acc.item(),4))
	else:
		best_accs.append(0.)
	best_acc=0.

def get_conv_names(model = None):
	conv_names = []
	for name, module in model.named_modules():
		if isinstance(module,nn.Conv2d):
			conv_names.append(name+'.weight')
	return conv_names

def get_bn_names(model = None):
	bn_names = []
	for name, module in model.named_modules():
		if isinstance(module,nn.BatchNorm2d):
			bn_names.append(name)
	return bn_names

def _state_dict(state_dict):
	rst = []
	for n, p in state_dict.items():
		if "total_ops" not in n and "total_params" not in n:
			rst.append((n.replace('module.', ''), p))
	rst = dict(rst)
	return rst

def get_relu_expect(arch = 'vgg_16_bn', model = None, layer_id = 0):
	relu_expect = []
	state_dict  = model.state_dict()

	if arch == 'vgg_16_bn':
		cfg         = [0,1,3,4,6,7,8,10,11,12,14,15,16]
		norm_weight = state_dict['features.norm'+str(cfg[layer_id])+'.weight']
		norm_bias   = state_dict['features.norm'+str(cfg[layer_id])+'.bias']
	elif arch == 'resnet_56':
		bn_names    = get_bn_names(model)
		norm_weight = state_dict[bn_names[layer_id]+'.weight']
		norm_bias   = state_dict[bn_names[layer_id]+'.bias']
	elif arch == 'googlenet':
		bn_names    = get_bn_names(model)
		name        = bn_names[layer_id]
		norm_weight = state_dict[name+'.weight']
		norm_bias   = state_dict[name+'.bias']

	num = len(norm_bias)
	for i in range(num):
		if norm_weight[i].item() < 0.: 
			norm_weight[i] = 1e-5
		_norm = norm(norm_bias[i].item(),norm_weight[i].item())
		relu_expect.append(_norm.expect(lb=0))
	return relu_expect

def get_expect_after_relu(_nxt_filter,relu_expect,saved=[]):
	n,h,w = _nxt_filter.size()
	assert(n==len(relu_expect))
	expect = 0.
	for i in range(n):
		if i in saved:
			expect += _nxt_filter[i].sum().item() * relu_expect[i]
	return expect

def get_effect_for_dstr(model,feature_id,next_layer_id,relu_expect_i,type='mean',slices=None):
	conv_names = get_conv_names(model=model)
	state_dict_conv = model.state_dict()[conv_names[next_layer_id]]
	num = len(state_dict_conv)
	offset = 0
	if slices is not None: offset = slices[0]
	value = 0.
	_max  = 0.
	for i in range(num):
		temp   = abs(state_dict_conv[i][feature_id+offset].sum().item() * relu_expect_i)
		value += temp
		_max   = max(_max,temp)
	if type == 'mean':
		return value/num
	elif type == 'maxm':
		return _max
	return value

def get_relu_expect_resnet_50(model,layer_id):
	relu_expect = []
	block_0_last_id = [3,13,26,45]
	block_dsamp_last_id = [4,14,27,46]
	block_last_id = [7]+[17,20]+[30,33,36,39]+[49]+[10,23,42]

	state_dict = model.state_dict()
	bn_names   = get_bn_names(model)

	if layer_id in block_0_last_id:
		norm_weight_1 = state_dict[bn_names[layer_id]+'.weight']
		norm_bias_1   = state_dict[bn_names[layer_id]+'.bias']
		norm_weight_2 = state_dict[bn_names[layer_id+1]+'.weight']
		norm_bias_2   = state_dict[bn_names[layer_id+1]+'.bias']
		num           = len(norm_bias_1)
		for i in range(num):
			if norm_weight_1[i].item() < 0.: norm_weight_1[i] = 1e-5
			if norm_weight_2[i].item() < 0.: norm_weight_2[i] = 1e-5	
			norm_1 = norm(norm_bias_1[i].item(),norm_weight_1[i].item())
			norm_2 = norm(norm_bias_2[i].item(),norm_weight_2[i].item())
			mu     = norm_bias_1[i].item()+norm_bias_2[i].item()
			sigma  = sqrt((norm_weight_1[i].item())**2+(norm_weight_2[i].item())**2)
			norm_3 = norm(mu,float(sigma))
			relu_expect.append(norm_3.expect(lb=0)) #
	elif layer_id in block_dsamp_last_id:
		return get_relu_expect_resnet_50(model,layer_id-1)
	elif layer_id in block_last_id:
		relu_expect_1 = get_relu_expect_resnet_50(model,layer_id-3)
		norm_weight   = state_dict[bn_names[layer_id]+'.weight']
		norm_bias     = state_dict[bn_names[layer_id]+'.bias']
		num           = len(norm_bias)
		for i in range(num):
			if norm_weight[i].item() < 0.: norm_weight[i] = 1e-5
			_norm = norm(norm_bias[i].item(),norm_weight[i].item())
			relu_expect.append(_norm.expect(lb=0)+relu_expect_1[i])
	else :
		norm_weight = state_dict[bn_names[layer_id]+'.weight']
		norm_bias   = state_dict[bn_names[layer_id]+'.bias']
		num         = len(norm_bias)
		for i in range(num):
			if norm_weight[i].item() < 0.: norm_weight[i] = 1e-5
			_norm = norm(norm_bias[i].item(),norm_weight[i].item())
			relu_expect.append(_norm.expect(lb=0))
	return relu_expect

def get_rank_resnet_50(model,num,layer_id,type='mean'):
	layer_last_id = [0,10,23,42,52]
	block_0_last_id = [3,13,26,45]
	block_last_id = [7]+[17,20]+[30,33,36,39]+[49]
	block_dsamp_last_id = [4,14,27,46]

	effects = []
	if layer_id == 52:
		relu_expect = get_relu_expect_resnet_50(model,layer_id-3)
		for i in range(num):
			effect = get_effect_for_dstr(model,i,layer_id-2,relu_expect[i],type=type)
			effects.append(effect)
	elif layer_id in layer_last_id:
		relu_expect = get_relu_expect_resnet_50(model,layer_id)
		for i in range(num):
			effect_1 = get_effect_for_dstr(model,i,layer_id+1,relu_expect[i],type=type)
			effect_2 = get_effect_for_dstr(model,i,layer_id+4,relu_expect[i],type=type)
			effects.append(effect_1+effect_2)
	elif layer_id in block_0_last_id+block_last_id+block_dsamp_last_id:
		_layer_id = layer_last_id[bisect.bisect_right(layer_last_id,layer_id)]
		return get_rank_resnet_50(model,num,_layer_id,type=type)
	else :
		relu_expect = get_relu_expect_resnet_50(model,layer_id)
		for i in range(num):
			effect = get_effect_for_dstr(model,i,layer_id+1,relu_expect[i],type=type)
			effects.append(effect)
	rank = list(np.argsort(effects))
	return rank

if __name__ == '__main__':

	init()

	if args.from_scratch is True:
		train_from_scratch()
	else :
		if args.arch == 'vgg_16_bn':
			iter_vgg16bn()
		elif args.arch == 'resnet_56':
			iter_resnet_56()
		elif args.arch == 'resnet_50':
			iter_resnet_50()
		elif args.arch == 'googlenet':
			iter_googlenet()
