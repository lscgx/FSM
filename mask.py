import torch
import numpy as np
import pickle
import utils

class mask_vgg_16_bn:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=4, rank = None, type = 1, arch="vgg_16_bn"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_vgg_16_bn'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id + 1) * param_per_cov:
                break
            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                if rank is None:
                    rank = np.load(prefix + str(cov_id) + subfix)
                    utils.logger.info('loading '+prefix + str(cov_id) + subfix)

                if type == 1:
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds  
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds  
                item.data = item.data * self.mask[index]
            if index > cov_id * param_per_cov and index < (cov_id + 1) * param_per_cov :
                self.mask[index] = torch.squeeze(inds)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index < cov_id * self.param_per_cov:
                continue
            if index in mask_keys:
                item.data = item.data * self.mask[index]

class mask_resnet_56:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3, rank = None, type = 1, arch="resnet_56"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_resnet_56'
        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):
            if index == (cov_id + 1)*param_per_cov:
                break
            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                if rank is None:
                    rank = np.load(prefix + str(cov_id) + subfix)
                    utils.logger.info('loading '+prefix + str(cov_id) + subfix)
                if type == 1:
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds  
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds  
                item.data = item.data * self.mask[index]
            elif index > cov_id*param_per_cov and index < (cov_id + 1)*param_per_cov:
                self.mask[index] = torch.squeeze(inds)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == (cov_id + 1)*self.param_per_cov:
                break
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_googlenet:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=4, rank = None, type = 1, arch="googlenet"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_googlenet'
        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):
            if index == (cov_id + 1) * param_per_cov:
                break
            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                if rank is None:
                    rank = np.load(prefix + str(cov_id) + subfix)
                    utils.logger.info('loading '+prefix + str(cov_id) + subfix)
                if type == 1:
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds 
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds 
                item.data = item.data * self.mask[index]
            if index > cov_id * param_per_cov and index < (cov_id + 1) * param_per_cov :
                self.mask[index] = torch.squeeze(inds)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov + 4: 
                break
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_resnet_50:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3, rank = None, type = 1, arch="resnet_50"):
        assert(rank is not None)
        params = self.model.parameters()

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_resnet_50'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):
            if index == (cov_id + 1) * param_per_cov:
                break
            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                if type == 1:
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds  
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds  
                item.data = item.data * self.mask[index]

            elif index > cov_id * param_per_cov and index < (cov_id + 1)* param_per_cov:
                self.mask[index] = torch.squeeze(inds)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == (cov_id + 1)*self.param_per_cov:
                break
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)