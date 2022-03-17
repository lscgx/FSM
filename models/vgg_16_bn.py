import torch
import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512          ],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, last_dim=512, num_class=10):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(OrderedDict([
        	('linear1', nn.Linear(last_dim, 512)),
        	('norm1', nn.BatchNorm1d(512)),
        	('relu1', nn.ReLU(inplace=True)),
        	('linear2', nn.Linear(512, num_class))
        	])
        )

    def forward(self, x):
        output = self.features(x)
        output = nn.AvgPool2d(kernel_size=2, stride=2)(output)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, ranks = None , batch_norm=False):
    layers = []

    input_channel = 3
    N = 13
    cnt_m = 0
    cnt_r = 0
    first_d = True
    for l in cfg:
        
        if l == 'M':
            layers.append(('pool%d' % cnt_m, nn.MaxPool2d(kernel_size=2, stride=2)))
            cnt_m += 1
            continue

        if(ranks is not None and cnt_r < N ): #and cnt_r < len(ranks) - 1 
            l = len(ranks[cnt_r])

        cnt = cnt_r + cnt_m

        layers.append(('conv%d' % cnt , nn.Conv2d(input_channel, l, kernel_size=3, padding=1)))

        if batch_norm:
            layers.append(('norm%d' % cnt, nn.BatchNorm2d(l)))

        layers.append(('relu%d' % cnt, nn.ReLU(inplace=True)))
        input_channel = l

        cnt_r += 1

    return nn.Sequential(OrderedDict(layers))

def vgg_16_bn(oristate_dict = None,ranks = None,self_cfg = None,compress_rate = None):
    _cfg = cfg['D']
    if self_cfg is not None: _cfg = self_cfg
    model = None 

    if oristate_dict is not None and ranks is not None:
        model = VGG(make_layers(_cfg, ranks , batch_norm=True),last_dim=len(ranks[-1]))
        state_dict = model.state_dict()
        cov_id = 0
        rank = None
        N = len(ranks)
        last_select_index = None

        for k,name in enumerate(oristate_dict):
            if k < 13*7 :
                if k%7 == 0:
                    rank = ranks[cov_id]
                    if last_select_index is not None:
                        for index_i,i in enumerate(rank):
                            for index_j,j in enumerate(last_select_index):
                                state_dict[name][index_i][index_j] = oristate_dict[name][i][j]
                    else:
                        for index_i,i in enumerate(rank):
                            state_dict[name][index_i] = oristate_dict[name][i]

                    last_select_index = rank
                    cov_id += 1
                elif k%7 == 6:
                    state_dict[name] = oristate_dict[name]
                else:
                    for index_i,i in enumerate(rank):
                        state_dict[name][index_i] = oristate_dict[name][i]
            elif k == 13*7:
                rank = [x for x in range(512)]
                for index_i,i in enumerate(rank):
                    for index_j,j in enumerate(last_select_index):
                        state_dict[name][index_i][index_j] = oristate_dict[name][i][j]
            else :
                state_dict[name] = oristate_dict[name]

        model.load_state_dict(state_dict)
    elif oristate_dict is None and ranks is not None:
        model = VGG(make_layers(_cfg, ranks , batch_norm=True),last_dim=len(ranks[-1]))
    elif compress_rate is None:
        model = VGG(make_layers(_cfg, batch_norm=True))
    else :
        tmp = [64]*2+[128]*2+[256]*3+[512]*6
        ranks = []
        for i,x in enumerate(compress_rate):
            ranks.append(list(range(math.ceil(tmp[i]*(1.-x)))))
        model = VGG(make_layers(_cfg, ranks , batch_norm=True),last_dim=len(ranks[-1]))
    return model
