import torch.nn as nn
import math

stage_repeat      = [3, 4, 6, 3]
stage_out_channel = [64] + [256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3
stage_channels    = [[64],[64,64,256],[128,128,512],[256,256,1024],[512,512,2048]]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    def __init__(self, inplanes , midplanes, planes, compress_rate = 0., stride=1, is_downsample=False):
        super(Bottleneck, self).__init__()
        expansion = 4

        inplanes  = math.ceil(inplanes * (1-compress_rate[0]))
        midplanes_1 = math.ceil(midplanes * (1-compress_rate[1]))
        midplanes_2 = math.ceil(midplanes * (1-compress_rate[2]))
        planes    = math.ceil(planes * (1-compress_rate[3]))

        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, midplanes_1)
        self.bn1 = norm_layer(midplanes_1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(midplanes_1, midplanes_2, stride)
        self.bn2 = norm_layer(midplanes_2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv1x1(midplanes_2, planes)
        self.bn3 = norm_layer(planes)
        self.relu3 = nn.ReLU(inplace=True)

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        self.midplanes_1 = midplanes_1
        self.midplanes_2 = midplanes_2

        self.is_downsample = is_downsample
        self.expansion = expansion

        if is_downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                norm_layer(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, compress_rate, num_classes=1000):
        super(ResNet50, self).__init__()

        # overall_channel, mid_channel = adapt_channel(compress_rate)
        self.num_blocks = stage_repeat

        layer_num = 0
        inplanes  = math.ceil(64 * (1-compress_rate[0]))
        self.conv1 = nn.Conv2d(3, inplanes , kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.ModuleList()
        self.layer2 = nn.ModuleList()
        self.layer3 = nn.ModuleList()
        self.layer4 = nn.ModuleList()

        layer_num += 1
        cnt = 1
        for i in range(len(stage_repeat)):
            if i == 0:
                eval('self.layer%d' % (i+1)).append(Bottleneck(stage_channels[layer_num-1][-1], stage_channels[layer_num][1], stage_channels[layer_num][2], compress_rate[(cnt-1)*3:cnt*3+1] , stride=1, is_downsample=True))
            else:
                eval('self.layer%d' % (i+1)).append(Bottleneck(stage_channels[layer_num-1][-1], stage_channels[layer_num][1], stage_channels[layer_num][2], compress_rate[(cnt-1)*3:cnt*3+1] , stride=2, is_downsample=True))

            cnt += 1
            for j in range(1, stage_repeat[i]):
                eval('self.layer%d' % (i+1)).append(Bottleneck(stage_channels[layer_num][-1], stage_channels[layer_num][1], stage_channels[layer_num][2],compress_rate[(cnt-1)*3:cnt*3+1]))
                cnt += 1
            layer_num +=1   

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(math.ceil(2048*(1.-compress_rate[-1])), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)
        for i, block in enumerate(self.layer4):
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet_50(compress_rate=None,oristate_dict = None,ranks = None):
    model = None 
    if oristate_dict is not None and compress_rate is not None: 
        if len(compress_rate) < 49:
            compress_rate = compress_rate + [0.] * (49 - len(compress_rate))
        if ranks is None :
            ranks = ['no_pruned'] * 53
        elif len(ranks) < 53:
            ranks = ranks + ['no_pruned'] * (53 - len(ranks))
            
        model = ResNet50(compress_rate=compress_rate)

        state_dict = model.state_dict()
        cov_id = 0
        N = 55
        in_rank = None  #conv1
        rank = None
        last_select_index = None

        for k,name in enumerate(oristate_dict):
            if k <= 317:
                if k in [24,84,162,276]: #downsample
                    rank = ranks[cov_id]
                    if rank == 'no_pruned': rank = list(range(len(state_dict[name])))
                    for _i,i in enumerate(rank):
                        for _j,j in enumerate(in_rank):
                            state_dict[name][_i][_j] = oristate_dict[name][i][j]
                    cov_id += 1
                elif k%6 == 0:
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
                if k in [0,60,138,252]:
                    in_rank = rank
            elif k == 318:
                rank = list(range(1000))
                for _i,i in enumerate(rank):
                    for _j,j in enumerate(last_select_index):
                        state_dict[name][_i][_j] = oristate_dict[name][i][j]
            elif k == 319:
                state_dict[name] = oristate_dict[name]

        model.load_state_dict(state_dict)
    elif compress_rate is not None:
        model = ResNet50(compress_rate=compress_rate)
    else :
        compress_rate = [0.] * 100
        model = ResNet50(compress_rate=compress_rate)
    return model