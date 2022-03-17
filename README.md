# (FSM)

## Environments

The code has been tested in the following environments:

- Python 3.8
- PyTorch 1.8.1
- cuda 10.2
- torchsummary, torchvision, thop, scipy, sympy

## Pre-trained Models

**CIFAR-10:**

[Vgg-16](https://drive.google.com/open?id=1i3ifLh70y1nb8d4mazNzyC4I27jQcHrE) | [ResNet56](https://drive.google.com/open?id=1f1iSGvYFjSKIvzTko4fXFCbS-8dw556T) | [GoogLeNet](https://drive.google.com/open?id=1rYMazSyMbWwkCGCLvofNKwl58W6mmg5c)

**ImageNet:**

[ResNet50](https://drive.google.com/open?id=1OYpVB84BMU0y-KU7PdEPhbHwODmFvPbB)

## Running Code

The experiment settings are as follows:

**1. VGGNet**

| Compression Rate           | Flops($\downarrow $) | Params($\downarrow $) | Accuracy |
| -------------------------- | -------------------- | --------------------- | -------- |
| [0.3]\*5+[0.35]\*3+[0.8]*5 | 106.8M(66.0%)        | 2.05M(86.3%)          | 93.73%   |
| [0.5]\*8+[0.8]\*5          | 59.7M(81.0%)         | 1.41M(90.6%)          | 92.86%   |

```shell
#VGGNet
#All run scripts can be cut-copy-paste from run.bat or run.sh.
python main.py \
--arch vgg_16_bn \
--resume [pre-trained model dir] \
--compress_rate [0.3]*5+[0.35]*3+[0.8]*5 \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--lr_decay_step 1 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 

python main.py \
--arch vgg_16_bn \
--from_scratch True \
--resume final_pruned_model/vgg_16_bn_1.pt \
--num_workers 1 \
--epochs 200 \
--gpu 0 \
--lr 0.01 \
--lr_decay_step 100,150 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 
```

**2. ResNet-56**

| Compression Rate                        | Flops($\downarrow $) | Params($\downarrow $) | Accuracy |
| --------------------------------------- | -------------------- | --------------------- | -------- |
| [0.]+[0.6,0.]\*9+[0.6,0.]\*9+[0.4,0.]*9 | 61.7M(51.2%)         | 0.48M(43.6%)          | 93.58%   |
| [0.]+[0.7,0.]\*9+[0.7,0.]\*9+[0.7,0.]*9 | 40.2M(68.2)          | 0.27M(68.5)           | 92.76%   |

```shell
#ResNet-56
python main.py \
--arch resnet_56 \
--resume [pre-trained model dir] \
--compress_rate [0.]+[0.6,0.]*9+[0.6,0.]*9+[0.4,0.]*9 \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--lr_decay_step 1 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 

python main.py \
--arch resnet_56 \
--from_scratch True \
--resume final_pruned_model/resnet_56_1.pt \
--num_workers 1 \
--epochs 300 \
--gpu 0 \
--lr 0.01 \
--lr_decay_step 150,225 \
--weight_decay 0.0005 \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 
```

**3. GoogLeNet**

| Compression Rate                     | Flops($\downarrow $) | Params($\downarrow $) | Accuracy |
| ------------------------------------ | -------------------- | --------------------- | -------- |
| [0.2]+[0.7]\*15+[0.8]\*9+[0.,0.4,0.] | 0.567B(63.0%)        | 2.75M(55.5%)          | 94.72%   |
| [0.2]+[0.9]\*24+[0.,0.4,0.]          | 0.376B(75.4%)        | 2.19M(64.6%)          | 94.29%   |

```shell
#GoogLeNet
python main.py \
--arch googlenet \
--resume [pre-trained model dir] \
--compress_rate [0.2]+[0.7]*15+[0.8]*9+[0.,0.4,0.] \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 

python main.py \
--arch googlenet \
--from_scratch True \
--resume final_pruned_model/googlenet_1.pt \
--num_workers 1 \
--epochs 200 \
--lr 0.01 \
--lr_decay_step 100,150 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 
```

**4. ResNet-50**

| Compression Rate                                             | Flops($\downarrow $) | Params($\downarrow $) | Top-1 Acc | Top-5 Acc |
| ------------------------------------------------------------ | -------------------- | --------------------- | --------- | --------- |
| [0.]+[0.2,0.2,0.2]\*1+[0.65,0.65,0.2]\*2+[0.2,0.2,0.2]\*1+[0.65,0.65,0.2]\*3+[0.2,0.2,0.2]\*1+[0.65,0.65,0.2]\*5+[0.2,0.2,0.1]+[0.2,0.2,0.1]\*2 | 1.76B(57.2%)         | 14.6M(42.8%)          | 75.43%    | 92.45%    |

```shell
#ResNet-50
python main.py \
--arch resnet_50 \
--resume [pre-trained model dir] \
--data_dir [dataset dir] \
--dataset ImageNet \
--compress_rate [0.]+[0.2,0.2,0.2]*1+[0.65,0.65,0.2]*2+[0.2,0.2,0.2]*1+[0.65,0.65,0.2]*3+[0.2,0.2,0.2]*1+[0.65,0.65,0.2]*5+[0.2,0.2,0.1]+[0.2,0.2,0.1]*2 \
--num_workers 4 \
--batch_size 64 \
--epochs 1 \
--lr_decay_step 1 \
--lr 0.001 \
--weight_decay 0. \
--input_size 224 \
--save_id 1 

python main.py \
--arch resnet_50 \
--from_scratch True \
--resume finally_pruned_model/resnet_50_1.pt \
--num_workers 4 \
--epochs 100 \
--lr 0.01 \
--lr_decay_step 30,60,90 \
--batch_size 64 \
--weight_decay 0.0001 \
--input_size 224 \
--data_dir [dataset dir] \
--dataset ImageNet \
--save_id 1
```

