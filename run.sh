
# --------------------------vgg_16_bn----------------------------------
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

# --------------------------resnet_56----------------------------------

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

# --------------------------googlenet----------------------------------

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

# ----------------------------resnet_50--------------------------------

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
