"""
@author : XuShuo
@when : 2024-1-11
@homepage : https://github.com/xushuo0629
"""
# data
path_img  = './data/train/fig_low/'
path_label = './data/train/fig_high/'
path_img2   = './data/test/fig_low/'
path_label2 = './data/test/fig_high/'

finetune = True
path_pretrain = './save/KBnet/model_pretrain1800fine.pt'

every_save = 5
batch_size = 6   #
epoch = 200
seed = 3407

# for KBnet continue
init_lr = 8e-7 # 学习率 3e-6
weight_decay = 1e-8 # adam 衰减参数
factor = 0.5
patience = 3 # 每间隔patience次迭代，学习率更新为factor倍
warmup = 0
clip = 1.0 # 梯度截断参数






