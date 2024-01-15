"""
@author : XuShuo
@when : 2024-1-11
@homepage : https://github.com/xushuo0629
"""
from models.skipXS import *
from models.DnCNN import *
from models.resnet import *
from models.KBarchs.kbnet_l_arch import *
from models.MIRarchs.MIRNet_model import *

import os
def My_mkdir(path):
    path = path.strip()  # 去除首位空格
    path = path.rstrip("\\")  # 去除尾部 \ 符号
    # 判断路径是否存在
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False
def My_deletePic(directory):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 判断文件类型为图片（这里只考虑常见的jpg、png等格式）
        if any(filename.endswith(extension) for extension in ['.jpg', '.jpeg', '.png']):
            file_path = os.path.join(directory, filename)
            try:  # 删除文件
                os.remove(file_path)
            except OSError as e:
                print('can not delete：{}, {}'.format(file_path, str(e)))
    print('delete images')

dtype = torch.cuda.FloatTensor


# model = skipXS(num_input_channels=1, num_output_channels=1,
#              num_channels_down=[32, 64, 128, 256, 512], filter_size_down=[3, 3, 3, 3, 3],
#              num_channels_up=[32, 64, 128, 256, 512], filter_size_up=[3, 3, 3, 3, 3],
#              num_channels_skip=[4, 4, 4, 4, 4], filter_skip_size=[1, 1, 1, 1, 1],
#              upsample_mode=['bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear'],
#              downsample_mode=['stride', 'stride', 'stride', 'stride', 'stride'],
#              need_sigmoid=False, need_bias=True, pad='zero', act_fun="LeakyReLU", need_attention=True).type(dtype)

model = KBNet_l(inp_channels=1, out_channels=1, dim=12,
                num_blocks=[2, 4, 4, 8], num_refinement_blocks=2,
                heads=[1, 2, 4, 8], ffn_expansion_factor=1.5, bias=False,
                blockname='KBBlock_l').type(dtype)

# model = DnCNN(channels=1, num_of_layers=7).type(dtype)

# model = ResNet(num_input_channels=1, num_output_channels=1, num_blocks=5,
#              num_channels=128, need_residual=True, act_fun='LeakyReLU',
#              need_sigmoid=False, norm_layer=nn.BatchNorm2d, pad='reflection').type(dtype)

# model = MIRNet(in_channels=1, out_channels=1, n_feat=64, kernel_size=3,
#                stride=2, n_RRG=3, n_MSRB=2, height=3, width=2, bias=False)

## 训练集
path_img   =  './data/train/fig_low/'
path_label =  './data/train/fig_high/'
path_out = './data/train/DXS-KBnet-pretrain1740fine/'
# batch_size = 10

# 测试集
# path_img   = './data/test/fig_low/'
# path_label = './data/test/fig_high/'
# path_out = './data/test/DXS-KBnet-pretrain1800fine/'

model_path = './save/KBnet/model_pretrain1800fine.pt'
batch_size = 8
My_mkdir(path_out)
My_deletePic(path_out)

