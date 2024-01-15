"""
@author : XuShuo
@when : 2024-1-11
@homepage : https://github.com/xushuo0629
"""
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from util.data_loader import DXS_dataset
from util.loss import MAPE, SMAPE
from conf_test import *
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from scipy import ndimage
from scipy.interpolate import griddata

from models.skipXS import *
from models.DnCNN import *
from models.resnet import *
from models.KBarchs.kbnet_s_arch import *
from models.KBarchs.kbnet_l_arch import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# 不加下面这两行，在我的环境里竟然会报错
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def interpolate_outliers(image, max_value=253):
    # 找到所有异常点的位置
    outlier_indices = np.where(image > max_value)

    # 找到所有正常点的位置
    valid_indices = np.where(image <= max_value)

    # 用于插值的坐标点 (非异常值)
    points = np.column_stack(valid_indices)

    # 用于插值的值 (非异常值)
    values = image[valid_indices]

    # 需要插值的坐标点 (异常值)
    points_outliers = np.column_stack(outlier_indices)

    # 执行插值
    image_interpolated = np.copy(image)
    image_interpolated[outlier_indices] = griddata(points, values, points_outliers, method='nearest')

    return image_interpolated


def interpolate_outliers2(image, max_value=252):
    # 获取异常点的位置
    mask_outliers = image > max_value

    # 创建一个相同大小的数组，但仅在异常点上有值
    image_outliers = np.copy(image)
    image_outliers[~mask_outliers] = 0

    # 创建仅包含非异常点的图像
    image_no_outliers = np.copy(image)
    image_no_outliers[mask_outliers] = 0

    # 使用高斯滤波对异常点进行模糊处理
    interpolation = ndimage.gaussian_filter(image_outliers, sigma=1)

    # 替换掉原图中的异常点
    image_corrected = image_no_outliers + interpolation
    image_corrected[image_corrected > max_value] = max_value  # 确保插值后的值不超过最大值

    return image_corrected



"""
设置模型 --- 转移在conf_test.py里
"""
# model = skipXS(num_input_channels=1, num_output_channels=1,
#              num_channels_down=[32, 64, 128, 256, 512], filter_size_down=[3, 3, 3, 3, 3],
#              num_channels_up=[32, 64, 128, 256, 512], filter_size_up=[3, 3, 3, 3, 3],
#              num_channels_skip=[4, 4, 4, 4, 4], filter_skip_size=[1, 1, 1, 1, 1],
#              upsample_mode=['bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear'],
#              downsample_mode=['stride', 'stride', 'stride', 'stride', 'stride'],
#              need_sigmoid=False, need_bias=True, pad='zero', act_fun="LeakyReLU", need_attention=True).type(dtype)

# model = KBNet_l(inp_channels=1, out_channels=1, dim=12,
#                 num_blocks=[2, 4, 4, 8], num_refinement_blocks=2,
#                 heads=[1, 2, 4, 8], ffn_expansion_factor=1.5, bias=False,
#                 blockname='KBBlock_l').type(dtype)

# model = DnCNN(channels=1, num_of_layers=7).type(dtype)

# model = ResNet(num_input_channels=1, num_output_channels=1, num_blocks=5,
#              num_channels=128, need_residual=True, act_fun='LeakyReLU',
#              need_sigmoid=False, norm_layer=nn.BatchNorm2d, pad='reflection').type(dtype)

print(f'The model has {count_parameters(model):,} trainable parameters')

"""
载入数据
"""
loader = DXS_dataset(path_img, path_label)
test_batch = DataLoader(loader, batch_size = batch_size, shuffle=False)

def test_model(num_examples):

    model.load_state_dict(torch.load(model_path))
    print('load pretrained model', model_path)
    total_test = []
    f = open('save/result/test.txt', 'w')
    with torch.no_grad():
        for i, (image,label,name) in enumerate(test_batch):
            # reshape data
            image1 = image.unsqueeze(1).to(device)  # batch*128*128       ---->   batch* 1 *128*128

            pred = model(image1)
            pred  = pred.clamp(min=0, max=255)

            for j in range(pred.shape[0]):
                pred_DXS = pred[j].squeeze().cpu().numpy()
                #pred_DXS = interpolate_outliers(pred_DXS0)
                #print('插值前max：{:.1f}'.format(np.max(pred_DXS0))+' 插值后max：{:.1f}'.format(np.max(pred_DXS)))

                label_img =  label[j].numpy()
                psnr_img = psnr(np.uint8(label_img),np.uint8(pred_DXS))
                ssim_img = ssim(np.uint8(label_img),np.uint8(pred_DXS))

                Tx = name[j] + ' psnr: {:.3f}'.format(psnr_img)+'   ssim_img: {:.3}'.format(ssim_img)

                # 将NumPy数组转换为PIL图像对象
                pil_image = Image.fromarray(np.uint8(pred_DXS),mode='L')
                # 将PIL图像对象保存为PNG图像
                pil_image.save( path_out+ name[j]+'.png')

                print(Tx)
                f.write(Tx+'\n')

    f.close()
    print('---------------------------------------------------')
    print('psnr: {:.3f}'.format(psnr_img)+'   ssim_img: {:.3}'.format(ssim_img))
    plt.show()

if __name__ == '__main__':
    test_model(num_examples=batch_size)
