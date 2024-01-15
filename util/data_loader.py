"""
@author : XuShuo
@when : 2022-10-18
@homepage : https://github.com/xushuo0629
"""
import torch
import numpy as np
from torch.utils.data import Dataset,ConcatDataset
from torchvision import transforms
import glob
from PIL import Image
import os

class DXS_dataset(Dataset): # 注意这里的父类继承
    def __init__(self, path_img , path_label, IfRandomCrop= False, seed0 = 2147483647, cp=3):
        super().__init__()
        self.image, self.label, self.name,self.image_np,self.label_np = [],[],[],[],[]
        self.IRC = IfRandomCrop
        self.seedcrop = seed0

        crop_pixel=cp
        # 使用 glob 来匹配目录中的 PNG 文件
        img_files = glob.glob(path_img +'*')
        label_files = glob.glob(path_label + '*')
        # 遍历匹配到的 PNG 文件
        self.name = [os.path.splitext(os.path.basename(file))[0] for file in img_files]  # 只获取文件名部分
        for path in img_files:
           i = Image.open(path).convert('L')

           width, height = i.size
           if width<128:  #针对实验图像处理
               print(np.max(i))
               i = i.rotate(90, expand=True)
               i = i.resize((128, 128)) # 80,440
               width, height = i.size
               i = np.array(i)

               i[ i== 0] = 1
               log_transformed = np.log10(255 / i)
               #由于输出可能有负值或超出255的值，我们将其重新缩放回[0, 255]的范围
               log_transformed = log_transformed - np.min(log_transformed)
               log_transformed = log_transformed / np.max(log_transformed) * 255
               i=  log_transformed
               print(np.max(i))
           else:
               i = i = np.array(i)

           i[0:crop_pixel, :] = 0  # 顶部三圈边缘
           i[-crop_pixel:, :] = 0  # 底部三圈边缘
           i[:, 0:crop_pixel] = 0  # 左侧三圈边缘
           i[:, -crop_pixel:] = 0  # 右侧三圈边缘
           self.image_np.append(i)

           tensor = torch.from_numpy(i.astype(np.float32))
           self.image.append(tensor)


        for path in label_files:
           l = Image.open(path).convert('L')
           l = l.resize((width, height))
           l = np.array(l)

           l[0:crop_pixel, :] = 0  # 顶部三圈边缘
           l[-crop_pixel:, :] = 0  # 底部三圈边缘
           l[:, 0:crop_pixel] = 0  # 左侧三圈边缘
           l[:, -crop_pixel:] = 0  # 右侧三圈边缘
           self.label_np.append(l)

           tensor = torch.from_numpy(l.astype(np.float32))
           self.label.append(tensor)
        print(f"读取到 {len(self.image)} 张图像")

    def __getitem__(self, index):
        if self.IRC:

           img = Image.fromarray(self.image_np[index])  #转PIL
           label = Image.fromarray(self.label_np[index])

           random_crop = transforms.RandomCrop(108)
           seed = self.seedcrop

           np.random.seed(seed)
           torch.manual_seed(seed)
           img = random_crop(img)
           img = np.array(img.resize((128, 128)))
           img = torch.from_numpy(img.astype(np.float32))

           np.random.seed(seed)
           torch.manual_seed(seed)
           label = random_crop(label)
           label = np.array(label.resize((128, 128)))
           label = torch.from_numpy(label.astype(np.float32))
        else:
            img = self.image[index]
            label = self.label[index]
        return img, label, self.name[index]

    def __len__(self):
        return len(self.image)
