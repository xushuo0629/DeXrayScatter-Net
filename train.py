"""
@author : XuShuo
@when : 2024-1-11
@homepage : https://github.com/xushuo0629
"""
import torch
from torch import nn, optim
import torch.cuda
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torch.optim import Adam

import math
import time
from util.data_loader import DXS_dataset
from util.epoch_timer import epoch_time
from util.loss import *
from conf_train import *
from util.graph import *
from util.setup_seed import *
from models.skipXS import *
from models.DnCNN import *
from models.resnet import *
from models.KBarchs.kbnet_l_arch import *
from models.MIRarchs.MIRNet_model import *

setup_seed(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def train(model, train_batch, optimizer, criterion1,criterion2,criterion3, clip):
    model.train()
    epoch_loss = 0
    for i, (image,label,name) in enumerate(train_batch):
        # src = batch.src
        # trg = batch.trg
        # reshape data
        image1 = image.unsqueeze(1).to(device)  # batch*128*128       ---->   batch* 1 *128*128
        label1 = label.unsqueeze(1).to(device)  # batch*128*128       ---->   batch* 1 *128*128

        optimizer.zero_grad()

        out = model(image1)
        out = out.clamp(min=0,max=255)

        # out = out.contiguous().view(-1, out.size(-1))  # batch*1*1 ----> batch*1\n",
        # z = z.contiguous().view(-1, 1)  # batch ----> batch*1\n",

        loss1 = criterion1(out, label1)
        loss2 = 1- criterion2(out, label1)
        loss3 = criterion3(out, label1)
        loss = loss1  + loss2 + loss3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        print('         step :', round(((i+1 )/ len(train_batch)) * 100, 2),
              '%, MSEloss :', loss1.item(),
              # ' SSIMloss :', loss2.item() ,
              ', CharbonnierLoss:', loss3.item(),'\r', end='')

    print(' ', '\n', end='')
    return epoch_loss / len(train_batch)


def run(total_epoch, best_loss):
    train_losses, bleus = [], []
    start_time = time.time()
    for step in range(total_epoch):
        train_loss = train(model, train_batch, optimizer, criterion1,criterion2,criterion3, clip)
        end_time = time.time()

        if step > warmup:
            scheduler.step(train_loss)

        train_losses.append(train_loss)
        # bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        f = open('save/result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()
        lr = optimizer.param_groups[0]['lr']
        # print('---------------------------------------------------')
        print(f'\t  Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | LR：  {lr:.2e} ')
        print('---------------------------------------------------')
        if (step+1) % every_save ==0 :
            t = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
            name = 'model-'+t+\
                       '-{:.4f}lr'.format(init_lr)+'-{:d}epoch'.format(epoch)+\
                     '-{:d}BatchSize'.format(batch_size)+'.pt'
            torch.save(model.state_dict(), 'save/' + name)
    torch.save(model.state_dict(), 'save/model.pt')
    draw()

"""
设置模型，优化器，损失函数
"""
# model = skipXS(num_input_channels=1, num_output_channels=1,
#              num_channels_down=[32, 64, 128, 256, 256], filter_size_down=[3, 3, 3, 3, 3],
#              num_channels_up=[32, 64, 128, 256, 256], filter_size_up=[3, 3, 3, 3, 3],
#              num_channels_skip=[4, 4, 4, 4, 4], filter_skip_size=[1, 1, 1, 1, 1],
#              upsample_mode=['bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear'],
#              downsample_mode=['stride', 'stride', 'stride', 'stride', 'stride'],
#              need_sigmoid=False, need_bias=True, pad='zero', act_fun="LeakyReLU", need_attention=True).type(dtype)
# model = KBNet_s(img_channel=1, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
#                  dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s', lightweight=False, ffn_scale=2).type(dtype)
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



optimizer = Adam(params= model.parameters(),
                 lr=init_lr,
                 betas=(0.9, 0.999),
                 weight_decay=weight_decay,
                 eps=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience,
                                                 min_lr=1e-8)

#criterion = nn.L1Loss()
criterion1 = nn.MSELoss(reduction="mean")
#criterion2 = PSNRLoss()
criterion2 = SSIM()
criterion3 = CharbonnierLoss()
# criterion = MAPE()
# criterion = SMAPE()
"""
初始化
"""
print(f'The model has {count_parameters(model):,} trainable parameters')

if finetune:
    model.load_state_dict(torch.load(path_pretrain))
    print('load pretrained model', path_pretrain)
else:
    model.apply(initialize_weights)
    print('initialize model weights')
"""
载入数据
"""


Loader = DXS_dataset(path_img, path_label)
Loader_aug = DXS_dataset(path_img, path_label,IfRandomCrop= True)
Loader = ConcatDataset([Loader, Loader_aug])

print('加入随机裁剪，增强数据集总数量为：', len(Loader))

train_batch = DataLoader(Loader, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=float('inf'))

