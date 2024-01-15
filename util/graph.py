"""
@author : XuShuo
@when : 2022-10-26
@homepage : https://github.com/xushuo0629
"""

import matplotlib.pyplot as plt
import re
import seaborn as sns
# 不加下面这两行，在我的环境里竟然会报错
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 貌似是libiomp5md.dll在虚拟环境里有重复的dll
# 加这两行是最粗暴的方法，以后遇到别的项目有相同问题也copy一下


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw():
    train = read('save/result/train_loss.txt')
    sns.set(color_codes=True, font='cmr10')
    sns.lineplot(data=train, color='r', ci=60, label='train')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.savefig("save/result/training curve.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    draw()


