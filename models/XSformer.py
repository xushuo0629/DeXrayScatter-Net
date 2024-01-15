"""
@author : XuShuo
@when : 2022-10-18
@homepage : https://github.com/xushuo0629
"""
import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """Implement the PE function"""
    def __init__(self, d_model, dropout, max_len=4):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class XSformer(nn.Module):
    def __init__(self, nhead, d_model, dim_feedforward, dropout, n_layers, n_mlp, mlp_hidden):
        super(XSformer, self).__init__()

        # 定义词向量，这步替换为物理嵌入
        # self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=128)
        # 定义Transformer。超参是我拍脑袋想的
        """
        一个用法参考： 
        self.transformer = nn.Transformer(d_model=128, 
                                          num_encoder_layers=2, 
                                          num_decoder_layers=2, 
                                          dim_feedforward=512, 
                                          batch_first=True)
        """
        self.transformer1 = nn.Transformer(nhead=nhead,
                                           d_model=d_model,
                                           dim_feedforward=dim_feedforward,
                                           batch_first=True,
                                           dropout=dropout,
                                           num_encoder_layers=n_layers,
                                           num_decoder_layers=n_layers)
        self.transformer2 = nn.Transformer(nhead=nhead,
                                           d_model=d_model,
                                           dim_feedforward=dim_feedforward,
                                           batch_first=True,
                                           dropout=dropout,
                                           num_encoder_layers=n_layers,
                                           num_decoder_layers=n_layers)
        self.transformer3 = nn.Transformer(nhead=nhead,
                                           d_model=d_model,
                                           dim_feedforward=dim_feedforward,
                                           batch_first=True,
                                           dropout=dropout,
                                           num_encoder_layers=n_layers,
                                           num_decoder_layers=n_layers)
        self.transformer4 = nn.Transformer(nhead=nhead,
                                           d_model=d_model,
                                           dim_feedforward=dim_feedforward,
                                           batch_first=True,
                                           dropout=dropout,
                                           num_encoder_layers=n_layers,
                                           num_decoder_layers=n_layers)

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        # 定义最后的线性层，
        self.n_mlp = n_mlp
        self.mlp_in = nn.Linear(d_model*4, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

        self.line_block = nn.Sequential(
            nn.Linear(mlp_hidden,mlp_hidden),
            nn.ReLU())

        self.layers = nn.ModuleList([self.line_block
                                     for i in range(n_mlp)])

        self.ac = nn.LeakyReLU(0.1)


    def forward(self, src, tgt):
        # 生成mask这一步省略
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        # src_key_padding_mask = CopyTaskModel.get_key_padding_mask(src)
        # tgt_key_padding_mask = CopyTaskModel.get_key_padding_mask(tgt)
        # 对src和trg的编码，省略
        # src = self.embedding(src)
        # tgt = self.embedding(tgt)
        # 给src的token增加位置信息，tgt改为能量编码(需不需要嵌入位置还要验证)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        A = self.transformer1(src, tgt)
        B = self.transformer2(src, tgt)
        C = self.transformer3(src, tgt)
        D = self.transformer4(src, tgt)
        out = torch.cat((A,B,C,D),2)
        # print('cat size',out.size()) #for debug
        # 对输出的batch*len(1)* d_model(6)进行预测计算: 暂时解释为等效层

        temp = self.mlp_in(out)
        for layer in self.layers:
            temp = layer(temp)
        out = self.mlp_out(temp)
        out = self.ac(out)
        """
        这里返回batch*1*1的结果。理解为等效层到累计因子的计算步骤，
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask  暂时不使用mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask

