from data_processor import *
from Transformer_B import *
from models import *
from plot_drawer import *

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


from torch.utils.data import DataLoader, RandomSampler

from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random

####
#2020.10.28
####

def split_train_test():
    door_number = 100
    split_number = int(door_number * 0.8)
    i = 0
    train_data = []
    test_data = []
    dga_name_list = []

    for filename in os.listdir("./data/dga_tokens/"):
        print(filename)
        full_name = "./data/dga_tokens/" + filename
        f = open(full_name, "rb")
        token = pickle.load(f)
        if len(token) < door_number:
            continue
        if filename == "normal.txt.pkl":
            continue
        change_token = []
        for _ in token:
            change_token.append((0, _[1]))
        random.shuffle(change_token)
        train_data.extend(change_token[:split_number])
        test_data.extend(change_token[split_number:door_number])
        i = i + 1
        dga_name_list.append(filename.split('.')[0])
        f.close()
    print("dga_data prepared")

    f = open("./data/dga_tokens/normal.txt.pkl", "rb")
    token = pickle.load(f)
    change_token = []
    for _ in token:
        change_token.append((1, _[1]))
    random.shuffle(change_token)
    train_data.extend(change_token[:split_number * i])
    test_data.extend(change_token[split_number * i:door_number * i])

    print("normal_deta prepared")
    print(dga_name_list)

    return train_data, test_data, i

def collate_fn(batch):
    lens = [i[1].shape[0] for i in batch]
    bsz, max_len = len(batch),  50

    data_tensor = torch.zeros(bsz, max_len, dtype=torch.long)
    label_tensor = torch.Tensor([i[0] for i in batch])

    for b_ix, b in enumerate(batch):
        data_tensor[b_ix, :b[1].shape[0]] = b[1].unsqueeze(0)

    return data_tensor, label_tensor, lens

#transformer

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            # 给需要mask的地方设置一个负无穷
        	attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)


        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

def residual(sublayer_fn,x):
    return sublayer_fn(x)+x

class LayerNorm(nn.Module):
    """实现LayerNorm。其实PyTorch已经实现啦，见nn.LayerNorm。"""

    def __init__(self, features, epsilon=1e-6):
        """Init.

        Args:
            features: 就是模型的维度。论文默认512
            epsilon: 一个很小的数，防止数值计算的除0错误
        """
        super(LayerNorm, self).__init__()
        # alpha
        self.gamma = nn.Parameter(torch.ones(features))
        # beta
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        """前向传播.

        Args:
            x: 输入序列张量，形状为[B, L, D]
        """
        # 根据公式进行归一化
        # 在X的最后一个维度求均值，最后一个维度就是模型的维度
        mean = x.mean(-1, keepdim=True)
        # 在X的最后一个维度求方差，最后一个维度就是模型的维度
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        """初始化。

        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, torch.tensor(position_encoding, dtype=torch.float)))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        #max_len = torch.max(input_len)
        max_len = torch.tensor(50)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = []
        for len in input_len:
            valid_input_pos =[i for i in range(1,len+1)]
            for i in range(max_len - len):
                valid_input_pos.append(0)
            input_pos.append(valid_input_pos)
        input_pos_t = tensor(input_pos)
        #input_pos = tensor([list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos_t)

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class Encoder(nn.Module):
    """多层EncoderLayer组成Encoder。"""

    def __init__(self,
               vocab_size,
               max_seq_len,
               class_num,
               num_layers=1,
               model_dim=128,
               num_heads=8,
               ffn_dim=512,
               dropout=0.0):
        super(Encoder, self).__init__()

        self.class_num = class_num
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim
        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(in_features=self.max_seq_len*self.model_dim, out_features=1)
        self.dense_dropout = nn.Dropout(p=0.5)
        self.seq_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = PositionalEncoding(model_dim, self.max_seq_len)

    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        output = output.view(-1, self.max_seq_len*self.model_dim)
        try:
            output = self.dense(self.dense_dropout(output))
        except RuntimeError:
            print(output.size())
        output = self.sigmoid(output)
        return output, attentions

#transformer

class Trainer:
    def __init__(self,train_data, test_data,char_size, class_num):
        print("initing trainer...")
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = 32
        self.char_size = char_size
        self.class_num = class_num
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.test_loader = None
        self._setup_data()
        self.model = Encoder(vocab_size=self.char_size, max_seq_len=50,class_num=self.class_num).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def _setup_data(self):
        train_set, test_set = self.train_data, self.test_data
        train_sampler = RandomSampler(train_set)
        test_sampler = RandomSampler(test_set)

        self.train_loader = DataLoader(train_set, sampler=train_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_set, sampler=test_sampler,
                                      batch_size=self.batch_size, collate_fn=collate_fn)

    def evaluate(self, data, plot = False):
        self.model.eval()

        data_len = len(data)
        total_loss = 0.0
        y_true, y_pred = [], []

        for ix, (x, y, lens) in enumerate(data):
            # print('*', flush=True, end='')
            sys.stdout.write('\rEvaluating {} examples...'.format((ix + 1) * x.shape[0]))
            with torch.no_grad():
                lens_t = torch.tensor(lens)
                x, y, lens_t = x.to(self.device),  y.to(self.device), lens_t.to(self.device)
                outputs = self.model(x, lens_t)
                # print('output:', output.shape)
                output_0 = outputs[0].squeeze()
                losses = self.criterion(output_0, y)
                total_loss += losses.item() # .data[0]
                probs = output_0.data.cpu().numpy().tolist()
                # print(type(probs))
                preds = [1 if p >= 0.5 else 0 for p in probs]
                y_pred.extend(preds)
                # print(y.data.cpu().numpy().tolist()[:10])
                y_true.extend([int(i) for i in y.data.cpu().numpy().tolist()])
        update_loss = total_loss / data_len
        print('Test loss:', update_loss)
        print(metrics.classification_report(np.asarray(y_true), np.asarray(y_pred)))
        print()

        if plot == True:
            y_preds_t = torch.tensor(y_pred)
            y_true_t = torch.tensor(y_true)
            stacked = torch.stack((y_true_t,y_preds_t),dim=1)
            cmt = torch.zeros(10, 10, dtype=torch.int64)
            for p in stacked:
                tl, pl = p.tolist()
                cmt[tl, pl] = cmt[tl, pl] + 1

            plot_confusion_matrix(cmt,[i for i in range(10)],name="normal_transformer")

        return metrics.accuracy_score(np.asarray(y_true), np.asarray(y_pred))

    def train(self):
        best_acc = 0.0
        for epoch in range(500):
            total_loss = 0.0
            print("for epoch..." + str(epoch))
            self.model.train()
            for x, y_batch,lens in self.train_loader:
                #print('=', flush=True, end='')
                lens_t =  torch.tensor(lens)
                x, y_batch,lens_t = x.to(self.device), y_batch.to(self.device), lens_t.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x,lens_t)
                output_0 = outputs[0].squeeze()
                loss = self.criterion(output_0, y_batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print('\nSummary for Epoch {}'.format(epoch + 1))
            print('Train Loss:', total_loss / len(self.train_loader))
            print('Train Metrics')
            train_acc = self.evaluate(self.train_loader)
            print('Test Metrics')
            test_acc = self.evaluate(self.test_loader,plot=False)
            print()

            if test_acc > best_acc:
                print('Saving model...')
                best_acc = test_acc
                print(best_acc)
                #torch.save(self.model.state_dict(), os.path.join(self.model_path, 'conv_one_shot_model.pt'))



def main():
    f = open("dictionary.pkl", 'rb')
    char_dict = pickle.load(f)
    f.close()
    print("loading dict complete")
    char_size = len(char_dict.items())
    train_data, test_data, class_num = split_train_test()
    trainer = Trainer(train_data, test_data, char_size, class_num)
    trainer.train()

if __name__ == '__main__':
    main()