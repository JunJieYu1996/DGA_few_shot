import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet_LSTM(nn.Module):
    def __init__(self, vocab_size:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = 200
        self.hidden_dim = 256
        self.hidden_layers = 3
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_dropout = nn.Dropout(p=0.5)
        self.dense_dropout = nn.Dropout(p=0.5)
        #0.5
        #bn
        self.encoders = nn.ModuleList([nn.LSTM(self.embed_dim,self.hidden_dim)])
        self.pool = nn.AvgPool1d(1)
        self.dense_hidden = nn.Linear(in_features=256, out_features=512)
        self.dense_out = nn.Linear(in_features=512, out_features=1)

    def _encode_and_pool(self, x, encoder):
        i = encoder(x)
        return torch.relu(encoder(x)[0])

    def _encoder(self, x):
        for e_ix, e in enumerate(self.encoders):
            x = self._encode_and_pool(x=x, encoder=e)

        return x

    def forward(self, x_left, x_right):
        x_left_embed = self.embed_dropout(self.embed(x_left)).permute(1, 0, 2) # -- only permute for conv
        x_right_embed = self.embed_dropout(self.embed(x_right)).permute(1, 0, 2) # -- only permute for conv

        # Run encoder
        x_left = self._encoder(x_left_embed)
        x_right = self._encoder(x_right_embed)

        # Pooling
        x_right = x_right.max(0)[0]
        x_left = x_left.max(0)[0]
        # print('x_right pooled:', x_right.shape)
        # print('x_left pooled:', x_left.shape)

        # final_encoded = x_left - x_right
        minus = x_left - x_right
        # plus = x_left + x_right
        # final_encoded = torch.cat([torch.abs(minus), plus], 1)
        final_encoded = torch.abs(minus)

        hidden_out = torch.relu(self.dense_hidden(final_encoded))
        logits_out = self.dense_out(self.dense_dropout(hidden_out))

        return logits_out




# Run it
if __name__ == '__main__':
    net = SiameseNet_LSTM(62)
    print(net)

    x_left = torch.zeros(size=(1, 34), dtype=torch.long)
    x_right = torch.zeros(size=(1, 34), dtype=torch.long)

    out = net(x_left, x_right)
    print(out.shape)
    # pool of square window of size=3, stride=2
    # m = nn.AvgPool1d(2)
    # input = torch.ones(4, 4,4)
    # input[2][2][2]=0
    # print(input)
    # output = m(input)
    # print(output)
