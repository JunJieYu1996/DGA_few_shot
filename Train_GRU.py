import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import pickle

import sys
import numpy as np

from sklearn import metrics

from data_processor import *
from plot_drawer import  *

#最简单的lstm多分类器，用10类dga，每类80个样本做训练，用剩下的20个做测试

class My_biGRU(nn.Module):
    def __init__(self, vocab_size:int, class_num:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.embed_dim = 128
        self.hidden_dim = 128
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.dense_dropout = nn.Dropout(p=0.5)
        self.GRU = nn.GRU(self.embed_dim,self.hidden_dim, bidirectional=True)
        self.dense = nn.Linear(in_features=256, out_features=self.class_num)
        self.softmax = nn.Softmax()

    def _encode_and_pool(self, x, encoder):
        i = encoder(x)
        return encoder(x)[0]

    def _encoder(self, x):
        for e_ix, e in enumerate(self.encoders):
            x = self._encode_and_pool(x=x, encoder=e)

        return x

    def forward(self, x, lens):
        embedding = self.embed(x).permute(1, 0, 2)
        x_embed = pack_padded_sequence(embedding, lens, enforce_sorted=False)
        # Run encoder
        outputs, final_hidden_state = self.GRU(x_embed)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        final_hidden_state_reshape = final_hidden_state.reshape(-1, 2 * self.hidden_dim)
        softmax_out = self.softmax(self.dense(self.dense_dropout(final_hidden_state_reshape )))
        #check = torch.sum(softmax_out,dim=1)
        return softmax_out

class Trainer:
    def __init__(self,train_data, test_data,char_size, class_num):
        print("initing trainer...")
        self.criterion = nn.CrossEntropyLoss()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = 32
        self.char_size = char_size
        self.class_num = class_num
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.test_loader = None
        self._setup_data()
        self.model = My_biGRU(vocab_size=self.char_size, class_num=self.class_num).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _setup_data(self):
        train_set,order_test_set = self.train_data,self.test_data
        test_set = order_test_set
        random.shuffle(test_set)
        test_1 = test_set[:int(len(test_set)*0.8)]
        test_2 = test_set[int(len(test_set)*0.8):]
        train_sampler = RandomSampler(test_1)
        test_sampler = RandomSampler(test_2)

        self.train_loader = DataLoader(test_1, sampler=train_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn_3)
        self.test_loader = DataLoader(test_2, sampler=test_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn_3)
    def evaluate(self, data,  plot = False):
        self.model.eval()

        data_len = len(data)
        total_loss = 0.0
        y_true, y_pred = [], []

        for ix, (x, y, lens) in enumerate(data):
            # print('*', flush=True, end='')
            sys.stdout.write('\rEvaluating {} examples...'.format((ix + 1) * x.shape[0]))
            with torch.no_grad():
                x, y = x.to(self.device), y.to(self.device,dtype=torch.int64)

                output = self.model(x, lens).squeeze()
                # print('output:', output.shape)
                losses = self.criterion(output, y)

                total_loss += losses.item() # .data[0]
                probs = output.data.cpu().numpy().tolist()
                # print(type(probs))
                preds = [p.index(max(p)) for p in probs]
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

            plot_confusion_matrix(cmt,[i for i in range(10)], name="normal_biGRU")

        return metrics.accuracy_score(np.asarray(y_true), np.asarray(y_pred))

    def train(self):
        best_acc = 0.0
        for epoch in range(500):
            total_loss = 0.0
            self.model.train()
            for x, y_batch, lens in self.train_loader:
                #print('=', flush=True, end='')
                x, y_batch= x.to(self.device), y_batch.to(self.device,dtype=torch.int64)
                self.optimizer.zero_grad()
                outputs = self.model(x, lens).squeeze()
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print('\nSummary for Epoch {}'.format(epoch + 1))
            print('Train Loss:', total_loss / len(self.train_loader))
            print('Train Metrics')
            train_acc = self.evaluate(self.train_loader)
            print('Test Metrics')
            test_acc = self.evaluate(self.test_loader, plot=True)
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
    train_data, test_data, class_num = split_train_test_2()
    trainer = Trainer(train_data, test_data, char_size, class_num)
    trainer.train()

if __name__ == '__main__':
    main()