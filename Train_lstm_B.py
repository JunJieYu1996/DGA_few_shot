import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import pickle
import os
import sys
import numpy as np
import random

from sklearn import metrics

#from data_processor import *
from plot_drawer import  *


####
#2020.10.28
####

def split_train_test():
    door_number = 500
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

class My_LSTM(nn.Module):
    def __init__(self, vocab_size:int, class_num:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.embed_dim = 128
        self.hidden_dim = 128
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.dense_dropout = nn.Dropout(p=0.5)
        self.LSTM = nn.LSTM(self.embed_dim, self.hidden_dim)
        self.dense = nn.Linear(in_features=128, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lens):
        embedding = self.embed(x).permute(1, 0, 2) # -- only permute for conv
        x_embed = pack_padded_sequence(embedding, lens, enforce_sorted=False)
        # Run encoder
        _ , (final_hidden_state, final_cell_state) = self.LSTM(x_embed)

        x_dense = self.dense(self.dense_dropout(final_hidden_state))
        sigmoid_out = self.sigmoid(x_dense)

        return sigmoid_out

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
        self.model = My_LSTM(vocab_size=self.char_size, class_num=self.class_num).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _setup_data(self):
        train_set, test_set = self.train_data,self.test_data
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
                x, y = x.to(self.device), y.to(self.device)

                output = self.model(x, lens).squeeze()
                # print('output:', output.shape)
                losses = self.criterion(output, y)

                total_loss += losses.item() # .data[0]
                probs = output.data.cpu().numpy().tolist()
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

            plot_confusion_matrix(cmt,[i for i in range(10)], name="normal_lstm")

        return metrics.accuracy_score(np.asarray(y_true), np.asarray(y_pred))

    def train(self):
        best_acc = 0.0
        for epoch in range(500):
            total_loss = 0.0
            self.model.train()
            for x, y_batch, lens in self.train_loader:
                #print('=', flush=True, end='')
                x, y_batch = x.to(self.device), y_batch.to(self.device)
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
            test_acc = self.evaluate(self.test_loader, plot=False)
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
    train_data, test_data, dga_class_num = split_train_test()
    trainer = Trainer(train_data, test_data, char_size, dga_class_num + 1)
    trainer.train()

if __name__ == '__main__':
    main()