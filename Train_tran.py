from data_processor import *
from Transformer import *
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
        self.model = Encoder(vocab_size=self.char_size, max_seq_len=50,class_num=self.class_num).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def _setup_data(self):
        train_set, order_test_set = self.train_data, self.test_data
        test_set = order_test_set
        random.shuffle(test_set)
        test_1 = test_set[:int(len(test_set) * 0.8)]
        test_2 = test_set[int(len(test_set) * 0.8):]
        train_sampler = RandomSampler(test_1)
        test_sampler = RandomSampler(test_2)

        self.train_loader = DataLoader(test_1, sampler=train_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn_3)
        self.test_loader = DataLoader(test_2, sampler=test_sampler,
                                      batch_size=self.batch_size, collate_fn=collate_fn_3)

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
                x, y, lens_t = x.to(self.device),  y.to(self.device,dtype=torch.int64), lens_t.to(self.device)
                outputs = self.model(x, lens_t)
                # print('output:', output.shape)
                output_0 = outputs[0]
                losses = self.criterion(output_0, y)
                total_loss += losses.item() # .data[0]
                probs = torch.softmax(output_0, dim=1).data.cpu().numpy().tolist()
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
                x, y_batch,lens_t = x.to(self.device), y_batch.to(self.device,dtype=torch.int64), lens_t.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x,lens_t)
                output_0 = outputs[0]
                loss = self.criterion(output_0, y_batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print('\nSummary for Epoch {}'.format(epoch + 1))
            print('Train Loss:', total_loss / len(self.train_loader))
            print('Train Metrics')
            train_acc = self.evaluate(self.train_loader)
            print('Test Metrics')
            test_acc = self.evaluate(self.test_loader,plot=True)
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