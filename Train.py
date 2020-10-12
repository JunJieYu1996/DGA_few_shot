from data_processor import *
from LSTM_models import *
from models import *

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import DataLoader, RandomSampler

from sklearn import metrics


class Trainer:
    def __init__(self,train_data, test_data,char_size):
        print("initing trainer...")
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = 32
        self.n_epochs = 10
        self.char_size = char_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.test_loader = None
        self._setup_data()
        self.model = SiameseNet_LSTM(vocab_size=self.char_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _setup_data(self):
        train_set = Pair_Loader(self.train_data, base_num=len(self.train_data))
        test_set = Pair_Loader(self.test_data,base_num=len(self.test_data))
        print('Train label distribution:')
        train_set.get_label_freq()
        print('Test label distribution:')
        test_set.get_label_freq()
        train_sampler = RandomSampler(train_set)
        test_sampler = RandomSampler(test_set)
        self.train_loader = DataLoader(train_set, sampler=train_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_set, sampler=test_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn)
    def evaluate(self, data):
        self.model.eval()

        data_len = len(data)
        total_loss = 0.0
        y_true, y_pred = [], []

        for ix, (x_left, x_right, y) in enumerate(data):
            # print('*', flush=True, end='')
            sys.stdout.write('\rEvaluating {} examples...'.format((ix + 1) * x_left.shape[0]))
            with torch.no_grad():
                x_left, x_right, y = x_left.to(self.device), x_right.to(self.device), y.to(self.device)

                output = self.model(x_left, x_right).squeeze()
                # print('output:', output.shape)
                losses = self.criterion(output, y)

                total_loss += losses.item() # .data[0]
                probs = torch.sigmoid(output).data.cpu().numpy().tolist()
                # print(type(probs))
                preds = [1 if p >= 0.5 else 0 for p in probs]
                y_pred.extend(preds)
                # print(y.data.cpu().numpy().tolist()[:10])
                y_true.extend([int(i) for i in y.data.cpu().numpy().tolist()])

        update_loss = total_loss / data_len
        print('Test loss:', update_loss)
        print(metrics.classification_report(np.asarray(y_true), np.asarray(y_pred)))
        print()

        return metrics.accuracy_score(np.asarray(y_true), np.asarray(y_pred))

    def train(self):
        best_acc = 0.0
        for epoch in range(100):
            total_loss = 0.0
            print("for epoch..." + str(epoch))
            self.model.train()
            for x_left, x_right, y_batch in self.train_loader:
                #print('=', flush=True, end='')
                x_left, x_right, y_batch = x_left.to(self.device), x_right.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x_left, x_right).squeeze()
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print('\nSummary for Epoch {}'.format(epoch + 1))
            print('Train Loss:', total_loss / len(self.train_loader))
            print('Train Metrics')
            train_acc = self.evaluate(self.train_loader)
            print('Test Metrics')
            test_acc = self.evaluate(self.test_loader)
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
    char_size = len(char_dict.items())
    train_data, test_data, _ = split_train_test_2()
    trainer = Trainer(train_data, test_data, char_size)
    trainer.train()

if __name__ == '__main__':
    main()